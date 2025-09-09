import argparse
import sys
import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.autograd import Variable

sys.path.append('.')

from config import cfg
from utils.set import *
from utils.time import get_hms
from utils.logger import setup_logger
from utils.eval_reid import eval_func
from utils.serialization import save_checkpoint, load_checkpoint
from data.build_dataloader import make_data_loader
from model import build_model
from solver.build import make_optimizer, make_optimizer_with_center
from solver.lr_scheduler import WarmupMultiStepLR
from utils.loss import OriTripletLoss, L2NormSimilarity, TripletLoss_WRT, PairwiseMatchingLoss
from engine.trainer import *
from engine.inference import *


from model.backbones.transmatcher import TransMatcher

def eval(cfg, epoch, dataset, query_loader, gallery_loader, model, matcher, num_query, num_gallery):
    # set model as evaluating mode
    model.eval()
    batch_size = cfg.TEST.IMGS_PER_BATCH
    
    if cfg.SOLVER.METHOD in ['base', 'agw']:
        cmc, mAP = inference_with_base(cfg, model, query_loader, gallery_loader)
    elif cfg.SOLVER.METHOD in ['porsche']:
        cmc, mAP = inference_with_porsche(cfg, model, dataset, num_query, num_gallery, matcher, query_loader, gallery_loader)
    else:
        raise KeyError("Unsupported method: {}".format(cfg.SOLVER.METHOD))
    
    return cmc, mAP

def train(cfg, epoch, train_loader, model, optimizer, num_classes, criterion, use_cuda):
    # set model as training mode
    model.train()
    # adjust report interval
    if cfg.DATASETS.NAMES in ['msvr310', 'wmveid863']:
        report_interval = 10
    else:
        report_interval = 100

    if cfg.SOLVER.METHOD in ['base', 'agw']:
        id_criterion = nn.CrossEntropyLoss()
        tri_criterion = OriTripletLoss(batch_size=cfg.SOLVER.IMGS_PER_BATCH, margin=cfg.SOLVER.RANGE_MARGIN)
        if use_cuda:
            id_criterion = nn.DataParallel(id_criterion, device_ids=range(torch.cuda.device_count()))
            tri_criterion = nn.DataParallel(tri_criterion, device_ids=range(torch.cuda.device_count()))
        do_train_with_base(cfg, model, train_loader, optimizer, \
                           id_criterion, tri_criterion, epoch, report_interval)
    elif cfg.SOLVER.METHOD in ['porsche']:
        tri_criterion = TripletLoss_WRT()
        if use_cuda:
            tri_criterion = nn.DataParallel(tri_criterion, device_ids=range(torch.cuda.device_count()))
        do_train_with_porsche(cfg, model, train_loader, optimizer,\
                           criterion, tri_criterion, epoch, report_interval)
    else:
        raise KeyError("Unsupported method: {}".format(cfg.SOLVER.METHOD))

    
def main():
    print("========== Configs: {} ==========".format(args.config_file.split('/')[-1]))
    use_cuda = torch.cuda.is_available()
    setup_logger(cfg.OUTPUT_DIR)

    best_map = 0
    best_top1 = 0

    print("========== [Phase 1] Data Preparation: {} ==========".format(cfg.DATASETS.NAMES))
    dataset, train_loader, query_loader, gallery_loader, num_query, num_gallery, num_classes = make_data_loader(cfg, reconst=False)

    print("========== [Phase 2] Model Setup: {} ==========".format(cfg.SOLVER.METHOD))
    # prepare model
    model = build_model(cfg, num_classes)
    if use_cuda:
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    if cfg.SOLVER.METHOD == 'porsche':
        num_features = model.module.num_features
        matcher1 = TransMatcher(seq_len=16 * 32, d_model=num_features, num_decoder_layers=cfg.MODEL.NUM_TRANS_LAYERS, 
                                dim_feedforward=cfg.MODEL.D_FF).cuda()
        matcher2 = TransMatcher(seq_len=8 * 16, d_model=num_features, num_decoder_layers=cfg.MODEL.NUM_TRANS_LAYERS, 
                                dim_feedforward=cfg.MODEL.D_FF).cuda()
        matcher3 = TransMatcher(seq_len=4 * 8, d_model=num_features, num_decoder_layers=cfg.MODEL.NUM_TRANS_LAYERS,
                                dim_feedforward=cfg.MODEL.D_FF).cuda()
        criterion1 = PairwiseMatchingLoss(matcher1).cuda()
        criterion2 = PairwiseMatchingLoss(matcher2).cuda()
        criterion3 = PairwiseMatchingLoss(matcher3).cuda()
        criterion = [criterion1, criterion2, criterion3]
        matcher = [matcher1, matcher2, matcher3]
    else:
        criterion = matcher = None
    
    # prepare optimizer and scheduler
    if cfg.SOLVER.METHOD in ['porsche']:
        criterion1, criterion2, criterion3 = criterion
        visible_param_ids = set(map(id, model.module.visible_module.parameters()))
        thermal_param_ids = set(map(id, model.module.thermal_module.parameters()))
        base_param_ids = set(map(id, model.module.base_module.parameters()))

        new_params = [p for p in model.module.parameters() if id(p) not in visible_param_ids
                      and id(p) not in thermal_param_ids and id(p) not in base_param_ids]
        param_groups = [
            {'params': model.module.visible_module.parameters(), 'lr': cfg.SOLVER.BASE_LR},
            {'params': model.module.thermal_module.parameters(), 'lr': cfg.SOLVER.BASE_LR},
            {'params': model.module.base_module.parameters(), 'lr': cfg.SOLVER.BASE_LR},
            {'params': new_params, 'lr': cfg.SOLVER.BASE_LR},
            {'params': criterion1.matcher.parameters(), 'lr': cfg.SOLVER.BASE_LR}, 
            {'params': criterion2.matcher.parameters(), 'lr': cfg.SOLVER.BASE_LR},
            {'params': criterion3.matcher.parameters(), 'lr': cfg.SOLVER.BASE_LR}]
        optimizer = torch.optim.SGD(param_groups, lr=cfg.SOLVER.BASE_LR,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        
        # scheduler = MultiStepLR(optimizer, milestones=cfg.SOLVER.STEPS, gamma=cfg.SOLVER.GAMMA)
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                    cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    else:
        optimizer = make_optimizer(cfg, model)
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                    cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    print("========== [Phase 3] Training Model ==========")
    print('| Training Epochs: ' + str(cfg.SOLVER.MAX_EPOCHS))
    start_epoch = cfg.SOLVER.START_EPOCHS
    elapsed_time = 0
    meanAP = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top10 = AverageMeter()
    top20 = AverageMeter()

    # load checkpoint
    if cfg.SOLVER.RESUME:
        checkpoint_path = os.path.join(cfg.TEST.WEIGHT, 'checkpoint.pth.tar')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        print('| Load checkpoint from: {}'.format(checkpoint_path))
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = cfg.SOLVER.VAL_EPOCHS + 1

    val_epoch = cfg.SOLVER.VAL_EPOCHS
    
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        start_time = time.time()
        epoch1 = epoch + 1
        train(cfg, epoch1, train_loader, model, optimizer, num_classes, criterion, use_cuda)
        scheduler.step()
        if (epoch1 >= val_epoch) and (epoch1 % 5 == 0):
            print('\n| Evaluating Model!')
            test_time = time.time()
            cmc, mAP = eval(cfg, epoch1, dataset, query_loader, gallery_loader, model, matcher, num_query, num_gallery)
            print('| Evaluate Time = ' + str(time.time() - test_time))

            meanAP.update(mAP.item())
            top1.update(cmc[0].item())
            top5.update(cmc[4].item())
            top10.update(cmc[9].item())
            top20.update(cmc[19].item())
            best_top1 = max(best_top1, cmc[0].item())
            is_best = (mAP > best_map)
            best_map = max(mAP, best_map)

            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'criterion1': criterion1.state_dict(),
                'criterion2': criterion2.state_dict(),
                'criterion3': criterion3.state_dict(),
                'epoch': epoch1,
                'best_map': best_map,
                'best_top1': best_top1,
            }, is_best, fpath=os.path.join(cfg.OUTPUT_DIR, 'checkpoint.pth.tar'))
            print("| Validation Epoch #%d\tmAP: %.3f%%  top1: %.3f%%  top5: %.3f%%  top10: %.3f%%  top20: %.3f%%" 
                  % (epoch1, meanAP.val, top1.val, top5.val, top10.val, top20.val))
            epoch_time = time.time() - start_time
            elapsed_time += epoch_time
    print('\n| Elapsed Time = %d:%02d:%02d\n' % (get_hms(elapsed_time)))

    print('========== [Phase 4] Testing Model =========')
    print('* Test results : best_rank@1 = %.2f%%' % (best_top1))
    print('* Test results : best_mAP = %.2f%%' % (best_map))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Vehicle Re-ID Training')
    parser.add_argument("--config_file", default="./configs/porsche.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if args.config_file != "":
        cfg.set_new_allowed(is_new_allowed=True)
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    # Set gpu
    if cfg.SOLVER.GPU_ID is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.SOLVER.GPU_ID
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Set output direction
    if cfg.OUTPUT_DIR and not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    # Set random seed
    set_random_seed(cfg.SOLVER.RANDOM_SEED)
    
    main()