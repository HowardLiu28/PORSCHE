# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn as nn
import time
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

from utils.eval_reid import eval_func, evaluate_all
from utils.util import to_torch, to_numpy

def inference_with_base(
        cfg,
        model,
        query_loader,
        gallery_loader
):
    print('==> Extracting Query Feature...')
    start_q = time.time()
    with torch.no_grad():
        query_feats, q_pids, q_camids = [], [], []
        for batch_idx, (rgb_data, label, camid, _) in enumerate(query_loader):
            # query data only contains RGB data
            rgb_data = Variable(rgb_data.cuda())
            feat, feat_att = model(rgb_data, rgb_data, modal=cfg.SOLVER.MODE[0]) # modal=1 means only use rgb encoder
            query_feats.append(feat)
            q_pids.extend(np.asarray(label))
            q_camids.extend(np.asarray(camid))
        query_feats = torch.cat(query_feats, dim=0)
        query_feats = F.normalize(query_feats, dim=1, p=2)
    print('Extracting Time:\t {:.3f}'.format(time.time() - start_q))

    print('==> Extracting Gallery Feature...')
    start_g = time.time()
    with torch.no_grad():
        gallery_feats, g_pids, g_camids = [], [], []
        for batch_idx, (ir_data, label, camid, _) in enumerate(gallery_loader):
            # gallery data only contains IR data
            ir_data = Variable(ir_data.cuda())
            feat, feat_att = model(ir_data, ir_data, modal=cfg.SOLVER.MODE[1])   # model=2 means only use ir encoder
            gallery_feats.append(feat)
            g_pids.extend(np.asarray(label))
            g_camids.extend(np.asarray(camid))
        gallery_feats = torch.cat(gallery_feats, dim=0)
        gallery_feats = F.normalize(gallery_feats, dim=1, p=2)
    print('Extracting Time:\t {:.3f}'.format(time.time() - start_g))

    # compute the similarity
    distmat = torch.matmul(query_feats, torch.transpose(gallery_feats, 0, 1)).cpu().numpy()
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)

    # evaluation
    cmc, mAP = eval_func(-distmat, q_pids, g_pids, q_camids, g_camids)
    cmc = torch.from_numpy(cmc)
    mAP = torch.from_numpy(np.asarray(mAP))

    return cmc, mAP

def pairwise_distance(matcher, prob_fea, gal_fea, gal_batch_size=4, prob_batch_size=4096):
    with torch.no_grad():
        num_gals = gal_fea.size(0)
        num_probs = prob_fea.size(0)
        score = torch.zeros(num_probs, num_gals, device=prob_fea.device)
        matcher.eval()
        for i in range(0, num_probs, prob_batch_size):
            j = min(i + prob_batch_size, num_probs)
            matcher.make_kernel(prob_fea[i: j, :, :, :].cuda())
            for k in range(0, num_gals, gal_batch_size):
                k2 = min(k + gal_batch_size, num_gals)
                score[i: j, k: k2] = matcher(gal_fea[k: k2, :, :, :].cuda())
        # scale matching scores to make them visually more recognizable
        score = torch.sigmoid(score / 10)
    return (1. - score).cpu()  # [p, g]

def extract_cnn_feature(model, inputs, modal):
    model = model.cuda().eval()
    with torch.no_grad():
        output_1, output_2, output_3, _, _, _ = model(inputs, inputs, modal)
    return output_1, output_2, output_3


def extract_features(model, data_loader, modal, verbose=False):
    fea_time = 0
    data_time = 0
    features1 = OrderedDict()
    features2 = OrderedDict()
    features3 = OrderedDict()
    labels = OrderedDict()
    end = time.time()

    if verbose:
        print('Extract Features...', end='\t')

    for i, (imgs, pids, camid, fpaths) in enumerate(data_loader):
        data_time += time.time() - end
        end = time.time()
        start_infer = time.time()
        outputs = extract_cnn_feature(model, imgs, modal)
        infer_time = time.time() - start_infer
        per_sample_time = infer_time / len(fpaths)
        for fpath, output1, output2, output3, pid in zip(fpaths, outputs[0], outputs[1], outputs[2], pids):
            features1[fpath] = output1
            features2[fpath] = output2
            features3[fpath] = output3
            labels[fpath] = pid

        fea_time += time.time() - end
        end = time.time()

    if verbose:
        print('Feature time: {:.3f} seconds. Data time: {:.3f} seconds.'.format(fea_time, data_time))

    return features1, features2, features3, labels

class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, cfg, matcher, testset, query_loader, gallery_loader, gal_batch_size=4,
                 prob_batch_size=4096, tau=100, sigma=200, K=10, alpha=0.2):
        query = testset.query
        gallery = testset.gallery

        print('Compute similarity ...', end='\t')
        start = time.time()

        prob_fea1, prob_fea2, prob_fea3, _ = extract_features(self.model, query_loader, modal=cfg.SOLVER.MODE[0])
        # prob_fea1 = torch.cat([prob_fea1[f].unsqueeze(0) for f, _, _ in query], 0)
        prob_fea2 = torch.cat([prob_fea2[f].unsqueeze(0) for f, _, _ in query], 0)
        # prob_fea3 = torch.cat([prob_fea3[f].unsqueeze(0) for f, _, _ in query], 0)
        num_prob = len(query)
        num_gal = len(gallery)
        batch_size = gallery_loader.batch_size
        # dist1 = torch.zeros(num_prob, num_gal)
        dist2 = torch.zeros(num_prob, num_gal)
        # dist3 = torch.zeros(num_prob, num_gal)

        for i, (imgs, pids, camids, _) in enumerate(gallery_loader):
            gal_fea1, gal_fea2, gal_fea3 = extract_cnn_feature(self.model, imgs, modal=cfg.SOLVER.MODE[1]) 
            g0 = i * batch_size
            g1 = min(num_gal, (i + 1) * batch_size)
            # dist1[:, g0:g1] = pairwise_distance(matcher[0], prob_fea1, gal_fea1, batch_size, prob_batch_size)  # [p, g]
            dist2[:, g0:g1] = pairwise_distance(matcher[1], prob_fea2, gal_fea2, batch_size, prob_batch_size)
            # dist3[:, g0:g1] = pairwise_distance(matcher[2], prob_fea3, gal_fea3, batch_size, prob_batch_size)

        print('Time: %.3f seconds.' % (time.time() - start))
        
        # cmc1, mAP1 = evaluate_all(dist1, query=query, gallery=gallery)
        cmc2, mAP2 = evaluate_all(dist2, query=query, gallery=gallery)
        # cmc3, mAP3 = evaluate_all(dist3, query=query, gallery=gallery)

        cmc1, mAP1 = 0, 0
        cmc3, mAP3 = 0, 0

        return cmc1, mAP1, cmc2, mAP2, cmc3, mAP3


def inference_with_porsche(
        cfg,
        model,
        dataset,
        num_query,
        num_gallery,
        matcher,
        query_loader,
        gallery_loader
):
    batch_size = cfg.TEST.IMGS_PER_BATCH

    evaluator = Evaluator(model)

    cmc1, mAP1, cmc2, mAP2, cmc3, mAP3 = evaluator.evaluate(cfg, matcher, dataset, query_loader, gallery_loader, 
                                  gal_batch_size=batch_size, prob_batch_size=cfg.TEST.PROB_BS, 
                                  tau=cfg.TEST.TAU, sigma=cfg.TEST.SIGMA, K=cfg.TEST.K, alpha=cfg.TEST.ALPHA)
    # cmc = max(cmc1[0], cmc2[0], cmc3[0])
    # mAP = max(mAP1, mAP2, mAP3)
    return cmc2 * 100, mAP2 * 100