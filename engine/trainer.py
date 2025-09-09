# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils.set import *
from utils.loss import *
from torch.autograd import Variable
import time
from torch.nn.utils import clip_grad_norm_
from thop import profile
from thop import clever_format

def do_train_with_base(
    cfg,
    model,
    train_loader, 
    optimizer,
    id_criterion,
    tri_criterion,
    epoch,
    report_interval
):
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    correct = 0
    total = 0

    print('\n==> Training Epoch #{}'.format(epoch))
    for batch_idx, (train_data, label, camid, _) in enumerate(train_loader):
        # split RGB and IR data
        rgb_data = Variable(train_data[0].cuda())
        ir_data = Variable(train_data[1].cuda())

        label = torch.cat([label] * 2, dim=0)
        label = Variable(label.cuda())

        batch_size = cfg.SOLVER.IMGS_PER_BATCH

        # model forward
        feat, out = model(rgb_data, ir_data)

        # calculate id loss
        loss_id = id_criterion(out, label)
        
        # calculate tri loss
        loss_tri, batch_acc = tri_criterion(feat, label)
        
        # calculate accuracy
        correct += (batch_acc / 2)
        total += batch_size * 2
        _, predicted = out.max(1)
        correct += (predicted.eq(label).sum().item() / 2)

        # Overall loss
        loss = loss_id + loss_tri

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # log different loss components
        train_loss.update(loss.item(), batch_size * 2)
        id_loss.update(loss_id.item(), batch_size * 2)
        tri_loss.update(loss_tri.item(), batch_size * 2)
        
        if (batch_idx + 1) % report_interval == 0:
            sys.stdout.write('\r')
            sys.stdout.write(
                '| Epoch: [%3d/%3d]  Iter: [%3d/%3d]  Loss: %.4f  Id Loss: %.4f  Tri Loss: %.4f  Acc: %.4f'
                %(
                    epoch, 
                    cfg.SOLVER.MAX_EPOCHS,
                    batch_idx + 1,
                    len(train_loader), 
                    train_loss.avg,
                    id_loss.avg,
                    tri_loss.avg,
                    100. * correct / total
                )
            )

class BaseTrainer(object):
    def __init__(self, model, criterion1, criterion2, criterion3, criterion_tri, clip_value=16.0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.criterion3 = criterion3
        self.criterion_tri = criterion_tri
        self.criterion_div = nn.MSELoss(reduce=True, size_average=True)
        self.criterion_ce = nn.CrossEntropyLoss()
        self.clip_value = clip_value

    def train(self, cfg, epoch, max_epoch, data_loader, optimizer, report_interval):
        # Creates once at the beginning of training
        scaler = torch.cuda.amp.GradScaler()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses1 = AverageMeter()
        losses2 = AverageMeter()
        losses3 = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for batch_idx, inputs in enumerate(data_loader):
            # self.model.eval()
            self.model.train()
            self.criterion1.train()
            self.criterion2.train()
            self.criterion3.train()

            data_time.update(time.time() - end)
            inputs, targets = self._parse_data(inputs)

            optimizer.zero_grad()

            # Casts operations to mixed precision
            ############################  step1  ################################
            # with torch.cuda.amp.autocast():
            #     rgb_data = inputs[0].cuda()
            #     ir_data = inputs[1].cuda()
            #     batch_size = rgb_data.size(0)
            #     out1, out2, out3, x1_pool, x2_pool, x3_pool, f1, f2, f3 = self.model(rgb_data, ir_data, block=[8, 8, 0, 0], modal=0)
            #     feat1, feat2 = torch.chunk(out1, 2, 0)
            #     loss_id1, acc1 = self.criterion1(feat1, feat2, targets)
            #     loss_tri1, _ = self.criterion_tri(x1_pool, torch.cat((targets, targets), 0))
            #     finite_mask = loss_id1.isfinite()
            #     if finite_mask.any():
            #         loss_id1 = loss_id1[finite_mask].mean()
            #         acc1 = acc1[finite_mask].mean()
            #     else:
            #         loss_id1 = acc1 = None
            #     loss_div1 = self.criterion_div(f1[:batch_size], f1[batch_size:])
            #     w = loss_id1.item() / loss_div1.item()

            # if loss_id1 is None:
            #     continue
            
            # loss1 = loss_id1 + loss_tri1 + loss_div1 * w * 0.01

            # if self.clip_value > 0:
            #     # Scales the loss, and calls backward() to create scaled gradients
            #     scaler.scale(loss1).backward()
            #     # Unscales the gradients of optimizer's assigned params in-place
            #     scaler.unscale_(optimizer)
            # else:
            #     loss1.backward()

            # clip_grad_norm_(self.model.parameters(), self.clip_value)
            # clip_grad_norm_(self.criterion1.parameters(), self.clip_value)

            # if self.clip_value > 0:
            #     # Unscales gradients and calls or skips optimizer.step()
            #     scaler.step(optimizer)
            #     # Updates the scale for next iteration
            #     scaler.update()
            # else:
            #     optimizer.step()

            ############################  step2  ################################
            with torch.cuda.amp.autocast():
                rgb_data = inputs[0].cuda()
                ir_data = inputs[1].cuda()
                batch_size = rgb_data.size(0)
                out1, out2, out3, x1_pool, x2_pool, x3_pool, result1, result2, result3, f1, f2, f3 = self.model(rgb_data, ir_data, block=[4, 4, 4, 0], modal=0)
                feat1, feat2 = torch.chunk(out2, 2, 0)
                loss_id2, acc2 = self.criterion2(feat1, feat2, targets)
                loss_tri2, _ = self.criterion_tri(x2_pool, torch.cat((targets, targets), 0))
                loss_ce2 = self.criterion_ce(result2, torch.cat((targets, targets), 0))
                finite_mask = loss_id2.isfinite()
                if finite_mask.any():
                    loss_id2 = loss_id2[finite_mask].mean()
                    acc2 = acc2[finite_mask].mean()
                else:
                    loss_id2 = acc2 = None
                loss_div2 = self.criterion_div(f2[:batch_size], f2[batch_size:])
                w = loss_id2.item() / loss_div2.item()

            if loss_id2 is None:
                continue
            
            loss2 = cfg.SOLVER.LAMBDA_1 * loss_id2 + \
                cfg.SOLVER.LAMBDA_2 * (cfg.SOLVER.LAMBDA_A * loss_tri2 + loss_ce2) + \
                cfg.SOLVER.LAMBDA_3 * loss_div2 * w
            
            if self.clip_value > 0:
                # Scales the loss, and calls backward() to create scaled gradients
                scaler.scale(loss2).backward()
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
            else:
                loss2.backward()

            clip_grad_norm_(self.model.parameters(), self.clip_value)
            clip_grad_norm_(self.criterion2.parameters(), self.clip_value)

            if self.clip_value > 0:
                # Unscales gradients and calls or skips optimizer.step()
                scaler.step(optimizer)
                # Updates the scale for next iteration
                scaler.update()
            else:
                optimizer.step()

            ############################  step3  ################################
            with torch.cuda.amp.autocast():
                rgb_data = inputs[0].cuda()
                ir_data = inputs[1].cuda()
                batch_size = rgb_data.size(0)
                out1, out2, out3, x1_pool, x2_pool, x3_pool, result1, result2, result3, f1, f2, f3 = self.model(rgb_data, ir_data, block=[2, 2, 2, 2], modal=0)
                feat1, feat2 = torch.chunk(out3, 2, 0)
                loss_id3, acc3 = self.criterion3(feat1, feat2, targets)
                loss_ce3 = self.criterion_ce(result3, torch.cat((targets, targets), 0))
                loss_tri3, _ = self.criterion_tri(x3_pool, torch.cat((targets, targets), 0))
                finite_mask = loss_id3.isfinite()
                if finite_mask.any():
                    loss_id3 = loss_id3[finite_mask].mean()
                    acc3 = acc3[finite_mask].mean()
                else:
                    loss_id3 = acc3 = None
                loss_div3 = self.criterion_div(f3[:batch_size], f3[batch_size:])
                w = loss_id3.item() / loss_div3.item()

            if loss_id3 is None:
                continue
            
            loss3 = cfg.SOLVER.LAMBDA_1 * loss_id3 + \
                cfg.SOLVER.LAMBDA_2 * (cfg.SOLVER.LAMBDA_A * loss_tri3 + loss_ce3) + \
                cfg.SOLVER.LAMBDA_3 * loss_div3 * w
            
            if self.clip_value > 0:
                # Scales the loss, and calls backward() to create scaled gradients
                scaler.scale(loss3).backward()
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
            else:
                loss3.backward()

            clip_grad_norm_(self.model.parameters(), self.clip_value)
            clip_grad_norm_(self.criterion3.parameters(), self.clip_value)

            if self.clip_value > 0:
                # Unscales gradients and calls or skips optimizer.step()
                scaler.step(optimizer)
                # Updates the scale for next iteration
                scaler.update()
            else:
                optimizer.step()

            ############################  all  ################################
            # loss = loss1 + loss2 + loss3
            loss = loss2 + loss3
            losses.update(loss.item(), 2 * targets.size(0))
            # losses1.update(loss1.item(), 2 * targets.size(0))
            losses2.update(loss2.item(), 2 * targets.size(0))
            losses3.update(loss3.item(), 2 * targets.size(0))
            precisions.update(acc2.item(), 2 * targets.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            
            # with torch.no_grad():
            #     rgb_data = inputs[0].cuda()
            #     ir_data = inputs[1].cuda()
            #     original_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            #     original_model = original_model.cuda()
            #     # 只计算一次前向传播的FLOPs
            #     if batch_idx == 0:
            #         flops, _ = profile(original_model, inputs=(rgb_data, ir_data, 0, [2, 2, 2, 2]))
            #         # flops = clever_format([flops], "%.3f")
            #         print(f"FLOPs per sample: {flops}")

            if (batch_idx + 1) % report_interval == 0:
                sys.stdout.write('\r')
                sys.stdout.write(
                    '| Epoch: [%3d/%3d]  Iter: [%3d/%3d]  Loss: %.4f  Loss1: %.4f  Loss2: %.4f  Loss3: %.4f  Acc: %.4f  Time: %.4f'
                    %(
                        epoch, 
                        max_epoch,
                        batch_idx + 1,
                        len(data_loader), 
                        losses.avg,
                        losses1.avg,
                        losses2.avg,
                        losses3.avg,
                        precisions.avg,
                        batch_time.avg,
                    )
                )

        return losses.avg, precisions.avg

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, pids, _, _ = inputs
        inputs = imgs
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, inputs, targets):
        raise NotImplementedError

def do_train_with_porsche(
    cfg,
    model,
    train_loader, 
    optimizer,
    criterion,
    criterion_tri,
    epoch,
    report_interval
):
    criterion1, criterion2, criterion3 = criterion
    criterion1.train()
    criterion2.train()
    criterion3.train()

    trainer = Trainer(model, criterion1, criterion2, criterion3, criterion_tri, clip_value=16)
    
    print('\n==> Training Epoch #{}'.format(epoch))
    loss, acc = trainer.train(cfg, epoch, cfg.SOLVER.MAX_EPOCHS, train_loader, optimizer, report_interval)
    lr = list(map(lambda group: group['lr'], optimizer.param_groups))
    sys.stdout.write('\r')
    sys.stdout.write('\n| Finished epoch %d at lr=[Vis module: %g, The module: %g, Base module:%g, Trans Enc: %g, Matcher1: %g, Matcher2: %g, Matcher3: %g] AVG_Loss: %.4f  AVG_Acc: %.4f%%' % (epoch, lr[0], lr[1], lr[2], lr[3], lr[4], lr[5], lr[6], loss, 100 * acc))