import time
import copy
import json
import logging
import random 
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils.evaluate import evaluate_generator, save_generator_evaluate_metric
from utils.util import AverageMeter, save, accuracy, min_max_normalize, bn_calibration
from utils.optim import cal_hc_loss


class Trainer:
    def __init__(self, criterion, g_optimizer, lookup_table, backbone_pool, LOW_FLOPS, HIGH_FLOPS):
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.losses = AverageMeter()
        self.hc_losses = AverageMeter()

        self.device = device

        self.criterion = criterion
        self.optimizer = optimizer
        self.g_optimizer = g_optimizer
        self.scheduler = scheduler


        self.epochs = 50
        self.print_freq = 100

        self.backbone_pool = backbone_pool
        # ==============
        self.low_macs = LOW_FLOPS
        self.high_macs = HIGH_FLOPS
        self.hardware_pool = [i for i in range(self.low_macs, self.high_macs, 5)]
        self.hardware_index = 0
        random.shuffle(self.hardware_pool)
        # ==============

        self.alpha = 0.0002
        self.loss_penalty = 1.2

        self.lookup_table = lookup_table

        self.tau_decay = 0.95

        self.path_to_generator_eval = "./generator_evaluate.csv"
        self.path_to_save_generator =  "./best_generator.pth"
        self.path_to_best_avg_generator = "./best_avg.pth"
        self.path_to_fianl_generator = "./final.pth"

    def search_train_loop(self, val_loader, model, generator):
        tau = 5
        best_top1 = 0.0
        for epoch in range(self.epochs):
            logging.info("Start to train for search epoch {}".format(epoch))
            logging.info("Tau: {}".format(tau))
            self._generator_training_step(generator, model, val_loader, epoch, tau, info_for_logger="_gen_train_step")

            top1_avg, _ = self.generator_validate(generator, model, val_loader, epoch, tau, sample=True, info_for_logger="_gen_val_step_")


            evaluate_metric, total_loss, kendall_tau = evaluate_generator(generator, self.backbone_pool, self.lookup_table, self.low_macs, self.high_macs, self.alpha, self.loss_penalty)

            logging.info("Total loss : {}".format(total_loss))
            if best_loss > total_loss:
                logging.info("Best loss by now: {} Tau : {}.Save model".format(total_loss, kendall_tau))
                best_loss = total_loss
                save_generator_evaluate_metric(evaluate_metric, self.path_to_generator_eval)
                save(generator, self.g_optimizer, self.path_to_save_generator)
            if top1_avg > best_top1 and total_loss < 0.4:
                logging.info("Best top1-avg by now: {}.Save model".format(top1_avg))
                best_top1 = top1_avg
                save(generator, self.g_optimizer, self.path_to_best_avg_generator)
            save(generator, self.g_optimizer, "./logs/generator/{}.pth".format(total_loss))
            tau *= self.tau_decay
        logging.info("Best loss: {}".format(best_loss))
        save(generator, self.g_optimizer, self.path_to_fianl_generator)


    def _get_arch_param(self, generator, hardware_constraint=None, valid=False):
        # ====================== Strict fair sample
        if hardware_constraint is None:
            hardware_constraint = torch.tensor(self.hardware_pool[self.hardware_index]+random.random()-0.5, dtype=torch.float32).view(-1, 1)
            #hardware_constraint = torch.tensor(self.hardware_pool[self.hardware_index], dtype=torch.float32).view(-1, 1)
            self.hardware_index += 1
            if self.hardware_index == len(self.hardware_pool):
                self.hardware_index = 0
                random.shuffle(self.hardware_pool)
        else:
            hardware_constraint = torch.tensor(hardware_constraint, dtype=torch.float32).view(-1, 1)
        # ======================

        hardware_constraint = hardware_constraint.cuda()
        logging.info("Target macs : {}".format(hardware_constraint.item()))

        backbone = self.backbone_pool.get_backbone(hardware_constraint.item())
        backbone = backbone.cuda()

        normalize_hardware_constraint = min_max_normalize(self.high_macs, self.low_macs, hardware_constraint)

        noise = torch.randn(*backbone.shape)
        noise = noise.cuda()
        noise *= self.noise_weight

        arch_param = generator(backbone, normalize_hardware_constraint, noise)

        return hardware_constraint, arch_param

    def set_arch_param(self, generator, model, hardware_constraint=None, arch_param=None, tau=None):
        """Sample the sub-network from supernet by arch_param(generate from generator or user specific)
        """
        if tau is not None:
            hardware_constraint, arch_param = self._get_arch_param(generator)
            arch_param = F.gumbel_softmax(arch_param, dim=-1, tau=tau)

        arch_param = arch_param.cuda()
        model.module.set_arch_param(arch_param)

        return arch_param, hardware_constraint
    
    def _generator_training_step(self, generator, model, loader, epoch, tau, info_for_logger=""):
        start_time = time.time()
        generator.train()
        model.eval()

        for step, (X, y) in enumerate(loader):
            arch_param, hardware_constraint = self.set_arch_param(generator, model, tau=tau)

            macs = self.lookup_table.get_model_macs(arch_param)
            logging.info("Generate model macs : {}".format(macs))

            hc_loss = cal_hc_loss(macs.cuda(), hardware_constraint.item(), self.alpha, self.loss_penalty)

            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            N = X.shape[0]

            self.g_optimizer.zero_grad()
            outs = model(X, True)

            ce_loss = self.criterion(outs, y)
            loss = ce_loss + hc_loss
            logging.info("HC loss : {}".format(hc_loss))
            loss.backward()

            self.g_optimizer.step()
            self.g_optimizer.zero_grad()

            self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Train", hc_losses=hc_loss)
        self._epoch_stats_logging(start_time=start_time, epoch=epoch, val_or_train="train")
        for avg in [self.top1, self.top5, self.losses, self.hc_losses]:
            avg.reset()

    def generator_validate(self, generator, model, loader, epoch, tau, hardware_constraint=360, arch_param=None, sample=False, info_for_logger=""):
        if generator is not None:
            generator.eval()
        model.eval()
        start_time = time.time()

        if sample:
            hardware_constraint, arch_param = self._get_arch_param(generator, hardware_constraint, valid=True)
            arch_param = self.lookup_table.get_validation_arch_param(arch_param)
            arch_param, hardware_constraint = self.set_arch_param(generator, model, hardware_constraint=hardware_constraint, arch_param=arch_param)
        else:
            hardware_constraint = torch.tensor(hardware_constraint)
            hardware_constraint.cuda()
            
        macs = self.lookup_table.get_model_macs(arch_param)
        logging.info("Generate model macs : {}".format(macs))

        hc_loss = cal_hc_loss(macs.cuda(), hardware_constraint.item(), self.alpha, self.loss_penalty)

        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
                N = X.shape[0]

                outs = model(X, True)
                loss = self.criterion(outs, y)

                self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Valid", hc_losses=hc_loss)

        top1_avg = self.top1.get_avg()
        self._epoch_stats_logging(start_time=start_time, epoch=epoch, val_or_train="val")
        self.writer.add_scalar("train_vs_val/"+"val"+"_hc_", macs, epoch)
        for avg in [self.top1, self.top5, self.losses, self.hc_losses]:
            avg.reset()

        return top1_avg, macs.item()

    def _epoch_stats_logging(self, start_time, epoch, val_or_train, info_for_logger=""):
        top1_avg = self.top1.get_avg()
        logging.info(info_for_logger+val_or_train+":[{:3d}/{}] Final Prec@1 {:.4%} Time {:.2f}".format(epoch+1, self.epochs, top1_avg, time.time()-start_time))

    def _intermediate_stats_logging(self, outs, y, loss, step, epoch, N, len_loader, val_or_train, hc_losses=None):
        prec1, prec5 = accuracy(outs, y, topk=(1, 5))
        self.losses.update(loss.item(), N)
        self.top1.update(prec1.item(), N)
        self.top5.update(prec5.item(), N)

        if hc_losses is not None:
            self.hc_losses.update(hc_losses.item(), 1)

        if (step > 1 and step % self.print_freq==0) or step == len_loader -1 :
            logging.info(val_or_train+
                    ":[{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} HC Loss {:.3f}"
                    "Prec@(1, 3) ({:.1%}, {:.1%})".format(
                        epoch+1, self.epochs, step, len_loader-1, self.losses.get_avg(), self.hc_losses.get_avg(),
                        self.top1.get_avg(), self.top5.get_avg()
                        ))

