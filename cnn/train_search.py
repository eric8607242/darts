import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect

from donas_utils.generator import get_generator
from donas_utils.backbone_pool import BackbonePool
from donas_utils.lookup_table import LookUpTable
from donas_utils.util import min_max_normalize
from donas_utils.optim import cal_hc_loss


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=0.001, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10
ALPHA = 0.0001
LOSS_PENALTY = 1.2


def main():
  path_to_best_loss_eval = "./generator/best_loss_model_{}.csv".format(args.seed)
  path_to_best_model = "./generator/best_model_{}.pth".format(args.seed)

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  # ================= DONAS ==========================
  low_flops = args.low_flops
  high_flops = args.high_flops

  nodes, edges = model.get_arch_param_nums()
  lookup_table = LookUpTable(edges, nodes)
  arch_param_nums = nodes * edges

  generator = get_generator(20)
  generator = generator.cuda()
  backbone_pool = BackbonePool(nodes, edges, lookup_table, arch_param_nums)
  backbone = backbone_pool.get_backbone((low_flops+high_flops)/2)

  g_optimizer = torch.optim.Adam(generator.parameters(),
                                 weight_decay=0,
                                 lr=0.001,
                                 betas=(0.5, 0.999))

  tau = 5
  best_hc_loss = 100000

  # ================= DONAS ==========================

  architect = Architect(model, generator, args)


  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, low_flops, high_flops, backbone, tau)
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion, generator, backbone, (low_flops+high_flops)//2, lookup_table)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))

    evalulate_metric, total_loss, kendall_tau = evalulate_generator(generator, backbone, lookup_table, low_flops, high_flops)
    if total_loss < best_hc_loss:
        logger.log("Best hc loss : {}. Save model!".format(total_loss))
        save_generator_evaluate_metric(evalulate_metric, path_to_best_loss_eval)
        best_hc_loss = total_loss

    if valid_acc > best_top1:
        logger.log("Best top1-avg : {}. Save model!".format(valid_acc_top1))
        save_model(generator, path_to_best_model)
        best_top1 = valid_acc

    tau *= 0.95


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, low_flops, high_flops, backbone, tau):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  hardware_pool = [i for i in range(low_flops, high_flops, 5)] 
  hardware_index = 0

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(async=True)

    target_hc = torch.tensor(hardware_pool[hardware_index]+3*(random.random()-0.5), dtype=torch.float32).view(-1, 1)
    target_hc = target_hc.cuda()
    logger.info("Target hc : {}".foramt(target_hc.item()))

    backbone = backbone.cuda()
    normalalize_target_hc = min_max_normalize(high_flops, low_flops, target_hc)
    arch_param = generator(backbone, normalalize_target_hc)
    arch_param = arch_param.reshape(-1, arch_param.size(-1))
    alphas_normal = F.gumbel_softmax(arch_param[0], dim=-1, tau)
    alphas_reduce = F.gumbel_softmax(arch_param[1], dim=-1, tau)

    gen_hc = lookup_table.get_model_macs(alphas_normal, alphas_reduce)
    logger.info("Generator hc : {}".format(gen_hc))

    hc_loss = cal_hc_loss(gen_hc.cuda(), target_hc.item(), ALPHA, LOSS_PENALTY)

    hardware_index += 1
    if hardware_index == len(hardware_pool):
        hardware_index = 0
        random.shuffle(hardware_pool)

    #architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
    self.g_optimizer.zero_grad()
    g_loss = self.model._loss(input_valid, target_valid)
    loss = g_loss + hc_loss
    g_loss.backward()
    self.g_optimizer.step()

    # =========================================================================

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion, generator, backbone, target_hc):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  target_hc = torch.tensor(target_hc, dtype=torch.float32).view(-1, 1)
  target_hc = target_hc.cuda()
  logger.info("Target hc : {}".foramt(target_hc.item()))

  backbone = backbone.cuda()
  normalalize_target_hc = min_max_normalize(high_flops, low_flops, target_hc)
  arch_param = generator(backbone, normalalize_target_hc)
  arch_param = arch_param.reshape(-1, arch_param.size(-1))
  alphas_normal = lookup_table.get_validation_arch_param(arch_param[0])
  alphas_reduce = lookup_table.get_validation_arch_param(arch_param[1])

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

def evalulate_generator(generator, backbone, lookup_table, low_flops, high_flops):
  generator.eval()
  total_loss = 0

  evalulate_metric = {"gen_hc":[], "target_hc":[]}
  for target_hc in range(low_flops, high_flops, 5):
    target_hc = torch.tensor(target_hc, dtype=torch.float32).view(-1, 1)
    target_hc = target_hc.cuda()

    backbone = backbone.cuda()

    normalalize_target_hc = min_max_normalize(high_flops, low_flops, target_hc)

    arch_param = generator(backbone, normalalize_target_hc)
    arch_param = arch_param.reshape(-1, arch_param.size(-1))
    alphas_normal = lookup_table.get_validation_arch_param(arch_param[0])
    alphas_reduce = lookup_table.get_validation_arch_param(arch_param[1])

    gen_hc = lookup_table.get_model_macs(alphas_normal, alphas_reduce)
    hc_loss = cal_hc_loss(gene.cuda(), target_hc.item(), ALPHA, LOSS_PENALTY)

    evalulate_metric["gen_hc"].append(gen_hc.item())
    evalulate_metric["target_hc"].append(target_hc.item())

    total_loss += hc_loss.item()
  tar, _ = stats.kendalltau(evalulate_metric["gen_hc"], evalulate_metric["target_hc"])
  return evalulate_metric, total_loss, tau

if __name__ == '__main__':
  main() 

