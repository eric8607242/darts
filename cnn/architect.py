import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, generator, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.generator = generator
    self.optimizer = torch.optim.Adam(self.generator.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=0)

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):

    self.optimizer.zero_grad()
    self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

