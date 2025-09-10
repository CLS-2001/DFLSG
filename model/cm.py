import collections
from abc import ABC
from model import objectives
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, autograd
import logging


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())
        #********************************************************************************************
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        logger = logging.getLogger("DFLSG.train")
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)

        outputs = inputs.mm(ctx.features.t())


        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        logger = logging.getLogger("DFLSG.train")
        inputs, targets = ctx.saved_tensors

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.1):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.1, use_hard=True):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, i_feats,  targets):

        inputs_i = F.normalize(i_feats, dim=1).cuda()

        if self.use_hard:
            outputs = cm_hard(inputs_i, targets, self.features, self.momentum)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)


        cl_features = []
        for index in targets:
            cl_features.append(self.features[index])
        cl_features = torch.stack(cl_features, dim=0)
        outputs1 = inputs_i.mm(cl_features.t())

        # outputs /= self.temp
        # loss = F.cross_entropy(outputs, targets)
        tri_loss = objectives.compute_TRL(outputs1, targets, margin=0.1, tau=0.02)

        cosine_distance = 1 - outputs1
        center_loss = cosine_distance.mean()

        return  tri_loss + 0.08 * center_loss

