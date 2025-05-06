import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

class DiceLoss(nn.Module):
    def __init__(self, n_classes,weights=None):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.weights = weights

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(dim=1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if self.weights is None:
            self.weights = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * self.weights[i]
        return loss / self.n_classes



def PPC(contrast_logits, contrast_target):
    loss_ppc = F.cross_entropy(contrast_logits, contrast_target.long())
    # loss_ppc = F.cross_entropy(contrast_logits, contrast_target)
    return loss_ppc

def PPD(contrast_logits, contrast_target):
    logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
    loss_ppd = (1 - logits).pow(2).mean()

    return loss_ppd

class ModalityProtoCELoss(nn.Module):
    def __init__(self, weight=0.1):
        super(ModalityProtoCELoss, self).__init__()
        # self.num_proto=num_proto
        self.loss_mp_weight = weight
    def forward(self, feat_proto_sim, label, modal_label):

        kth_prototype_sim = feat_proto_sim[:, modal_label, :].squeeze().max(dim=1).values

        other_prototypes_sim = torch.mean(feat_proto_sim[:, [i for i in range(feat_proto_sim.size(1)) if i != modal_label], :].squeeze(), dim=1)

        loss = -torch.mean(kth_prototype_sim) + torch.mean(other_prototypes_sim)

        return self.loss_mp_weight * loss

class ModalityProtoContrastLoss(nn.Module):
    def __init__(self, weight=0.1):
        super(ModalityProtoContrastLoss, self).__init__()
        self.loss_mp_weight = weight

    def forward(self, feat_proto_sim, label, modal_label):
        # start_idx = modal_label * self.num_proto
        # end_idx = (modal_label + 1) * self.num_proto
        n, p, cls = feat_proto_sim.shape

        label_expand = label.to(torch.int64).view(n, 1, 1) #.contiguous().view(-1)

        ground_truth_similarity = torch.gather(feat_proto_sim, 2, label_expand.expand(n, p, 1)).squeeze(2)

        kth_prototype_sim = ground_truth_similarity[:, modal_label]

        other_prototypes_sim = torch.mean(ground_truth_similarity[:, [i for i in range(feat_proto_sim.size(1)) if i != modal_label]], dim=1)

        loss = -torch.mean(kth_prototype_sim) + torch.mean(other_prototypes_sim)

        return self.loss_mp_weight * loss



class PixelPrototypeCELoss(nn.Module):
    def __init__(self, weight=0.1):
        super(PixelPrototypeCELoss, self).__init__()
        self.loss_ppc_weight = weight
        self.loss_ppd_weight = weight

    def forward(self, contrast_logits, contrast_target):
        loss_ppc = PPC(contrast_logits, contrast_target)
        loss_ppd = PPD(contrast_logits, contrast_target)
        return  self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd

