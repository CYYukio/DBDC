import torch
import numpy as np

def EMA(cur_weight, past_weight, momentum=0.9):
    new_weight = momentum * past_weight + (1 - momentum) * cur_weight
    return new_weight

class DiffDW:
    def __init__(self, num_modality, accumulate_iters=50):
        self.last_dice = torch.zeros(num_modality).float().cuda() + 1e-8
        # self.dice_func = SoftDiceLoss(smooth=1e-8, do_bg=True)
        self.cls_learn = torch.zeros(num_modality).float().cuda()
        self.cls_unlearn = torch.zeros(num_modality).float().cuda()
        self.num_modality = num_modality
        self.dice_weight = torch.ones(num_modality).float().cuda()
        self.accumulate_iters = accumulate_iters
        self.weights = torch.FloatTensor(np.ones(self.num_modality)).cuda()
        # self.data_len = torch.tensor((24,16,16,16)).cuda()

    # def init_weights(self):
    #     # weights = np.ones(self.num_modality) * self.num_modality
    #     weights = np.ones(self.num_modality)
    #     self.weights = torch.FloatTensor(weights).cuda()
    #     return weights
    def get_weights(self):
        return self.weights

    def cal_weights(self, cur_dsc):
        delta_dice = cur_dsc - self.last_dice
        cur_cls_learn = torch.where(delta_dice > 0, delta_dice, 0) * torch.log(cur_dsc / self.last_dice)
        cur_cls_unlearn = torch.where(delta_dice <= 0, delta_dice, 0) * torch.log(cur_dsc / self.last_dice)
        self.last_dice = cur_dsc

        self.cls_learn = EMA(cur_cls_learn, self.cls_learn,
                             momentum=(self.accumulate_iters - 1) / self.accumulate_iters)
        self.cls_unlearn = EMA(cur_cls_unlearn, self.cls_unlearn,
                               momentum=(self.accumulate_iters - 1) / self.accumulate_iters)

        cur_diff = (self.cls_unlearn + 1e-8) / (self.cls_learn + 1e-8)
        cur_diff = torch.pow(cur_diff, 1 / 5)
        self.dice_weight = EMA(1. - cur_dsc, self.dice_weight,
                               momentum=(self.accumulate_iters - 1) / self.accumulate_iters)
        # self.dice_weight = EMA(cur_dsc, self.dice_weight, momentum=(self.accumulate_iters - 1) / self.accumulate_iters)

        # d_w = (self.dice_weight / torch.sum(self.dice_weight))
        # cur_diff = (cur_diff / torch.sum(cur_diff))
        weights = cur_diff * self.dice_weight

        print(f'diff_w: {cur_diff},   dice_w:{self.dice_weight}')

        self.weights = (weights / torch.sum(weights)) * self.num_modality
        return self.weights
        # weights = weights / weights.max()
        # return weights * self.num_modality
