import argparse
import logging
import os
import random
import shutil
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from networks.ProtoUNet import ProtoMMPUNet
from tqdm import tqdm
from dataloaders.dataset import DataSet, RandomGenerator
from utils.val_2D import test_single_volume_MMP, test_single_volume_1
from utils import losses
from utils.losses import PPC, PPD, PixelPrototypeCELoss, ModalityProtoContrastLoss
import gc
import math
from torch.cuda.amp import GradScaler, autocast
from utils.DiffW import DiffDW
from einops import rearrange
import h5py
from medpy import metric
from utils.sliding_window_inference import sliding_window_inference_mmp
from utils.weak_strong_augment import strong_weak_augmentation
from utils.big_mut import mut

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str,
                    default='./data/heart', help='data split path')
parser.add_argument('--data_dir', type=str,
                    default='/data//multimodal/heart', help='data store path')
parser.add_argument('--exp', type=str,
                    default='MMWHS/DBDC', help='experiment_name')
parser.add_argument('--w_proto', type=float, default=0.1)
parser.add_argument('--proj', type=int,  default=64,
                    help='proj')
parser.add_argument('--modality', type=str, default=['CT', 'MR'],
                    help='modality dataset')
parser.add_argument('--labeled_ratio', type=str, default='20p',
                    help='labeled data')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=8,
                    help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument("--warmupEpoch", type=int, default=5, help="maximum epoch to warm up")
parser.add_argument("--amp", default=True, help="Use amp for training")
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=1e-4,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[96, 96],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument("--proto_w", type=float, default=1.0, help="the weight of proto loss (you can change to suit the dataset)")
parser.add_argument("--losstype", type=str, default="ce_dice", help="the type of ce and dice loss")
parser.add_argument('-cr', '--consistency_rampup', type=float, default=None)
parser.add_argument('-r', '--cps_rampup', action='store_true', default=True)
parser.add_argument('-w', '--cps_w', type=float, default=1.0)
args = parser.parse_args()


criterion = torch.nn.CrossEntropyLoss(weight=None)
dice_loss = losses.DiceLoss(args.num_classes, weights=None)


def seg_loss(img, label, softmax=True):
    ce = criterion(img, label.long())
    if softmax:
        img_soft = F.softmax(img, dim=1)
        dice = dice_loss(img_soft, label)
    else:
        dice = dice_loss(img, label)

    return 0.5*ce+0.5*dice


def sigmoid_rampup(current, rampup_length):
    '''Exponential rampup from https://arxiv.org/abs/1610.02242'''
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch):
    if args.cps_rampup:
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        if args.consistency_rampup is None:
            args.consistency_rampup = args.max_epochs
        return args.cps_w * sigmoid_rampup(epoch, args.consistency_rampup)
    else:
        return args.cps_w

def kaiming_init_weight(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.kaiming_normal(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def org_init_weight(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.orthogonal(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def xavier_init_weight(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.xavier_normal(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def train(args):
    # train_vis = Visualizer(env='SUP/ACDC/sup/{}'.format(args.model))
    base_lr = args.base_lr
    batch_size = args.batch_size
    labeled_ratio =args.labeled_ratio
    max_epochs = args.max_epochs
    modality = args.modality
    patch_size = args.patch_size
    num_classes = args.num_classes

    pp_loss = PixelPrototypeCELoss(weight=args.w_proto)
    mpc_loss = ModalityProtoContrastLoss(weight=args.w_proto)

    model_G = ProtoMMPUNet(
        in_c=1, n_classes=num_classes, sub_proto_size=1, proj_dim=args.proj, n_modality=len(modality)
    ).cuda()
    model_G = kaiming_init_weight(model_G)
    logging.info(f'> init.. G-Net ')

    db_labeled = []
    db_unlabeled = []
    db_val = []
    # db_ldiff = []
    dataset_len = 0
    for i in range(len(modality)):
        logging.info(f'> init.. {modality[i]} dataset ')
        db_l = DataSet(args.base_dir, data_dir=args.data_dir, modality=modality[i], split=(f'labeled_{labeled_ratio}'), transform=RandomGenerator(patch_size))
        db_labeled.append(db_l)

        db_u = DataSet(args.base_dir, data_dir=args.data_dir, modality=modality[i], split=(f'unlabeled_{labeled_ratio}'),
                       transform=RandomGenerator(patch_size))
        db_unlabeled.append(db_u)

        if len(db_u) > dataset_len:
            dataset_len = len(db_u)
        db_v = DataSet(split_dir=args.base_dir, data_dir=args.data_dir, modality=modality[i], split='val')
        db_val.append(db_v)

    iter_per_epoch = math.ceil(dataset_len / batch_size)  # 用iter迭代要手算迭代数

    for i in range(len(modality)):
        db_labeled[i].set_repeat(dataset_len)
        db_unlabeled[i].set_repeat(dataset_len)

    loader_labeled = []
    loader_unlabeled = []
    loader_val = []
    # loader_ldiff = []
    for i in range(len(modality)):
        logging.info(f'> init.. {modality[i]} loader ')
        loader_l = DataLoader(db_labeled[i], batch_size=batch_size, shuffle=True,
                                      num_workers=1, pin_memory=False)
        loader_labeled.append(loader_l)

        loader_u = DataLoader(db_unlabeled[i], batch_size=batch_size, shuffle=True,
                              num_workers=1, pin_memory=False)
        loader_unlabeled.append(loader_u)

        loader_v = DataLoader(db_val[i], batch_size=1, shuffle=False,
                               num_workers=1)
        loader_val.append(loader_v)

    optimizer_G = optim.Adam(model_G.parameters(), lr=1e-4)

    if args.amp:
        scaler = GradScaler()

    best = 0
    iter_num = 0

    diffdw = DiffDW(len(args.modality), accumulate_iters=20)
    sw_aug = strong_weak_augmentation(crop_size=patch_size)
    for epoch_num in range(1, max_epochs + 1):
        gc.collect()
        torch.cuda.empty_cache()
        loss_list = []
        loss_fix_list = []

        # train
        model_G.train()

        iter_l = []
        iter_u = []
        for i in range(len(modality)):
            iter_1 = iter(loader_labeled[i])
            iter_2 = iter(loader_unlabeled[i])
            iter_l.append(iter_1)
            iter_u.append(iter_2)

        for _ in tqdm(range(iter_per_epoch), desc=f"Training epoch {epoch_num}", unit="iteration"):
            iter_num = iter_num + 1

            data_batch_labeled = []
            data_batch_unlabeled = []
            for i in range(len(modality)):
                batch_l = next(iter_l[i])
                data_batch_labeled.append(batch_l)
                batch_u = next(iter_u[i])
                data_batch_unlabeled.append(batch_u)

            for param in model_G.parameters():
                param.grad = None

            with autocast(enabled=args.amp):
                loss = 0
                if epoch_num < args.warmupEpoch:
                    for i in range(len(modality)):
                        img_i, lab_i = data_batch_labeled[i]['image'].cuda(), data_batch_labeled[i]['label'].cuda()

                        if random.random() < 0.25:
                            sw_aug.init_scale_rotate()
                            img_i, lab_i = sw_aug.strong_augment(img_i, lab_i, add_noise=False)
                        modal_label = torch.ones(img_i.shape[0], dtype=torch.int8)
                        logits = model_G.warm_up(x_2d=img_i, modal_label=modal_label * i)

                        loss_i = seg_loss(logits, lab_i)
                        weight_i = diffdw.get_weights()[i]
                        loss = loss + weight_i * loss_i
                    loss_list.append(loss.item())
                else:
                    for i in range(len(modality)):
                        img_l, lab = data_batch_labeled[i]['image'].cuda(), data_batch_labeled[i]['label'].cuda()
                        img_u = data_batch_unlabeled[i]['image'].cuda()

                        modal_label = torch.ones(img_l.shape[0], dtype=torch.int8)

                        # sup
                        if random.random() < 0.25:
                            sw_aug.init_scale_rotate()
                            img_l, lab = sw_aug.strong_augment(img_l, lab, add_noise=False)
                        outputs_G = model_G(x_2d=img_l, label=lab, use_prototype=True, modal_label=modal_label * i)
                        loss_logits_G = seg_loss(outputs_G["cls_seg"], lab)
                        loss_proto_G = seg_loss(outputs_G["proto_seg"], lab)

                        contrast_logits_p = outputs_G["contrast_logits"]
                        contrast_target_p = outputs_G["contrast_target"]

                        loss_protoc = pp_loss(contrast_logits_p, contrast_target_p)

                        loss_protom = mpc_loss(outputs_G["feat_proto_sim"], lab, modal_label[0])

                        loss_sup_G = loss_proto_G + loss_logits_G + loss_protom + loss_protoc

                        weight_i = diffdw.get_weights()[i]
                        loss_sup = weight_i * loss_sup_G


                        # fixmatch
                        sw_aug.init_scale_rotate()
                        img_u_weak = sw_aug.weak_augment(img_u)
                        img_u_strong = sw_aug.strong_augment(img_u, add_noise=True)

                        out_u_w = model_G(x_2d=img_u_weak, use_prototype=False, modal_label=modal_label * i)
                        max_u = torch.argmax(out_u_w['cls_seg'].detach(), dim=1, keepdim=False)
                        # mask_u = torch.max(F.softmax(out_u_w['cls_seg'], dim=1).detach(), dim=1, keepdim=False)[0] > 0.75

                        # max_u = mask_u * max_u
                        out_u_s = model_G(x_2d=img_u_strong, label=max_u, use_prototype=True, modal_label=modal_label * i)

                        out_u_wfp = model_G(x_2d=img_u_weak, label=max_u, use_prototype=True, modal_label=modal_label * i, drop=True)
                        loss_f = criterion(out_u_s['cls_seg'], max_u.long())
                        loss_f2 = criterion(out_u_s['proto_seg'], max_u.long())
                        loss_f3 = criterion(out_u_wfp['cls_seg'], max_u.long())
                        loss_f4 = criterion(out_u_wfp['proto_seg'], max_u.long())

                        loss_ppi = pp_loss(out_u_s["contrast_logits"], out_u_s["contrast_target"])
                        loss_ppm = mpc_loss(out_u_s["feat_proto_sim"], max_u, modal_label[0])

                        loss_ppi2 = pp_loss(out_u_wfp["contrast_logits"], out_u_wfp["contrast_target"])
                        loss_ppm2 = mpc_loss(out_u_wfp["feat_proto_sim"], max_u, modal_label[0])

                        loss_fixmatch = (loss_f + loss_f2 + loss_f3 + loss_f4) / 4 + loss_ppi + loss_ppm + loss_ppi2 + loss_ppm2

                        loss_fix_list.append(loss_fixmatch.item())

                        cps_w = get_current_consistency_weight(epoch_num//2)

                        loss_i = loss_sup + cps_w * loss_fixmatch

                        loss = loss + loss_i
                    loss_list.append(loss.item())

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer_G)
                scaler.update()
            else:
                optimizer_G.zero_grad()
                loss.backward()
                optimizer_G.step()


        if epoch_num <= args.warmupEpoch:
            logging.info(f'\n>>> epoch {epoch_num} '
                         f'avg loss : {np.mean(loss_list)}')
        else:
            logging.info(f'\n>>> epoch {epoch_num} '
                         f'avg loss : {np.mean(loss_list)} '
                         f'fix_loss : {np.mean(loss_fix_list)} ')
        update_w(args, model_G=model_G, diffW=diffdw)
        if epoch_num > args.warmupEpoch and epoch_num % 1 == 0:

            best = val_model_G(model_G, val_loader=loader_val, best_performance=best, modality=modality,
                           num_classes=num_classes)
            logging.info(f'>>> best G-Net, dice: {best} <<<')


    return "Training Finished!"


def val_model_G(model, val_loader, modality, best_performance, num_classes):
    model.eval()
    performance_all = 0
    for i in range(len(modality)):
        metric_list = 0.0
        for i_batch, sampled_batch in enumerate(tqdm(val_loader[i])):
            metric_i = test_single_volume_MMP(
                sampled_batch["image"], sampled_batch["label"], model, input_size=args.patch_size, classes=num_classes, modal_idx=i)
            metric_list += np.array(metric_i)
        metric_list = metric_list / len(val_loader[i])

        performance = np.mean(metric_list, axis=0)[0]
        performance_all = performance_all + performance
        logging.info(f'>>> valid G-Net-modal: {modality[i]}, mean_dice: {performance} <<<')
    p = performance_all / len(modality)
    if p > best_performance:
        best_performance = p
        save_best = os.path.join(snapshot_path, 'G_best_model.pth')
        torch.save(model.state_dict(), save_best)
        logging.info(f'\n>>> save G-Net:{modality}<<<\n')
    save = os.path.join(snapshot_path, 'newest_model.pth')
    torch.save(model.state_dict(), save)
    model.train()

    return best_performance


def update_w(args, model_G, diffW):
    metric_all = np.zeros(len(args.modality))
    for i in range(len(args.modality)):
        with open(args.base_dir + f'/{args.modality[i]}/labeled_{args.labeled_ratio}.list', 'r') as f:
            image_list = f.readlines()
        image_list = sorted([item.replace('\n', '').split(".")[0]
                             for item in image_list])
        # n = len(image_list) // 2
        # # Use random.sample to select n elements from the list
        # image_list = random.sample(image_list, n)

        metric_modali = 0.0
        for case in tqdm(image_list):
            h5f = h5py.File(args.data_dir + f"/{args.modality[i]}/volume/{case}.h5", 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]
            dsc = test_DBDC(image, label, model_G, input_size=args.patch_size, classes=args.num_classes, modal_idx=i)
            metric_modali += dsc
        metric_modali = metric_modali / len(image_list)
        metric_all[i] = metric_modali
        print(f'modality {args.modality[i]} training valid: {metric_modali}')
    weights = diffW.cal_weights(torch.from_numpy(metric_all).cuda())
    print('weight of modality: ', weights.cpu().numpy())


def test_DBDC(image, label, net, input_size, classes, modal_idx):
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        net.eval()
        out = sliding_window_inference_mmp(slice, net, modal_idx=modal_idx, input_size=input_size, stride=(48, 48),
                                           n_classes=classes)
        out = torch.argmax(torch.softmax(out, dim=0), dim=0)
        out = out.cpu().detach().numpy()
        prediction[ind] = out

    metric_list = []
    for i in range(1, classes):
        metric_list.append(dice_score(
            prediction == i, label == i))
    return np.mean(metric_list)

def dice_score(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        return dice
    else:
        return 0.01

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    modal = "_".join(args.modality)

    snapshot_path = "/DBDC/checkpoint/{}/{}/{}".format(
        args.exp, modal, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args)
