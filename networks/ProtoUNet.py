import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange, repeat
from networks.sinkhorn import distributed_sinkhorn
from sklearn.cluster import KMeans


class Modulation(nn.Module):
    def __init__(self, channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(Modulation, self).__init__()
        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(channels))
            self.beta = nn.Parameter(torch.zeros(channels))
        else:
            self.gamma = None
            self.beta = None

        if self.track_running_stats:
            self.running_mean = torch.zeros(channels)
            self.running_var = torch.ones(channels)
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            if self.track_running_stats:
                self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                self.running_var = self.momentum * var + (1 - self.momentum) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            y = self.gamma * x_hat + self.beta
        else:
            y = x_hat
        return y



class ModalitySpecificModulation(nn.Module):
    def __init__(self, channels, num_modality, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(ModalitySpecificModulation, self).__init__()
        self.MLMB = nn.ModuleList(
            [Modulation(channels, eps, momentum, affine, track_running_stats) for _ in range(num_modality)])

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x, modal_label):
        self._check_input_dim(x)
        ml = self.MLMB[modal_label[0]]
        x = ml(x)
        return x

def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print(
            "# old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
                momentum,
                torch.norm(old_value, p=2),
                (1 - momentum),
                torch.norm(new_value, p=2),
                torch.norm(update, p=2),
            )
        )
    return update


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, num_modality, kernel=3, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=kernel, padding=(kernel//2), stride=stride)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=kernel, padding=(kernel // 2), stride=stride)
        self.bn1 = ModalitySpecificModulation(channels=out_c, num_modality=num_modality)
        self.bn2 = ModalitySpecificModulation(channels=out_c, num_modality=num_modality)


    def forward(self, x, modal_label):
        x = self.conv1(x)
        x = self.bn1(x, modal_label)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x, modal_label)
        x = F.leaky_relu(x)
        return x


class Conv(nn.Module):
    def __init__(self, in_c, out_c, num_modality=2, kernel=3, stride=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, padding=(kernel//2), stride=stride)
        self.bn = ModalitySpecificModulation(channels=out_c, num_modality=num_modality)


    def forward(self, x, modal_label):
        x = self.conv(x)
        x = self.bn(x, modal_label)
        x = F.leaky_relu(x)
        return x


class out_layer(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1):
        super(out_layer, self).__init__()
        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size=kernel, padding=(kernel//2), stride=stride)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv2 = nn.Conv2d(in_c, in_c//2, kernel_size=kernel, padding=(kernel//2), stride=stride)
        self.bn2 = nn.BatchNorm2d(in_c//2)
        self.conv3 = nn.Conv2d(in_c//2, out_c, kernel_size=kernel, padding=(kernel//2), stride=stride)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        return x



class UNet_2D(nn.Module):
    def __init__(self, in_c, num_classes, num_modality):
        super(UNet_2D, self).__init__()
        self.conv0 = Conv(in_c=in_c, out_c=32,num_modality=num_modality)
        self.downblock1 = ConvBlock(in_c=32, out_c=64, num_modality=num_modality)
        self.downblock2 = ConvBlock(in_c=64, out_c=128, num_modality=num_modality)
        self.downblock3 = ConvBlock(in_c=128, out_c=256, num_modality=num_modality)

        self.middle = ConvBlock(in_c=256, out_c=256, num_modality=num_modality)

        self.upblock3 = ConvBlock(in_c=512, out_c=256, num_modality=num_modality)
        self.upblock2 = ConvBlock(in_c=384, out_c=128, num_modality=num_modality)
        self.upblock1 = ConvBlock(in_c=192, out_c=64, num_modality=num_modality)

        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear')

        self.out_layer1_1 = Conv(in_c=64, out_c=64, num_modality=num_modality)
        self.out_layer1_2 = Conv(in_c=64, out_c=32, num_modality=num_modality)
        self.out_layer2 = nn.Conv2d(32, num_classes, kernel_size=3, padding=1, stride=1)
        # self.out_layer = [out_layer(in_c=64, out_c=num_classes).cuda() for _ in range(num_modality)]
        # self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


    def forward(self, x, modal_label, drop=False):
        x0 = self.conv0(x, modal_label)
        x1 = self.downblock1(x0, modal_label)
        x2 = self.downblock2(self.downsample(x1), modal_label)
        x3 = self.downblock3(self.downsample(x2), modal_label)

        m = self.middle(self.downsample(x3), modal_label)

        if drop:
            m = F.dropout(m, p=0.5)

        x4 = self.upblock3(torch.cat([x3, self.upsample(m)], dim=1), modal_label)
        x5 = self.upblock2(torch.cat([x2, self.upsample(x4)], dim=1), modal_label)
        x6 = self.upblock1(torch.cat([x1, self.upsample(x5)], dim=1), modal_label)

        x = self.out_layer1_1(x6, modal_label)
        x = self.out_layer1_2(x, modal_label)
        x = self.out_layer2(x)
        # x = self.out_layer2[modal_label](x)
        # x = F.softmax(x, dim=1)
        feat = torch.cat([x6, self.up1(x5), self.up2(x4)], dim=1)

        return feat, x


class ProjectionV1(nn.Module):
    '''
    Exploring Cross-Image Pixel Contrast for Semantic Segmentation
    '''

    def __init__(self, base_channels, proj_dim):
        super(ProjectionV1, self).__init__()
        self.conv1_1 = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        self.conv1_2 = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        # self.bn = normalization(base_channels, norm='msbn', num_modality=num_modality)
        self.conv2 = nn.Conv2d(base_channels, proj_dim, kernel_size=1)

    def forward(self, x):
        # return F.normalize(self.proj(x), p=2, dim=1)
        x = self.conv1_1(x)
        x = F.leaky_relu(x)
        x = self.conv1_2(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        return x


class ProtoMMPUNet(nn.Module):
    def __init__(self, in_c, n_classes, n_modality, proj_dim=256, l2_norm=True, proto_mom=0.999, sub_proto_size=1, proto=None, temp=10):
        super(ProtoMMPUNet, self).__init__()
        self.in_c = in_c
        self.n_classes = n_classes
        self.n_modality = n_modality
        self.temp = temp
        self.proj_dim = proj_dim
        # proto params
        self.l2_norm = l2_norm
        self.proto_mom = proto_mom
        self.num_proto = sub_proto_size
        self.all_proto_size = n_modality * sub_proto_size
        self.backbone = UNet_2D(in_c=self.in_c, num_classes=self.n_classes, num_modality=self.n_modality)
        self.proj_head = ProjectionV1(base_channels=448, proj_dim=proj_dim)
        self.ini = [False for i in range(self.n_modality)]

        # initialize prototypes
        if proto is None:
            self.prototypes = nn.Parameter(torch.zeros(self.n_classes, self.all_proto_size, proj_dim),
                                               requires_grad=False)
            print(f'proto: {self.n_modality} * [{self.n_classes},{self.num_proto},{proj_dim}]')
        else:
            self.prototypes = nn.Parameter(proto, requires_grad=False)


        self.feat_norm = nn.LayerNorm(proj_dim, eps=1e-05)
        self.mask_norm = nn.LayerNorm(self.n_classes, eps=1e-05)

    def chunked_layer_norm(self, inputs, norm='feat', chunk_size=1024):
        outputs = torch.empty_like(inputs).cuda()

        for start in range(0, inputs.size(0), chunk_size):
            end = min(start + chunk_size, inputs.size(0))
            if norm=='feat':
                outputs[start:end,:] = self.feat_norm(inputs[start:end,:])
            elif norm=='mask':
                outputs[start:end,:] = self.mask_norm(inputs[start:end,:])
            else:
                return None
        return outputs


    def kmeans(self, feature, sub_proto_size):
        """

        :param feature: size:(n,256) n is the number of features whose label is 1 or 0
        :param sub_proto_size:
        :return: cluster center for each clustern size:(sub_proto_size,256)
        """
        kmeans = KMeans(n_clusters=sub_proto_size, random_state=0, n_init=10).fit(feature)
        centroids = kmeans.cluster_centers_
        return centroids

    def prototype_learning(self, out_feat, nearest_proto_distance, label, feat_proto_sim, modal_label):
        """
        :param out_feat: [bs*h*w, dim] pixel feature
        :param nearest_proto_distance: [bs, cls_num, h, w]
        :param label: [bs*h*w] segmentation label
        :param feat_proto_sim: [bs*h*w, sub_cluster, cls_num]
        """

        cosine_similarity = feat_proto_sim.reshape(feat_proto_sim.shape[0], -1)  # (h*w) * (sub_cluster*cls_num)
        proto_logits = cosine_similarity
        # proto_target = label.clone().float()  # (n, )
        proto_target = torch.zeros_like(label).float()  # (n, ) n = h*w
        pred_seg = torch.max(nearest_proto_distance, 1)[1]
        mask = (label == pred_seg.view(-1))

        # clustering for each class, online
        # update the prototypes
        protos = self.prototypes.detach().clone()  # C,
        for id_c in range(self.n_classes):

            init_q = feat_proto_sim[
                ..., id_c
            ]  # n, k, cls => n, k
            init_q = init_q[label == id_c, ...]
            if init_q.shape[0] == 0:  # no such class
                # print("no class :",id_c)
                continue

            q, indexs = distributed_sinkhorn(init_q)
            # q: (n, 10) one-hot prototype label
            # indexes: (n, ) prototype label # torch.unique(indexs) = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            # print(f'q is nan {torch.isnan(q).any()}, is inf {torch.isinf(q).any()}')

            m_k = mask[label == id_c]

            c_k = out_feat[label == id_c, ...]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.all_proto_size)

            m_q = q * m_k_tile  # n x self.num_prototype

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile  # n x embedding_dim

            n_ = m_q.shape[0]
            f = torch.zeros(self.all_proto_size, self.proj_dim).cuda()

            batch_size = n_ // 8
            for i in range(8):
                # 分割 A 和 B
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                A_part = m_q.transpose(0, 1)[:, start_idx:end_idx]
                B_part = c_q[start_idx:end_idx, :]

                # 执行矩阵乘法并累加结果
                f += torch.mm(A_part, B_part)

            # f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim
            # start_time = time.time()
            # f = torch.sparse.mm(m_q.transpose(0, 1).to_sparse(), c_q.to_sparse())
            # f = f.to_dense()
            # print('time: {:.2f}s'.format(time.time() - start_time))

            n = torch.sum(m_q, dim=0)  # (n, p) => p

            if torch.sum(n) > 0:
                if self.l2_norm:
                    f = F.normalize(f, p=2, dim=-1)  # [p, 720]
                # print(f'new val is nan {torch.isnan(f[n != 0, :]).any()}, is inf {torch.isinf(f[n != 0, :]).any()}')
                new_value = momentum_update(
                    old_value=protos[id_c, n != 0, :],
                    new_value=f[n != 0, :],
                    momentum=self.proto_mom,
                    # debug=True if id_c == 1 else False,
                    debug=False,
                )  # [p, dim]
                protos[id_c, n != 0, :] = new_value  # [cls, p, dim]

            proto_target[label == id_c] = indexs.float() + (
                    self.all_proto_size * id_c
            )  # (n, ) cls*k classes totally

        if self.l2_norm:
            self.prototypes = nn.Parameter(l2_normalize(protos), requires_grad=False)
        else:
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        return proto_logits, proto_target

    def warm_up(self,x_2d, modal_label=None):

        assert len(x_2d.shape) == 4
        feature2d, classifer2d = self.backbone(x_2d, modal_label=modal_label)
        return classifer2d

    def initialize(self, x, label, modal_label):
        feature2d, classifer2d = self.backbone(x, modal_label)

        if self.proj_head == None:
            embedding = feature2d
        else:
            embedding = self.proj_head(feature2d)
        # b, dim, h, w, z = embedding.shape
        out_feat = rearrange(embedding, "b c h w -> (b h w) c")
        out_feat = self.feat_norm(out_feat)  # (n, dim) n, 256
        # out_feat = self.chunked_layer_norm(out_feat, norm='feat')  # (n, dim) n, 256

        if self.l2_norm:
            out_feat = l2_normalize(out_feat)  # cosine sim norm
        features = out_feat.detach().clone().cpu()

        label = label.contiguous().view(-1)
        label = label.detach().clone().cpu()
        feat_center_list = []
        idx = modal_label[0]
        for i in range(self.n_classes):
            feat = features[label == i]
            if feat.numel() == 0:
                return 0.0
            feat_centroids = self.kmeans(feat, self.num_proto)  # numpy.array (1, 256)
            self.prototypes[i,idx*self.num_proto:(idx+1)*self.num_proto,:] = torch.from_numpy(feat_centroids).float().cuda()
        self.prototypes.requires_grad_(False)

        print(f'initialize protos modality:{idx}')
        self.ini[idx] = True
        return 1.0


    def forward(self, x_2d, label=None, use_prototype=False, modal_label=None, drop=False):
        """

        :param x: size:(B*D,C,H,W)
        :param label: (B*D,H,W)
        :param use_prototype: after several pretraining iterations it will be True
        :return:
        """
        if not all(self.ini) and use_prototype:
            # use_prototype = False
            if not self.ini[modal_label[0]]:
                ini = self.initialize(x_2d, label, modal_label)
                if ini == 1.0:
                    print(
                        f'initialize prototype for modality {modal_label[0]} succeed')
                else:
                    print(ini)
        if drop:
            feature2d, classifer2d = self.backbone(x_2d, modal_label, drop=drop)
        else:
            feature2d, classifer2d = self.backbone(x_2d, modal_label)  # feat(b*d,64,h,w) cls(b*d,2,h,w)
        return_dict = {}
        return_dict["cls_seg"] = classifer2d
        if self.proj_head == None:
            embedding = feature2d
        else:
            embedding = self.proj_head(feature2d)
        # return_dict["feature"] = feature3d
        b, dim, h, w = embedding.shape
        out_feat = rearrange(embedding, "b c h w -> (b h w) c")
        out_feat = self.feat_norm(out_feat)  # (n, dim)
        # out_feat = self.chunked_layer_norm(out_feat, norm='feat')  # (n, dim) n, 256

        if self.l2_norm:
            out_feat = l2_normalize(out_feat)  # cosine sim norm
        return_dict["out_feat"] = out_feat

        if self.l2_norm:
            for i in range(self.n_modality):
                self.prototypes.data.copy_(l2_normalize(self.prototypes))
        else:
            for i in range(self.n_modality):
                self.prototypes.data.copy_(self.prototypes)
        # cosine sim

        feat_proto_sim = torch.einsum(
            "nd,kmd->nmk", out_feat, self.prototypes
        )  # [n, dim], [csl, p, dim] -> [n, p, cls]: n=(b h w z)
        return_dict["feat_proto_sim"] = feat_proto_sim
        # nearest_proto_distance = torch.amax(feat_proto_sim, dim=1)
        nearest_proto_distance = feat_proto_sim[:,modal_label[0]*self.num_proto:(modal_label[0]+1)*self.num_proto,:].squeeze()
        nearest_proto_distance = self.mask_norm(nearest_proto_distance)


        nearest_proto_distance = rearrange(
            nearest_proto_distance, "(b h w) k -> b k h w", b=b, h=h, w=w
        )  # [n, cls] -> [b, cls, h, w] -> correspond the s in equ(6)
        return_dict["proto_seg"] = nearest_proto_distance / self.temp
        if use_prototype:
            label_expand = label.contiguous().view(-1)
            contrast_logits, contrast_target = self.prototype_learning(
                out_feat,
                nearest_proto_distance,
                label_expand,
                feat_proto_sim,
                modal_label
            )
            return_dict["contrast_logits"] = contrast_logits
            return_dict["contrast_target"] = contrast_target

        return return_dict


class ModalityProtoContrastLoss(nn.Module):
    def __init__(self, num_proto=1):
        super(ModalityProtoContrastLoss, self).__init__()
        self.loss_mp_weight = 0.01
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
