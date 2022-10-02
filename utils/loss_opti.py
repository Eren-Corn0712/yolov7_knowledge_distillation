# Loss functions

from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.general import LOGGER, colorstr
from utils.general import bbox_iou, box_iou, box_ciou, xywh2xyxy
from utils.loss import FocalLoss, smooth_BCE
from utils.my_utils import de_parallel


def subset_of_pred(pred: Tensor,
                   indices: Tuple,
                   anchor: Tensor,
                   num_classes: int) -> Tuple[Tensor, Tensor, Tensor]:
    pxy, pwh, pobj, pcls = pred[indices].split((2, 2, 1, num_classes), 1)
    pxy = pxy.sigmoid() * 2 - 0.5
    pwh = (pwh.sigmoid() * 2) ** 2 * anchor
    pbox = torch.cat((pxy, pwh), 1)  # predicted box
    return pbox, pobj, pcls


@torch.no_grad()
def find_n_positive(p, targets, model, n=3):
    det = de_parallel(model).model[-1]
    de_model = de_parallel(model)
    na, nt = det.na, targets.shape[0]  # number of anchors and number of targets
    indices, anch = [], []
    gain = torch.ones(7, device=targets.device).long()  # normalized to grids-pace gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices. shape (3,n,7)

    if n == 3:
        g = 0.5  # bias
    elif n == 5:
        g = 1.0
    else:
        raise NotImplemented(f"find_{n}_positive grid is not implemented.")

    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    for i in range(det.nl):
        anchors, shape = det.anchors[i], p[i].shape
        gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        t = targets * gain  # shape(3,n,7)
        if nt:
            # Matches
            # (3,n,2) / (3,1,2)
            r = t[..., 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1 / r).max(2)[0] < de_model.hyp['anchor_t']  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter shape (match_anchor_num, 7)

            # Offsets
            gxy = t[:, 2:4]  # grid xy (n, 2)
            gxi = gain[[2, 3]] - gxy  # inverse (n, 2)
            j, k = ((gxy % 1 < g) & (gxy > 1)).T  # j shape(n,) k shape(n,)
            l, m = ((gxi % 1 < g) & (gxi > 1)).T  # l shape(n,) m shape(n)
            j = torch.stack((torch.ones_like(j), j, k, l, m))  # shape (5,n)
            t = t.repeat((5, 1, 1))[j]  # shape(n,7) -> (5, n, 7) filter the target
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
        a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
        gij = (gxy - offsets).long()  # convert to int
        gi, gj = gij.T  # grid indices

        # Append
        indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
        anch.append(anchors[a])  # anchors

    return indices, anch


@torch.no_grad()
def build_targets(p, targets, imgs, model, n=3):
    det = de_parallel(model).model[-1]

    indices, anch = find_n_positive(p, targets, model, n)

    matching_bs = [[] for pp in p]
    matching_as = [[] for pp in p]
    matching_gjs = [[] for pp in p]
    matching_gis = [[] for pp in p]
    matching_targets = [[] for pp in p]
    matching_anchs = [[] for pp in p]

    nl = len(p)

    for batch_idx in range(p[0].shape[0]):
        b_idx = targets[:, 0] == batch_idx
        this_target = targets[b_idx]
        if this_target.shape[0] == 0:
            continue

        # get target xyxy
        txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
        txyxy = xywh2xyxy(txywh)

        pxyxys = []
        p_cls = []
        p_obj = []

        from_which_layer = []
        all_b = []
        all_a = []
        all_gj = []
        all_gi = []
        all_anch = []

        # Get Positive anchor encoder to predict box
        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            idx = (b == batch_idx)
            b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
            all_b.append(b)
            all_a.append(a)
            all_gj.append(gj)
            all_gi.append(gi)
            all_anch.append(anch[i][idx])
            from_which_layer.append(torch.ones(size=(len(b),)) * i)

            fg_pred = pi[b, a, gj, gi]
            p_obj.append(fg_pred[:, 4:5])
            p_cls.append(fg_pred[:, 5:])

            grid = torch.stack([gi, gj], dim=1)
            pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * det.stride[i]  # / 8.

            pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * det.stride[i]  # / 8.
            pxywh = torch.cat([pxy, pwh], dim=-1)
            pxyxy = xywh2xyxy(pxywh)
            pxyxys.append(pxyxy)

        pxyxys = torch.cat(pxyxys, dim=0)
        if pxyxys.shape[0] == 0:
            continue

        p_obj = torch.cat(p_obj, dim=0)
        p_cls = torch.cat(p_cls, dim=0)
        from_which_layer = torch.cat(from_which_layer, dim=0)
        all_b = torch.cat(all_b, dim=0)
        all_a = torch.cat(all_a, dim=0)
        all_gj = torch.cat(all_gj, dim=0)
        all_gi = torch.cat(all_gi, dim=0)
        all_anch = torch.cat(all_anch, dim=0)

        pair_wise_iou = box_iou(txyxy, pxyxys)  # (targets, predictions)
        pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

        gt_cls_per_image = (
            F.one_hot(this_target[:, 1].to(torch.int64), det.nc)
            .float()
            .unsqueeze(1)
            .repeat(1, pxyxys.shape[0], 1)
        )

        num_gt = this_target.shape[0]

        cls_preds_ = (
                p_cls.float().sigmoid_().unsqueeze(0).repeat(num_gt, 1, 1)
                * p_obj.sigmoid_().unsqueeze(0).repeat(num_gt, 1, 1)
        )
        y = cls_preds_.sqrt_()
        pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
            torch.log(y / (1 - y)), gt_cls_per_image, reduction="none").sum(-1)

        del cls_preds_

        cost = (pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss)

        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        topk_ious, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks

        anchor_matching_gt = matching_matrix.sum(0)

        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1

        fg_mask_inboxes = matching_matrix.sum(0) > 0
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

        from_which_layer = from_which_layer[fg_mask_inboxes]
        all_b = all_b[fg_mask_inboxes]
        all_a = all_a[fg_mask_inboxes]
        all_gj = all_gj[fg_mask_inboxes]
        all_gi = all_gi[fg_mask_inboxes]
        all_anch = all_anch[fg_mask_inboxes]

        this_target = this_target[matched_gt_inds]

        for i in range(nl):
            layer_idx = from_which_layer == i
            matching_bs[i].append(all_b[layer_idx])
            matching_as[i].append(all_a[layer_idx])
            matching_gjs[i].append(all_gj[layer_idx])
            matching_gis[i].append(all_gi[layer_idx])
            matching_targets[i].append(this_target[layer_idx])
            matching_anchs[i].append(all_anch[layer_idx])

    for i in range(nl):
        if matching_targets[i]:
            matching_bs[i] = torch.cat(matching_bs[i], dim=0)
            matching_as[i] = torch.cat(matching_as[i], dim=0)
            matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
            matching_gis[i] = torch.cat(matching_gis[i], dim=0)
            matching_targets[i] = torch.cat(matching_targets[i], dim=0)
            matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
        else:
            matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
            matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
            matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
            matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
            matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
            matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

    return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs


class DistillKLDivLoss(nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    """

    def __init__(self, temperature):
        super(DistillKLDivLoss, self).__init__()
        self.T = temperature

    def forward(self, z_s, z_t):
        if z_s.size(0) != z_t.size(0):
            raise ValueError("Teacher and Student Input size must be equal.")
        input_size = z_t.size(0)

        soft_target = F.softmax(z_t / self.T, dim=-1)
        soft_prob = F.log_softmax(z_s / self.T, dim=-1)
        loss = F.kl_div(soft_prob, soft_target, size_average=False, log_target=False) * (self.T ** 2) / input_size
        return loss


class SPKDLoss(nn.Module):
    """
    Similarity-Preserving Knowledge Distillation
    """

    def __init__(self, model):
        super(SPKDLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters
        self.device = device
        self.hyp = h

    def forward(self, stu_feats, tea_feats):
        loss = torch.zeros(1, device=self.device)
        batch_size = stu_feats[0].size(0)
        for stu_feat, tea_feat in zip(stu_feats, tea_feats):
            spkd_loss = self.compute_spkd_loss(stu_feat, tea_feat)
            loss += spkd_loss

        loss = loss / (batch_size ** 2)
        loss *= self.hyp['spkd']
        loss_dict = {'SPKD': loss}
        return loss_dict

    def matmul_and_normalize(self, z):
        z = F.adaptive_avg_pool2d(z, (1, 1))
        z = torch.flatten(z, 1)
        similarity_mat = torch.matmul(z, torch.t(z))
        return F.normalize(similarity_mat, p=2.0, dim=1)

    def compute_spkd_loss(self, stu_feat, tea_feat):
        tea_feat = self.matmul_and_normalize(tea_feat)
        stu_feat = self.matmul_and_normalize(stu_feat)
        return torch.norm(tea_feat - stu_feat) ** 2


class SCKDLoss(nn.Module):
    """
    similarity correspondence
    """

    def __init__(self, model):
        super(SCKDLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters
        self.device = device
        self.hyp = h

    def forward(self, stu_feats, tea_feats):
        loss = torch.zeros(1, device=self.device)
        for stu_feat, tea_feat in zip(stu_feats, tea_feats):
            stu_feat = stu_feat.flatten(2, 3)  # (b,s_c,hw)
            tea_feat = tea_feat.flatten(2, 3)  # (b,t_c,hw)
            stu_feat = F.softmax(stu_feat, dim=-1)
            tea_feat = F.softmax(tea_feat, dim=-1)

            sim = torch.matmul(F.normalize(stu_feat, p=2, dim=-1),
                               F.normalize(tea_feat, p=2, dim=-1).permute(0, 2, 1))  # (b, s_c, t_c)

            k = 1
            sim_ind = sim.topk(k, 2)[1]  # (b, s_c, k)

            ind_tea_feat = torch.gather(
                tea_feat.unsqueeze(1).expand(-1, stu_feat.size(1), -1, -1),  # (b, s_c, t_c, hw)
                dim=2,
                index=sim_ind.unsqueeze(-1).expand(-1, -1, -1, stu_feat.size(-1))  # (b, s_c, k, hw)
            ).permute(0, 2, 1, 3).flatten(0, 1)  # (b,s_c,k,hw) -> (b*k, s_c, hw)

            batch_size = stu_feat.size(0)
            batch_index = torch.arange(batch_size, device=self.device).repeat_interleave(k)

            loss += torch.sum(-ind_tea_feat * F.log_softmax(stu_feat[batch_index], dim=-1), dim=-1).mean(-1).mean(-1)

        loss /= len(stu_feats)
        loss *= self.hyp['sckd']
        loss_dict = {'SCKD': loss}
        return loss_dict


class ComputeLossOTA:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossOTA, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = de_parallel(model).model[-1]
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        self.device = device
        self.model = model
        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, imgs):  # predictions, targets, model
        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)

        bs, as_, gjs, gis, targets, anchors = build_targets(p, targets, imgs, self.model, 5)

        pre_gen_gains = [torch.tensor(pp.shape, device=self.device)[[3, 2, 3, 2]] for pp in p]

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi, target, anchor = bs[i], as_[i], gjs[i], gis[i], targets[i], anchors[
                i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=self.device)  # target obj

            grid = torch.stack([gi, gj], dim=1)
            selected_tbox = target[:, 2:6] * pre_gen_gains[i]
            selected_tbox[:, :2] -= grid
            selected_tcls = target[:, 1].long()

            n = b.shape[0]  # number of targets
            if n:
                pbox, pobj, pcls = subset_of_pred(pi, (b, a, gj, gi), anchor, self.nc)

                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification

                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


class ComputeLogitKDLossV0:
    # Compute Losses and KD Losses
    def __init__(self, teacher, student, autobalance=False):
        super(ComputeLogitKDLossV0, self).__init__()
        device = next(student.parameters()).device  # get model device
        h = student.hyp

        # Define Criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        self.student = student
        self.teacher = teacher
        self.s = de_parallel(student).model[-1]  # Detect() module
        self.t = de_parallel(teacher).model[-1]  # Detect() module

        self.balance = {3: [4.0, 1.0, 0.4]}.get(self.s.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(self.s.stride).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.hyp, self.autobalance = BCEcls, BCEobj, h, autobalance

        self.device = device
        self.temperature = h.get('temperature', 1.0)
        self.KLDcls = DistillKLDivLoss(self.temperature)

    def __call__(self, tea_p, stu_p, targets, imgs) -> Dict:
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss

        lkdcls = torch.zeros(1, device=self.device)  # KD Cls Loss
        lkdbox = torch.zeros(1, device=self.device)  # KD box Loss
        lkdobj = torch.zeros(1, device=self.device)  # KD Obj Loss

        # Image index,anchor index, grid_y_index, grid_x_index, matching target, matching anchor wh
        teacher_matching = build_targets(tea_p, targets, imgs, self.teacher)
        student_matching = build_targets(stu_p, targets, imgs, self.student)
        teacher_matching = tuple(zip(*teacher_matching))
        student_matching = tuple(zip(*student_matching))

        pre_gen_gains = [torch.tensor(pp.shape, device=self.device)[[3, 2, 3, 2]] for pp in tea_p]  # x,y,w,h shape(4,)

        for i, (tea_pi, stu_pi) in enumerate(zip(tea_p, stu_p)):  # layer index, layer predictions
            t_b, t_a, t_gj, t_gi, t_target, t_anchor = teacher_matching[i][:]
            s_b, s_a, s_gj, s_gi, s_target, s_anchor = student_matching[i][:]

            # teacher target-box and cls
            grid = torch.stack([t_gi, t_gj], dim=1)
            tea_selected_tbox = t_target[:, 2:6] * pre_gen_gains[i]
            tea_selected_tbox[:, :2] -= grid
            tea_selected_tcls = t_target[:, 1].long()

            # student target-box and cls
            grid = torch.stack([s_gi, s_gj], dim=1)
            stu_selected_tbox = s_target[:, 2:6] * pre_gen_gains[i]
            stu_selected_tbox[:, :2] -= grid
            stu_selected_tcls = s_target[:, 1].long()

            tea_tobj = torch.zeros(tea_pi.shape[:4], dtype=tea_pi.dtype, device=self.device)  # B x 3 x 80 x 80
            stu_tobj = torch.zeros(stu_pi.shape[:4], dtype=stu_pi.dtype, device=self.device)  # target obj

            tea_n = t_b.shape[0]  # number of targets
            stu_n = s_b.shape[0]  # number of targets

            # Student
            if stu_n:
                # target-subset of predictions
                stu_pbox, stu_pobj, stu_pcls = subset_of_pred(stu_pi, (s_b, s_a, s_gj, s_gi), s_anchor, self.s.nc)

                # Bounding Box
                iou = bbox_iou(stu_pbox.T, stu_selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                stu_tobj[s_b, s_a, s_gj, s_gi] = (1.0 - self.student.gr) + \
                                                 self.student.gr * iou.detach().clamp(0).type(stu_tobj.dtype)

                # Classification
                if self.s.nc > 1:  # cls loss (only if multiple classes)
                    stu_t = torch.full_like(stu_pcls, self.cn, device=self.device)  # targets
                    stu_t[range(stu_n), stu_selected_tcls] = self.cp
                    lcls += self.BCEcls(stu_pcls, stu_t)  # BCE

            # Teacher
            if tea_n:
                # Teacher Predict[Teacher Assign] target-subset of predictions
                tea_pbox, tea_pobj, tea_pcls = subset_of_pred(tea_pi, (t_b, t_a, t_gj, t_gi), t_anchor, self.t.nc)
                # Student Predict[Teacher Assign] target-subset of predictions
                spta_pbox, spta_pobj, spta_pcls = subset_of_pred(stu_pi, (t_b, t_a, t_gj, t_gi), t_anchor, self.t.nc)

                # Bounding Box
                kd_iou = bbox_iou(spta_pbox.T, tea_selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lkdbox += (1.0 - kd_iou).mean()  # kd iou loss

                # Objectness
                tea_tobj[t_b, t_a, t_gj, t_gi] = \
                    (1.0 - self.teacher.gr) + self.teacher.gr * kd_iou.detach().clamp(0).type(tea_tobj.dtype)

                if self.t.nc > 1:  # cls loss (only if multiple classes)
                    tea_t = torch.full_like(tea_pcls, self.cn, device=self.device)  # targets
                    tea_t[range(tea_n), tea_selected_tcls] = self.cp
                    tea_acc = (tea_pcls.argmax(1) == tea_t.argmax(1)).sum(0) / tea_n

                    if tea_acc >= 0.95:
                        lkdcls += self.KLDcls(spta_pcls, tea_pcls)
                    else:
                        lkdcls += self.BCEcls(spta_pcls, tea_t)

                lkdobj += self.BCEobj(stu_pi[..., 4], tea_tobj)

            obji = self.BCEobj(stu_pi[..., 4], stu_tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lkdbox *= self.hyp['kdbox']
        lkdobj *= self.hyp['kdobj']
        lkdcls *= self.hyp['kdcls']

        loss_dict = {'box': lbox,
                     'obj': lobj,
                     'cls': lcls,
                     'kd-box': lkdbox,
                     'kd-obj': lkdobj,
                     'kd-cls': lkdcls}
        return loss_dict


class ComputeLogitKDLossV2:
    # Compute Losses and KD Losses
    def __init__(self, teacher, student, autobalance=False):
        super(ComputeLogitKDLossV2, self).__init__()
        device = next(student.parameters()).device  # get model device
        h = student.hyp

        # Define Criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        self.student = student
        self.teacher = teacher
        self.s = de_parallel(student).model[-1]  # Detect() module
        self.t = de_parallel(teacher).model[-1]  # Detect() module

        self.balance = {3: [4.0, 1.0, 0.4]}.get(self.s.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(self.s.stride).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.hyp, self.autobalance = BCEcls, BCEobj, h, autobalance

        self.device = device

        self.temperature = h.get('temperature', 1.0)
        self.KLDcls = DistillKLDivLoss(self.temperature)

    def __call__(self, tea_p, stu_p, targets, imgs):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss

        lkdcls = torch.zeros(1, device=self.device)  # KD Cls Loss
        lkdbox = torch.zeros(1, device=self.device)  # KD box Loss
        lkdobj = torch.zeros(1, device=self.device)  # KD Obj Loss

        # Image index,anchor index, grid_y_index, grid_x_index, matching target, matching anchor wh
        teacher_matching = build_targets(tea_p, targets, imgs, self.teacher)
        student_matching = build_targets(stu_p, targets, imgs, self.student)
        teacher_matching = list(zip(*teacher_matching))
        student_matching = list(zip(*student_matching))

        pre_gen_gains = [torch.tensor(pp.shape, device=self.device)[[3, 2, 3, 2]] for pp in tea_p]

        for i, (tea_pi, stu_pi) in enumerate(zip(tea_p, stu_p)):  # layer index, layer predictions
            t_b, t_a, t_gj, t_gi, t_target, t_anchor = teacher_matching[i]
            s_b, s_a, s_gj, s_gi, s_target, s_anchor = student_matching[i]

            # teacher target-box and cls
            grid = torch.stack([t_gi, t_gj], dim=1)
            tea_selected_tbox = t_target[:, 2:6] * pre_gen_gains[i]
            tea_selected_tbox[:, :2] -= grid
            tea_selected_tcls = t_target[:, 1].long()

            # student target-box and cls
            grid = torch.stack([s_gi, s_gj], dim=1)
            stu_selected_tbox = s_target[:, 2:6] * pre_gen_gains[i]
            stu_selected_tbox[:, :2] -= grid
            stu_selected_tcls = s_target[:, 1].long()

            tea_tobj = torch.zeros(tea_pi.shape[:4], dtype=tea_pi.dtype, device=self.device)  # 3 x 80 x 80 x 15
            stu_tobj = torch.zeros(stu_pi.shape[:4], dtype=stu_pi.dtype, device=self.device)  # target obj

            tea_n = t_b.shape[0]  # number of targets
            stu_n = s_b.shape[0]  # number of targets

            # Student
            if stu_n:
                # target-subset of predictions
                stu_pbox, stu_pobj, stu_pcls = \
                    subset_of_pred(stu_pi, (s_b, s_a, s_gj, s_gi), s_anchor, self.s.nc)

                # Bounding Box
                iou = bbox_iou(stu_pbox.T, stu_selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                stu_tobj[s_b, s_a, s_gj, s_gi] = (1.0 - self.student.gr) + \
                                                 self.student.gr * iou.detach().clamp(0).type(stu_tobj.dtype)

                # Classification
                if self.s.nc > 1:  # cls loss (only if multiple classes)
                    stu_t = torch.full_like(stu_pcls, self.cn, device=self.device)  # targets
                    stu_t[range(stu_n), stu_selected_tcls] = self.cp
                    lcls += self.BCEcls(stu_pcls, stu_t)  # BCE

            # Teacher
            if tea_n:
                # Teacher Predict[Teacher Assign] target-subset of predictions
                tea_pbox, tea_pobj, tea_pcls = subset_of_pred(tea_pi, (t_b, t_a, t_gj, t_gi), t_anchor, self.t.nc)

                s2t_pair_wise_iou = box_ciou(stu_pbox, tea_selected_tbox)
                iou_idx = torch.argmax(s2t_pair_wise_iou, 1)
                del s2t_pair_wise_iou

                kd_iou = bbox_iou(stu_pbox.T, tea_selected_tbox[iou_idx], x1y1x2y2=False, CIoU=True)
                lkdbox += (1.0 - kd_iou).mean()  # iou loss

                tea_iou = bbox_iou(tea_pbox.T, tea_selected_tbox, x1y1x2y2=False, CIoU=True)
                # Objectness
                tea_tobj[t_b, t_a, t_gj, t_gi] = (1.0 - self.teacher.gr) + \
                                                 self.teacher.gr * tea_iou.detach().clamp(0).type(tea_tobj.dtype)

                if self.t.nc > 1:  # cls loss (only if multiple classes)
                    tea_t = torch.full_like(tea_pcls, self.cn, device=self.device)  # targets
                    tea_t[range(tea_n), tea_selected_tcls] = self.cp
                    tea_acc = (tea_pcls.argmax(1) == tea_t.argmax(1)).sum(0) / tea_n

                    if tea_acc >= 0.95:
                        lkdcls += self.KLDcls(stu_pcls, tea_pcls[iou_idx])
                    else:
                        lkdcls += self.BCEcls(stu_pcls, tea_t[iou_idx])

                lkdobj += self.BCEobj(stu_pi[..., 4], tea_tobj)

            obji = self.BCEobj(stu_pi[..., 4], stu_tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lkdbox *= self.hyp['kdbox']
        lkdobj *= self.hyp['kdobj']
        lkdcls *= self.hyp['kdcls']

        bs = stu_tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + lkdbox + lkdcls + lkdobj
        return loss * bs, torch.cat((lbox, lobj, lcls, lkdbox, lkdcls, lkdobj, loss)).detach()


class ComputeLogitKDLossV3:
    # Compute Losses and KD Losses
    def __init__(self, teacher, student, autobalance=False):
        super(ComputeLogitKDLossV3, self).__init__()
        device = next(student.parameters()).device  # get model device
        h = student.hyp

        # Define Criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        self.student = student
        self.teacher = teacher
        self.s = de_parallel(student).model[-1]  # Detect() module
        self.t = de_parallel(teacher).model[-1]  # Detect() module

        self.balance = {3: [4.0, 1.0, 0.4]}.get(self.s.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(self.s.stride).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.hyp, self.autobalance = BCEcls, BCEobj, h, autobalance

        self.device = device

        self.temperature = h.get('temperature', 1.0)
        self.KLDcls = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def __call__(self, tea_p, stu_p, targets, imgs):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss

        lkdcls = torch.zeros(1, device=self.device)  # KD Cls Loss
        lkdbox = torch.zeros(1, device=self.device)  # KD box Loss
        lkdobj = torch.zeros(1, device=self.device)  # KD Obj Loss

        # Image index,anchor index, grid_y_index, grid_x_index, matching target, matching anchor wh
        teacher_matching = build_targets(tea_p, targets, imgs, self.teacher)
        student_matching = build_targets(stu_p, targets, imgs, self.student)
        teacher_matching = list(zip(*teacher_matching))
        student_matching = list(zip(*student_matching))

        pre_gen_gains = [torch.tensor(pp.shape, device=self.device)[[3, 2, 3, 2]] for pp in tea_p]

        for i, (tea_pi, stu_pi) in enumerate(zip(tea_p, stu_p)):  # layer index, layer predictions
            t_b, t_a, t_gj, t_gi, t_target, t_anchor = teacher_matching[i][:]
            s_b, s_a, s_gj, s_gi, s_target, s_anchor = student_matching[i][:]

            # teacher target-box and cls
            grid = torch.stack([t_gi, t_gj], dim=1)
            tea_selected_tbox = t_target[:, 2:6] * pre_gen_gains[i]
            tea_selected_tbox[:, :2] -= grid
            tea_selected_tcls = t_target[:, 1].long()

            # student target-box and cls
            grid = torch.stack([s_gi, s_gj], dim=1)
            stu_selected_tbox = s_target[:, 2:6] * pre_gen_gains[i]
            stu_selected_tbox[:, :2] -= grid
            stu_selected_tcls = s_target[:, 1].long()

            tea_tobj = torch.zeros(tea_pi.shape[:4], dtype=tea_pi.dtype, device=self.device)  # 3 x 80 x 80 x 15
            stu_tobj = torch.zeros(stu_pi.shape[:4], dtype=stu_pi.dtype, device=self.device)  # target obj

            tea_n = t_b.shape[0]  # number of targets
            stu_n = s_b.shape[0]  # number of targets

            # Student
            if stu_n:
                # target-subset of predictions
                stu_pbox, stu_pobj, stu_pcls = subset_of_pred(stu_pi, (s_b, s_a, s_gj, s_gi), s_anchor, self.s.nc)

                # Bounding Box
                iou = bbox_iou(stu_pbox.T, stu_selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectiveness
                stu_tobj[s_b, s_a, s_gj, s_gi] = (1.0 - self.student.gr) + \
                                                 self.student.gr * iou.detach().clamp(0).type(stu_tobj.dtype)

                # Classification
                if self.s.nc > 1:  # cls loss (only if multiple classes)
                    stu_t = torch.full_like(stu_pcls, self.cn, device=self.device)  # targets
                    stu_t[range(stu_n), stu_selected_tcls] = self.cp
                    lcls += self.BCEcls(stu_pcls, stu_t)  # BCE

            # Teacher
            if tea_n:
                tea_pbox, tea_pobj, tea_pcls = subset_of_pred(tea_pi, (t_b, t_a, t_gj, t_gi), t_anchor, self.t.nc)

            obji = self.BCEobj(stu_pi[..., 4], stu_tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lkdbox *= self.hyp['kdbox']
        lkdobj *= self.hyp['kdobj']
        lkdcls *= self.hyp['kdcls']

        bs = stu_tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + lkdbox + lkdcls + lkdobj
        return loss, torch.cat((lbox, lobj, lcls, lkdbox, lkdcls, lkdobj)).detach()


def smart_logit_kd_loss(teacher, student, name):
    if name == "V0":
        loss_fun = ComputeLogitKDLossV0(teacher, student)
    elif name == "V2":
        loss_fun = ComputeLogitKDLossV2(teacher, student)
    elif name == "V3":
        loss_fun = ComputeLogitKDLossV3(teacher, student)
    elif name is None:
        LOGGER.info(f"We didn't use Logit KD Loss")
        loss_fun = None
    else:
        raise NotImplementedError(f'Logit KD Loss function {name} not implemented.')

    LOGGER.info(f"{colorstr('Logit KD Loss Function:')} {type(loss_fun).__name__}")

    return loss_fun


def smart_feat_kd_loss(model, name):
    if name == "SPKD":
        loss_fun = SPKDLoss(model)
    elif name == "SCKD":
        loss_fun = SCKDLoss(model)
    elif name is None:
        LOGGER.info(f"We didn't use Feature KD Loss")
        loss_fun = None
    else:
        raise NotImplementedError(f'Feature KD Loss function {name} not implemented.')

    LOGGER.info(f"{colorstr('Feature KD Loss Function:')} {type(loss_fun).__name__}")
    return loss_fun


if __name__ == "__main__":
    pass
