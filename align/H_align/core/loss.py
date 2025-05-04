import torch
from H_align.core.utils import torch_skew_symmetric
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def batch_episym(x1, x2, F):
    batch_size, num_pts = x1.shape[0], x1.shape[1]
    x1 = torch.cat([x1, x1.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    x2 = torch.cat([x2, x2.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    F = F.reshape(-1,1,3,3).repeat(1,num_pts,1,1)
    x2Fx1 = torch.matmul(x2.transpose(2,3), torch.matmul(F, x1)).reshape(batch_size,num_pts)
    Fx1 = torch.matmul(F,x1).reshape(batch_size,num_pts,3)
    Ftx2 = torch.matmul(F.transpose(2,3),x2).reshape(batch_size,num_pts,3)
    ys = x2Fx1**2 * (
            1.0 / (Fx1[:, :, 0]**2 + Fx1[:, :, 1]**2 + 1e-15) +
            1.0 / (Ftx2[:, :, 0]**2 + Ftx2[:, :, 1]**2 + 1e-15))
    return ys

class MatchLoss(object):
  def __init__(self, config):
    self.loss_essential = config.loss_essential
    self.loss_classif = config.loss_classif
    self.use_fundamental = config.use_fundamental
    self.obj_geod_th = config.obj_geod_th
    self.geo_loss_margin = config.geo_loss_margin
    self.loss_essential_init_iter = config.loss_essential_init_iter

  def run(self, global_step, data, logits, e_hat):
    R_in, t_in, y_in, pts_virt = data['Rs'], data['ts'], data['ys'], data['virtPts']
    # Get groundtruth Essential matrix
    e_gt_unnorm = torch.reshape(torch.matmul(
        torch.reshape(torch_skew_symmetric(t_in), (-1, 3, 3)),
        torch.reshape(R_in, (-1, 3, 3))
    ), (-1, 9))

    e_gt = e_gt_unnorm / torch.norm(e_gt_unnorm, dim=1, keepdim=True)

    ess_hat = e_hat
    if self.use_fundamental:
        ess_hat = torch.matmul(torch.matmul(data['T2s'].transpose(1,2), ess_hat.reshape(-1,3,3)),data['T1s'])
        # get essential matrix from fundamental matrix
        ess_hat = torch.matmul(torch.matmul(data['K2s'].transpose(1,2), ess_hat.reshape(-1,3,3)),data['K1s']).reshape(-1,9)
        ess_hat = ess_hat / torch.norm(ess_hat, dim=1, keepdim=True)

    # we do not use the l2 loss, just save the value for convenience 
    L2_loss = torch.mean(torch.min(
        torch.sum(torch.pow(ess_hat - e_gt, 2), dim=1),
        torch.sum(torch.pow(ess_hat + e_gt, 2), dim=1)
    ))
    
    # The groundtruth epi sqr
    gt_geod_d = y_in[:, :]
    is_pos = (gt_geod_d < self.obj_geod_th).type(logits.type())
    is_neg = (gt_geod_d >= self.obj_geod_th).type(logits.type())
    
    pts1_virts, pts2_virts = pts_virt[:, :, :2], pts_virt[:,:,2:]
    geod = batch_episym(pts1_virts, pts2_virts, e_hat)
    essential_loss = torch.min(geod, self.geo_loss_margin*geod.new_ones(geod.shape))
    essential_loss = essential_loss.mean()

#    essential_loss = contrastive_loss(self.geo_loss_margin, is_pos, data['xs'], e_hat)
    # Classification loss
    with torch.no_grad(): 
        pos = torch.sum(is_pos)
        pos_num = F.relu(pos - 1) + 1
        neg = torch.sum(is_neg)
        neg_num = F.relu(neg - 1) + 1
        pos_w = neg_num / pos_num

    classif_loss = binary_cross_entropy_with_logits(logits, is_pos, pos_weight=pos_w, reduce=True)
    
    precision = torch.mean(
        torch.sum((logits > 0).type(is_pos.type()) * is_pos, dim=1) /
        torch.sum((logits > 0).type(is_pos.type()) * (is_pos + is_neg), dim=1)
    )
    recall = torch.mean(
        torch.sum((logits > 0).type(is_pos.type()) * is_pos, dim=1) /
        torch.sum(is_pos, dim=1)
    )

    loss = 0
    # Check global_step and add essential loss
    if self.loss_essential > 0 and global_step >= self.loss_essential_init_iter:
        loss += self.loss_essential * essential_loss 
    if self.loss_classif > 0:
        loss += self.loss_classif * classif_loss

    return [loss, (self.loss_essential * essential_loss).item(), (self.loss_classif * classif_loss).item(), L2_loss.item(), precision.item(), recall.item()]

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                     reduce=False, reduction='elementwise_mean', pos_weight=None):

    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)

    if pos_weight is None:
        ce_loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    else:
        log_weight = 1 + (pos_weight - 1) * target
        ce_loss = input - input * target + log_weight * (max_val + ((-max_val).exp() + (-input - max_val).exp()).log())
            
    if weight is not None:
        ce_loss = ce_loss * weight

    if reduction == False:
        return ce_loss
    elif reduction == 'elementwise_mean':
        return ce_loss.mean()
    else:
        return ce_loss.sum()

def contrastive_loss(ess_loss_margin, label, xs, E):
    pts1, pts2 = xs[:, 0, :, :2], xs[:, 0, :, 2:]
  
    geod = batch_episym(pts1, pts2, E).view(-1)
    label = label.view(-1)
    geod_p = torch.nonzero(label)
    geod_n = torch.nonzero(1 - label)
    geod_p = geod[geod_p]
    geod_n = geod[geod_n]

    zero_p = torch.zeros(geod_p.size()).type(geod_p.type())
    zero_n = torch.zeros(geod_n.size()).type(geod_n.type())

    loss_p = torch.sum(torch.max(geod_p - 1e-4, zero_p))
    loss_p = torch.min(loss_p, ess_loss_margin*loss_p.new_ones(loss_p.shape))
    loss_n = torch.sum(torch.max(1e-4 - geod_n, zero_n))
    loss_p = torch.min(loss_n, ess_loss_margin*loss_n.new_ones(loss_n.shape))

    essential_loss = loss_p.mean() + loss_n.mean()
 
    return essential_loss

class Loss_v3(object):
    def __init__(self, config):
        self.loss_essential = config.loss_essential
        self.loss_classif = config.loss_classif
        self.loss_A = config.loss_A
        self.use_fundamental = config.use_fundamental
        self.obj_geod_th = config.obj_geod_th
        self.geo_loss_margin = config.geo_loss_margin
        self.ess_loss_margin = config.ess_loss_margin
        self.loss_essential_init_iter = config.loss_essential_init_iter
        self.loss_EpipolarGeometryDis = config.loss_EpipolarGeometryDis

    def A_loss(self, A, label):
        bs, _, _ = A.size()
        label = label.float().unsqueeze(-1)
        label = torch.bmm(label, label.permute(0, 2, 1))
        
        label = label.view(bs, -1)
        A = A.view(bs, -1)

        with torch.no_grad(): 
            pos = torch.sum(label)
            pos_num = F.relu(pos - 1) + 1
            total = torch.numel(label)
            neg_num = F.relu(total - pos - 1) + 1
            pos_w = neg_num / pos_num

        classi_loss = binary_cross_entropy_with_logits(A, label, pos_weight=pos_w, reduce=True)
        return classi_loss

    def run(self, global_step, data, logits, e_hat, A_hat, x_new, flag_loss_EGD):
        
        R_in, t_in, y_in, pts_virt = data['Rs'], data['ts'], data['ys'], data['virtPts']
        
        # Get groundtruth Essential matrix
        e_gt_unnorm = torch.reshape(torch.matmul(
            torch.reshape(torch_skew_symmetric(t_in), (-1, 3, 3)),
            torch.reshape(R_in, (-1, 3, 3))
        ), (-1, 9))
		
        e_gt = e_gt_unnorm / torch.norm(e_gt_unnorm, dim=1, keepdim=True)
        if flag_loss_EGD :
            x2fx1 = batch_episym(x_new[:, :, :, :2].squeeze(1), x_new[:, :, :, 2:].squeeze(1), e_gt)
            #dis = x_new.squeeze(1) - data['xs']
            # f_gt = e_gt_unnorm.reshape(-1, 1, 3, 3).repeat(1, x_new.shape[2], 1, 1)
            # x1 =  torch.cat([x_new[:, :, :, :2], x_new.new_ones([x_new.shape[0], x_new.shape[1], x_new.shape[2], 1]).detach()], dim=-1)
            # x2 =  torch.cat([x_new[:, :, :, 2:], x_new.new_ones([x_new.shape[0], x_new.shape[1], x_new.shape[2], 1]).detach()], dim=-1)
            # x2fx1 = torch.matmul(x2.permute(0, 2, 1, 3), torch.matmul(f_gt, x1 .permute(0, 2, 3, 1))).squeeze()


        ess_hat = e_hat
        if self.use_fundamental:
            ess_hat = torch.matmul(torch.matmul(data['T2s'].transpose(1,2), ess_hat.reshape(-1,3,3)),data['T1s'])
            # get essential matrix from fundamental matrix
            ess_hat = torch.matmul(torch.matmul(data['K2s'].transpose(1,2), ess_hat.reshape(-1,3,3)),data['K1s']).reshape(-1,9)
            ess_hat = ess_hat / torch.norm(ess_hat, dim=1, keepdim=True)


        # Essential/Fundamental matrix loss
        pts1_virts, pts2_virts = pts_virt[:, :, :2], pts_virt[:, :, 2:]
     
        geod = batch_episym(pts1_virts, pts2_virts, e_hat)
        essential_loss = torch.min(geod, self.geo_loss_margin*geod.new_ones(geod.shape))
        essential_loss = essential_loss.mean()

        # Classification loss
        # The groundtruth epi sqr
        gt_geod_d = y_in[:, :]
        is_pos = (gt_geod_d < self.obj_geod_th).type(e_hat.type())
        is_neg = (gt_geod_d >= self.obj_geod_th).type(e_hat.type())


        
    #    essential_loss = contrastive_loss(self.ess_loss_margin, is_pos, data['xs'], e_hat)
        with torch.no_grad(): 
            pos = torch.sum(is_pos, dim=-1, keepdim=True)
            pos_num = F.relu(pos - 1) + 1
            neg = torch.sum(is_neg, dim=-1, keepdim=True)
            neg_num = F.relu(neg - 1) + 1
            pos_w = neg_num / pos_num
            
        classif_loss = 0    
        for i in range(len(logits)):
            classif_loss += binary_cross_entropy_with_logits(logits[i], is_pos, pos_weight=pos_w, reduce=True)

        with torch.no_grad():
            precision = torch.mean(
                torch.sum((logits[-1] > 0).type(is_pos.type()) * is_pos, dim=1) /
                torch.sum((logits[-1] > 0).type(is_pos.type()) * (is_pos + is_neg), dim=1)
            )
            recall = torch.mean(
                torch.sum((logits[-1] > 0).type(is_pos.type()) * is_pos, dim=1) /
                torch.sum(is_pos, dim=1)
            )

    #    A_loss = self.A_loss(A_hat, is_pos)
        A_loss = essential_loss.new_ones(1)
        loss = 0
        # Check global_step and add essential loss
        if self.loss_essential > 0 and global_step >= self.loss_essential_init_iter:
            loss += self.loss_essential * essential_loss 
        if self.loss_classif > 0:
            loss += self.loss_classif * classif_loss
        if flag_loss_EGD > 0:
            loss_egd = torch.mean(x2fx1.flatten()[is_pos.flatten() > 0])
            loss += loss_egd

            #loss_dis = torch.mean(torch.exp(dis))
            # loss += loss_dis
        else:
            loss_egd = 0.0
    #    loss += self.loss_A * A_loss
        return loss, essential_loss, classif_loss, A_loss, precision, recall, loss_egd

class Loss_v2(nn.Module):
    def __init__(self, config):
        super(Loss_v2, self).__init__()
        self.obj_geod_th = config.obj_geod_th 

    def loss_classi(self, output, label):

        pos = torch.sum(label)
        pos_num = F.relu(pos - 1) + 1
        total = torch.numel(label)
        neg_num = F.relu(total - pos - 1) + 1
        pos_w = neg_num / pos_num

        classi_loss = binary_cross_entropy_with_logits(output, label, pos_weight=pos_w, reduce=True)
        
        return classi_loss

    def forward(self, output, A, ys):
        gt_geod_d = ys[:, :]
        label = (gt_geod_d < self.obj_geod_th).type(output.type())

        loss1 = self.loss_classi(output, label)

        mask = (output > 0).float()

        p = torch.sum(mask * label, dim=-1) / (torch.sum(mask, dim=-1) + 1e-5)
 
        r = torch.sum(mask * label, dim=-1) / (torch.sum(label, dim=-1)  + 1e-5)

        return loss1, p.mean(), r.mean()