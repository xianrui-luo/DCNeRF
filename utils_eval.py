import numpy as np
import cv2
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

# from pytorch_ssim import ssim
from pytorch_msssim import SSIM  # see details in https://pypi.org/project/pytorch-msssim/
from torch.nn import functional as F
import torch.nn as nn
from PIL import Image 
import numpy as np
from skimage.io import imsave


from torch.autograd import Variable
from torchvision import models
from collections import namedtuple
import pdb

import time
import random

import copy

class LaplacianRegularizer(nn.Module):
    def __init__(self):
        super(LaplacianRegularizer, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='sum')

    def forward(self, f):
        loss = 0.
        for i in range(f.shape[2]):
            for j in range(f.shape[3]):
                up = max(i-1, 0)
                down = min(i+1, f.shape[2] - 1)
                left = max(j-1, 0)
                right = min(j+1, f.shape[3] - 1)
                term = f[:,:,i,j].view(f.shape[0], f.shape[1], 1, 1).\
                        expand(f.shape[0], f.shape[1], down - up+1, right-left+1)
                loss += self.mse_loss(term, f[:, :, up:down+1, left:right+1])
        return loss

def resize(img, size=512, strict=False):
    short = min(img.shape[:2])
    scale = size/short
    if not strict:
        img = cv2.resize(img, (round(
            img.shape[1]*scale), round(img.shape[0]*scale)), interpolation=cv2.INTER_NEAREST)
    else:
        img = cv2.resize(img, (size,size), interpolation=cv2.INTER_NEAREST)
    return img


def crop(img, size=512):
    try:
        y, x = random.randint(
            0, img.shape[0]-size), random.randint(0, img.shape[1]-size)
    except Exception as e:
        y, x = 0, 0
    return img[y:y+size, x:x+size, :]


def load_image(filename, size=None, use_crop=False):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size:
        img = resize(img, size=size)
    if use_crop:
        img = crop(img, size)
    return img

def get_latest_ckpt(path):
    try:
        list_of_files = glob.glob(os.path.join(path,'*')) 
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    except ValueError:
        return None

def save_params(state, params):
    state['model_params'] = params
    return state

def load_params(state):
    params = state['model_params']
    del state['model_params']
    return state, params




ssim_module = SSIM(data_range=1, size_average=True, channel=3)


def l1_loss_func(pred, gt, weight=None):
    if weight is not None:
        return torch.mean(torch.abs(pred - gt) * weight) / torch.mean(weight)
    else:
        return torch.mean(torch.abs(pred - gt))


def log_l1_loss_func(pred, gt, weight=None):
    if weight is not None:
        return torch.log10(torch.mean(torch.abs(pred - gt) * weight) / torch.mean(weight))
    else:
        return torch.log10(torch.mean(torch.abs(pred - gt)))


def grad_x(image):
    image = F.pad(image, [0, 1, 0, 0], mode="replicate")
    return image[:, :, :, :-1] - image[:, :, :, 1:]


def grad_y(image):
    image = F.pad(image, [0, 0, 0, 1], mode="replicate")
    return image[:, :, :-1, :] - image[:, :, 1:, :]


def grad_loss_func(pred, gt, weight=None):
    if weight is not None:
        return 0.5 * torch.mean(torch.abs(grad_x(pred) - grad_x(gt)) * weight) / torch.mean(weight) + \
               0.5 * torch.mean(torch.abs(grad_y(pred) - grad_y(gt)) * weight) / torch.mean(weight)
    else:
        return 0.5 * torch.mean(torch.abs(grad_x(pred) - grad_x(gt))) + \
               0.5 * torch.mean(torch.abs(grad_y(pred) - grad_y(gt)))


def log_grad_loss_func(pred, gt, weight=None):
    if weight is not None:
        return 0.5 * torch.log10(torch.mean(torch.abs(grad_x(pred) - grad_x(gt)) * weight) / torch.mean(weight)) + \
               0.5 * torch.log10(torch.mean(torch.abs(grad_y(pred) - grad_y(gt)) * weight) / torch.mean(weight))
    else:
        return 0.5 * torch.log10(torch.mean(torch.abs(grad_x(pred) - grad_x(gt)))) + \
               0.5 * torch.log10(torch.mean(torch.abs(grad_y(pred) - grad_y(gt))))


def mask_loss_func(mask):
    return torch.mean(torch.abs(grad_x(mask)) + torch.abs(grad_y(mask)))


def grad_smooth_loss_func(pred, mask):
    return torch.mean((torch.abs(grad_x(pred)) + torch.abs(grad_y(pred))) * mask) / (torch.mean(mask) + 1e-10)



# def psnr_loss_func(pred, gt):
#     mse = torch.mean((pred - gt) ** 2)
#     return 20 * torch.log10(1.0 / (torch.sqrt(mse) + 1e-10))


def ssim_loss_func(pred, gt):
    # return ssim(pred, gt)
    return 1 - ssim_module(pred, gt)


def regular_loss_func(residual, weight):
    if weight is not None:
        return torch.mean(weight * torch.abs(residual)) / torch.mean(weight)
    else:
        return torch.mean(torch.abs(residual))


def log_regular_loss_func(residual, weight):
    if weight is not None:
        return torch.log10((torch.mean(weight * torch.abs(residual)) + 0.001) / torch.mean(weight))
    else:
        return torch.log10(torch.mean(torch.abs(residual)) + 0.001)

# def residual_grad_loss_func(residual):
#     return torch.mean(torch.abs(grad_x(residual)) + torch.abs(grad_y(residual))) / 2


def reconstruct_loss_func(pred, pred_adapt):
    pred_rec = F.interpolate(pred, size=(pred_adapt.shape[2], pred_adapt.shape[3]), mode='bilinear', align_corners=False, recompute_scale_factor=True)
    return F.l1_loss(pred_rec, pred_adapt)


def calc_rel(pred, gt, data_range=1):
    if data_range == 1:
        pred = pred * 255
        gt = gt * 255
    return torch.mean(torch.abs(pred - gt) / torch.clamp(gt, 1.0, 255.0)).item()


def calc_mae(pred, gt, data_range=1):
    if data_range == 1:
        pred = pred * 255
        gt = gt * 255
    return torch.mean(torch.abs(pred - gt)).item()


def calc_rmse(pred, gt, data_range=1):
    if data_range == 1:
        pred = pred * 255
        gt = gt * 255
    return torch.sqrt(torch.mean((pred - gt) ** 2)).item()


def calc_psnr(pred, gt, data_range=1, mask=None):
    if data_range == 1:
        pred = pred * 255
        gt = gt * 255
    if mask is None:
        mask = torch.ones_like(pred)
    mse = torch.mean((pred - gt) ** 2 * mask) / torch.mean(mask)
    return 20 * torch.log10(255.0 / (torch.sqrt(mse) + 1e-10)).item()


def calc_ssim(pred, gt, data_range=1,mask=None):
    if data_range == 255:
        pred = pred / 255
        gt = gt / 255
    return ssim_module(pred, gt).item()

def mask_loss_func(mask):
    return torch.mean(torch.abs(grad_x(mask)) + torch.abs(grad_y(mask)))
    


def create_loss_model(vgg, end_layer, use_maxpool=True): # end_layer = 3, 8, 15, 22 (relu1_2, relu2_2, relu3_3, relu4_3)
    vgg = copy.deepcopy(vgg)
    model = nn.Sequential()
    i = 0

    for layer in list(vgg):

        if i > end_layer:
            break

        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            if use_maxpool:
                model.add_module(name, layer)
            else:
                avgpool = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
                model.add_module(name, avgpool)
        i += 1
    return model


class AdversarialLoss(nn.Module):
    """
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs).to(outputs.device)
            loss = self.criterion(outputs, labels)
            return loss


def gradient_x(img):
    img = F.pad(img, (0, 1, 0, 0), mode="replicate")
    return img[:,:,:,:-1] - img[:,:,:,1:]


def gradient_y(img):
    img = F.pad(img, (0, 0, 0, 1), mode="replicate")
    return img[:,:,:-1,:] - img[:,:,1:,:]


def get_depth_smoothness(pred, images, regularizer='Sigmoid'):
    depth_gradient_x = gradient_x(pred)
    depth_gradient_y = gradient_y(pred)
    image_gradient_x = gradient_x(images)
    image_gradient_y = gradient_y(images)
    #weight_x = torch.exp(-torch.mean(torch.abs(image_gradient_x), 1, keepdim=True))
    #weight_y = torch.exp(-torch.mean(torch.abs(image_gradient_y), 1, keepdim=True))

    if regularizer == 'Sigmoid':
        weight_x = torch.exp(-torch.mean(torch.abs(image_gradient_x), 1, keepdim=True))
        weight_y = torch.exp(-torch.mean(torch.abs(image_gradient_y), 1, keepdim=True))
    elif regularizer == 'Logistic':
        weight_x = torch.log(1 + torch.exp(-torch.mean(torch.abs(image_gradient_x), 1, keepdim=True)))
        weight_y = torch.log(1 + torch.exp(-torch.mean(torch.abs(image_gradient_y), 1, keepdim=True)))
    elif regularizer == 'Exp':
        c = 1.0
        weight_x = torch.exp(-torch.mean(torch.pow(image_gradient_x/c, 2)/2., 1, keepdim=True))
        weight_y = torch.exp(-torch.mean(torch.pow(image_gradient_y/c, 2)/2., 1, keepdim=True))
    else:
        weight_x = torch.exp(-torch.mean(torch.abs(image_gradient_x), 1, keepdim=True))
        weight_y = torch.exp(-torch.mean(torch.abs(image_gradient_y), 1, keepdim=True))

    smoothness_x = torch.mean(torch.abs(depth_gradient_x * weight_x))
    smoothness_y = torch.mean(torch.abs(depth_gradient_y * weight_y))
    return smoothness_x + smoothness_y


def pyramid_image_guidance_regularizer(preds, images, scales=4, regularizer='Sigmoid'):
    regularization = 0.

    for scale in range(scales):
        step = pow(2, scale)
        regularization += get_depth_smoothness(preds[:,:,::step,::step], images[:,:,::step,::step], regularizer)

    return regularization