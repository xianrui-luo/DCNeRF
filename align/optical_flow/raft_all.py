#!/usr/bin/env python
# encoding: utf-8

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from skimage import exposure
from models.raft import RAFT
from models.utils import flow_viz
from models.utils.utils import InputPadder, bilinear_sampler,coords_grid


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Bokeh Rendering = Training Stage', fromfile_prefix_chars='@')


parser.add_argument('--model', default='models/raft-things.pth')
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

args = parser.parse_args()

def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()
    cv2.imwrite('demo.jpg', img_flo[:, :, [2,1,0]])  # pjw add


def viz_flow(flo):
    flo = flo[0].permute(1,2,0).cpu().numpy()
    flo = flow_viz.flow_to_image(flo)
    return flo


def transform(img1, flo1):
    #img:[B,C,H,W] flo:[B,2,H,W]
    img=img1.clone()
    flo=flo1.clone()
    
    flo[:, 0] = flo[:, 0] / ((flo.shape[3] - 1) / 2)
    flo[:, 1] = flo[:, 1] / ((flo.shape[2] - 1) / 2)
    theta = torch.tensor([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img.size()).cuda()
    grid = (grid + flo.transpose(1, 2).transpose(2, 3)).clamp(min=-1, max=1)
    # grid = (grid - flo.transpose(1, 2).transpose(2, 3)).clamp(min=-1, max=1)
    img = F.grid_sample(img, grid)
    return img
def findpixel(img1, flo1):
    #img:[B,C,H,W] flo:[B,2,H,W]


    img=img1.clone()
    flo=flo1.clone()
    # import torch.nn.functional as F
    flo[:, 0] = flo[:, 0] / ((flo.shape[3] - 1) / 2)
    flo[:, 1] = flo[:, 1] / ((flo.shape[2] - 1) / 2)
    theta = torch.tensor([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img.size()).cuda()
    grid = (grid + flo.transpose(1, 2).transpose(2, 3)).clamp(min=-1, max=1)

    
    # grid = (grid - flo.transpose(1, 2).transpose(2, 3)).clamp(min=-1, max=1)
    # img = F.grid_sample(img, grid)
    # cv2.imwrite('demo.png', img[0].cpu().numpy().transpose(1, 2, 0)[..., ::-1])
    return grid


def length_sq(x):
    return torch.sum(torch.square(x),1,keepdims=True)

def occlusion(flow_fw, flow_bw):
    mag_sq = length_sq(flow_fw) + length_sq(flow_bw)
    flow_bw_warped = transform(flow_bw, flow_fw)
    flow_fw_warped = transform(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    # occ_thresh =  0.6*mag_sq + 30
    # occ_thresh =  0.01 * mag_sq + 5
    occ_thresh =  0.01 * mag_sq + 0.5
    occ_fw = length_sq(flow_diff_fw) > occ_thresh
    occ_bw = length_sq(flow_diff_bw) > occ_thresh
    return occ_fw, occ_bw



def plot(x_img, y_img):
    x_img = cv2.cvtColor(x_img, cv2.COLOR_RGB2GRAY)
    y_img = cv2.cvtColor(y_img, cv2.COLOR_RGB2GRAY)
    xs = x_img.flatten()
    ys = y_img.flatten()
    plt.scatter(xs[::100], ys[::100], s=1, alpha=0.2)
    plt.show()


def demo(args):
    raft = torch.nn.DataParallel(RAFT(args))
    raft.load_state_dict(torch.load(args.model))

    raft = raft.module
    raft.cuda()
    raft.eval()

    root = '/data2/luoxianrui/data/registration_deblur/focal_stack_final'

    wide_root = os.path.join(root, 'wide_match_fg')
    # wide_root = os.path.join(root, 'original/bg')
    main_root = os.path.join(root, 'original/fg')  # fg, bg

    save_warp_root = os.path.join(root, 'wide_warp_fg')  # fg, bg
    # save_warp_root = os.path.join(root, 'original_bg_flow')  # fg, bg

    save_confidence_root = os.path.join(root, 'consistency_fg')  # fg, bg

    os.makedirs(save_warp_root, exist_ok=True)
    os.makedirs(save_confidence_root, exist_ok=True)


    wide_paths = [os.path.join(wide_root, name) for name in sorted(os.listdir(wide_root))]
    main_paths = [os.path.join(main_root, name) for name in sorted(os.listdir(main_root))]
    save_warp_paths = [os.path.join(save_warp_root, name) for name in sorted(os.listdir(main_root))]
    save_conf_paths = [os.path.join(save_confidence_root, name) for name in sorted(os.listdir(main_root))]


    print(len(wide_paths))

    assert len(wide_paths) == len(main_paths)

    for i in tqdm(range(len(wide_paths))):
        start=time.time()

        wide_path = wide_paths[i]
        main_path = main_paths[i]

        save_conf_path = save_conf_paths[i]
        save_warp_path = save_warp_paths[i]



        wide = cv2.imread(wide_path).astype(np.float32) / 255.0
        wide = cv2.cvtColor(wide, cv2.COLOR_BGR2RGB)
        main = cv2.imread(main_path).astype(np.float32) / 255.0
        main = cv2.cvtColor(main, cv2.COLOR_BGR2RGB)

        h_wide_ori, w_wide_ori = wide.shape[:2]
        h_main_ori, w_main_ori = main.shape[:2]
        k=0
        #if h>w 
        if h_main_ori>w_main_ori:
            k=1
            wide= np.rot90(wide,1).copy()
            main= np.rot90(main,1).copy()


        # if use color transform
        multi = True if wide.shape[-1] > 1 else False
        wide=exposure.match_histograms(wide,main,multichannel=multi)#multichannel=True if RGB images

        wide = torch.from_numpy(wide).permute(2, 0, 1).unsqueeze(0).contiguous()
        main = torch.from_numpy(main).permute(2, 0, 1).unsqueeze(0).contiguous()
        ratio = 4


        with torch.no_grad():
            wide = wide.cuda()
            main = main.cuda()
            h_wide_ori, w_wide_ori = wide.shape[2:]
            h_main_ori, w_main_ori = main.shape[2:]


            h_wide_re, w_wide_re = h_wide_ori // ratio, w_wide_ori // ratio
            
            h_main_re, w_main_re = h_main_ori // ratio, w_main_ori // ratio

            wide_low = F.interpolate(wide, (h_wide_re, w_wide_re), mode='bilinear', align_corners=False)
            main_low = F.interpolate(main, (h_main_re, w_main_re), mode='bilinear', align_corners=False)

            h_pad = (h_wide_re // 8 + 1) * 8
            w_pad = (w_wide_re // 8 + 1) * 8

            pad_top_wide = (h_pad - h_wide_re) // 2
            pad_bottom_wide = h_pad - h_wide_re - pad_top_wide
            pad_left_wide = (w_pad - w_wide_re) // 2
            pad_right_wide = w_pad - w_wide_re - pad_left_wide

            pad_top_main = (h_pad - h_main_re) // 2
            pad_bottom_main = h_pad - h_main_re - pad_top_main
            pad_left_main = (w_pad - w_main_re) // 2
            pad_right_main = w_pad - w_main_re - pad_left_main

            wide_pad = nn.ReplicationPad2d((pad_left_wide, pad_right_wide, pad_top_wide, pad_bottom_wide))(wide_low)
            main_pad = nn.ReplicationPad2d((pad_left_main, pad_right_main, pad_top_main, pad_bottom_main))(main_low)


            _,flow_up_low_m2w=raft(main_pad*255,wide_pad*255,iters=20,test_mode=True)#如果从广角到主摄，应该是算主摄到广角的光流
            _,flow_up_low_w2m=raft(wide_pad*255,main_pad*255,iters=20,test_mode=True)#如果从广角到主摄，应该是算主摄到广角的光流

            flow_up_low_m2w = flow_up_low_m2w[..., pad_top_wide:-pad_bottom_wide, pad_left_wide:-pad_right_wide]
            flow_up_low_w2m = flow_up_low_w2m[..., pad_top_wide:-pad_bottom_wide, pad_left_wide:-pad_right_wide]

            
            flow_up_m2w = F.interpolate(flow_up_low_m2w, (h_wide_ori, w_wide_ori), mode='bilinear', align_corners=True) * ratio
            flow_up_w2m = F.interpolate(flow_up_low_w2m, (h_wide_ori, w_wide_ori), mode='bilinear', align_corners=True) * ratio


            confidence=occlusion(flow_up_m2w,flow_up_w2m)[0].float()
            confidence=1-confidence
            wide_warp = transform(wide* 255, flow_up_m2w) / 255



            h_main_low, w_main_low = main.shape[2:]
            h_wide_low, w_wide_low = wide.shape[2:]


            d_top = (h_wide_low - h_main_low) // 2
            d_bottom = h_wide_low - h_main_low - d_top
            d_left = (w_wide_low - w_main_low) // 2
            d_right = w_wide_low - w_main_low - d_left
            # print(d_top)


            wide_warp = wide_warp[..., d_top:-d_bottom, d_left:-d_right]
            confidence = confidence[..., d_top:-d_bottom, d_left:-d_right]



        wide = wide[0].cpu().clone().permute(1, 2, 0).numpy()
        main = main[0].cpu().clone().permute(1, 2, 0).numpy()
        wide_warp = wide_warp[0].cpu().clone().permute(1, 2, 0).numpy()
        confidence = confidence[0].cpu().clone().squeeze(0).numpy()
        flow_up_low_m2w = flow_up_low_m2w[0].cpu().clone().permute(1, 2, 0).numpy()
        
        wide_warp = cv2.cvtColor(wide_warp, cv2.COLOR_RGB2BGR) * 255
        wide = cv2.cvtColor(wide, cv2.COLOR_RGB2BGR) * 255
        if k==1:

            wide_warp=np.rot90(wide_warp,3)

        cv2.imwrite(save_conf_path, confidence*255)
        cv2.imwrite(save_warp_path, wide_warp)
        print('计时: ', round(time.time() - start, 0), '秒', end="\r")


if __name__ == '__main__':

    demo(args)