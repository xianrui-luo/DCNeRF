import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np

from imageio import imread
import h5py

import json
import cv2
import pickle
from tqdm import trange
from extract_patches.core import extract_patches
import time
from tqdm import tqdm
from H_align.model.HardNet import HardNet
from H_align.model.Utils import cv2_scale, np_reshape
import torchvision.transforms as transforms

from H_align.config import get_config, print_usage
from H_align.methods import feature_matching as matching

from H_align.core.config import get_config as get_nmnet_config
from H_align.model.model_new1 import NM_Net_v2 as NM_Net
# from core.utils import tocuda
from matplotlib.patches import ConnectionPatch

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

import argparse
from skimage import exposure
from optical_flow.models.raft import RAFT
from optical_flow.models.utils import flow_viz
from optical_flow.models.utils.utils import InputPadder, bilinear_sampler,coords_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description='Bokeh Rendering = Training Stage', fromfile_prefix_chars='@')


parser.add_argument('--model', default='optical_flow/models/raft-things.pth')
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

args = parser.parse_args()


def get_transforms(color):

    MEAN_IMAGE = 0.443728476019
    STD_IMAGE = 0.20197947209

    transform = transforms.Compose([
        transforms.Lambda(cv2_scale),
        transforms.Lambda(np_reshape), transforms.ToTensor(),
        transforms.Normalize((MEAN_IMAGE,), (STD_IMAGE,))
    ])

    return transform

def l_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)

def get_SIFT_keypoints(sift, img, lower_detection_th=False):

    # convert to gray-scale and compute SIFT keypoints
    keypoints = sift.detect(img, None)

    response = np.array([kp.response for kp in keypoints])
    respSort = np.argsort(response)[::-1]

    pt = np.array([kp.pt for kp in keypoints])[respSort]
    size = np.array([kp.size for kp in keypoints])[respSort]
    angle = np.array([kp.angle for kp in keypoints])[respSort]
    response = np.array([kp.response for kp in keypoints])[respSort]

    return pt, size, angle, response

def np_skew_symmetric(v):
    zero = np.zeros_like(v[:, 0])

    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)

    return M

def get_episym(x1, x2, dR, dt):
    num_pts = len(x1)

    # Make homogeneous coordinates
    x1 = np.concatenate([
        x1, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)
    x2 = np.concatenate([
        x2, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)

    # Compute Fundamental matrix
    dR = dR.reshape(1, 3, 3)
    dt = dt.reshape(1, 3)

    F = np.repeat(np.matmul(np.reshape(np_skew_symmetric(dt), (-1, 3, 3)), dR
                            ).reshape(-1, 3, 3), num_pts, axis=0)

    x2Fx1 = np.matmul(x2.transpose(0, 2, 1), np.matmul(F, x1)).flatten()
    Fx1 = np.matmul(F, x1).reshape(-1, 3)
    Ftx2 = np.matmul(F.transpose(0, 2, 1), x2).reshape(-1, 3)

    ys = x2Fx1 ** 2 * (
            1.0 / (Fx1[..., 0] ** 2 + Fx1[..., 1] ** 2) +
            1.0 / (Ftx2[..., 0] ** 2 + Ftx2[..., 1] ** 2))

    return ys.flatten()


def corresponing_idx(images, name1, name2):
    warn = ['False', 'False']
    idx1, idx2 = 0, 0
    for i in images:
        if images[i].name == name1 + '.jpg':
            idx1 = int(images[i].camera_id)
            warn[0] = 'True'
        if images[i].name == name2 + '.jpg':
            idx2 = int(images[i].camera_id)
            warn[1] = 'True'
    return idx1, idx2, warn


def norm_input(x):
    x_mean = np.mean(x, axis=0)
    dist = x - x_mean
    meandist = np.sqrt((dist ** 2).sum(axis=1)).mean()
    f = meandist / np.sqrt(2)
    cx = x_mean[0]
    cy = x_mean[1]
    return f, cx, cy


def compute_matches(descs1, descs2, cfg, kps1=None, kps2=None):
    if cfg.num_opencv_threads > 0:
        cv2.setNumThreads(cfg.num_opencv_threads)

    # Get matches through the matching module defined in the function argument
    method_match = cfg.method_dict['config_{}_{}'.format(
        cfg.dataset, cfg.task)]['matcher']['method']
    matches, ellapsed = getattr(matching,
                                method_match).match(descs1, descs2, cfg, kps1,
                                                    kps2)

    return matches, ellapsed

def load_testing_data(method):

    original_dir=''# main camera
    wide_dir='' # ultra-wide camera

    new_wide_dir='/' # save path
    if not os.path.exists(new_wide_dir):
        os.makedirs(new_wide_dir)
    filelist=os.listdir(original_dir)

    for name in filelist:
        rot=0
        img2_path = os.path.join(original_dir,name)
        img1_path = os.path.join(wide_dir,name)

        outFilename = name
        
        outfilepath=os.path.join(new_wide_dir,outFilename)
        print("Saving aligned image : ", outfilepath); 

        print('extract keypoints and descriptors')
        
        #### extract keypoints and descriptors ####

        # sift2 = cv2.xfeatures2d.SIFT_create(
        sift2 = cv2.SIFT_create(
            contrastThreshold=-10000, edgeThreshold=-10000)
        img2 = cv2.imread(img2_path)
        img1 = cv2.imread(img1_path)


        h,w,_=img2.shape
        if h>w:
            rot=1
            img1=np.rot90(img1,1)
            img2=np.rot90(img2,1)
            
        ## target resolution
        w=3648
        h=2736
        new_h=h//4
        new_w=w//4

        img2_low=cv2.resize(img2,(new_w,new_h))

        img2_gray = cv2.cvtColor(img2_low, cv2.COLOR_BGR2RGB)
        img2_gray = cv2.cvtColor(l_clahe(img2_gray), cv2.COLOR_RGB2GRAY)
        keypoints2, scales2, _, scores2 = get_SIFT_keypoints(sift2, img2_gray)
        kpts2 = [
            cv2.KeyPoint(
                x=keypoints2[i][0],
                y=keypoints2[i][1],
                _size=scales2[i],
                _angle=0) for i, point in enumerate(keypoints2)
        ]
        patches2 = extract_patches(
            kpts2, img2_gray, 32, 12.0)
        kp2 = np.array([(x.pt[0], x.pt[1]) for x in kpts2]).reshape(-1, 2)
        patches2 = np.array(patches2)[:2048].astype(np.uint8)
        kp2 = kp2[:2048]


        # sift1 = cv2.xfeatures2d.SIFT_create(
        #     contrastThreshold=-10000, edgeThreshold=-10000)
        sift1 = cv2.SIFT_create(
            contrastThreshold=-10000, edgeThreshold=-10000)


        #if the input of main and wide image is not the same size
        #ensure the ratio of h/w same
        # img1_crop=img1[:,192:]
        # img1_crop=img1_crop[:,:-192]
        
        img1_crop=img1

        img1_low=cv2.resize(img1_crop,(new_w,new_h))
        img1_gray = cv2.cvtColor(img1_low,cv2.COLOR_BGR2RGB)
        img1_gray = cv2.cvtColor(l_clahe(img1_gray), cv2.COLOR_RGB2GRAY)
        
        keypoints1, scales1, _, scores1 = get_SIFT_keypoints(sift1,img1_gray)
        kpts1 = [
            cv2.KeyPoint(
                x=keypoints1[i][0],
                y=keypoints1[i][1],
                _size=scales1[i],
                _angle=0) for i, point in enumerate(keypoints1)
        ]
        patches1 = extract_patches(
            kpts1, img1_gray, 32, 12.0)
        kp1 = np.array([(x.pt[0], x.pt[1]) for x in kpts1]).reshape(-1, 2)
        patches1 = np.array(patches1)[:2048].astype(np.uint8)
        kp1 = kp1[:2048]


        

        #### extract hardnet descriptors ####
        hardnet_pretrained_path = 'H_align/pretrained/hardnet/HardNet++.pth'
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        model1 = HardNet()
        model1.load_state_dict(torch.load(hardnet_pretrained_path)['state_dict'])
        model1 = model1.cuda()
        model1.eval()

        des1 = np.zeros((patches1.shape[0],128))
        des2 = np.zeros((patches2.shape[0],128))
        transforms = get_transforms(False)
        data_1 = torch.stack([transforms(patch) for patch in patches1]).cuda()
        data_2 = torch.stack([transforms(patch) for patch in patches2]).cuda()
        with torch.no_grad():
            out_a = model1(data_1)
            des1 = out_a.cpu().detach().numpy()
            out_b = model1(data_2)
            des2 = out_b.cpu().detach().numpy()

        

        #### compute matches ####
        cfg, unparsed = get_config()
        result = compute_matches(des1,des2,cfg,kp1,kp2)
        matches = result[0]

        t1_0 = time.time()

        #### NM-Net ###
        kp1 = kp1[matches[0]]
        kp2 = kp2[matches[1]]
        kps = np.concatenate([kp1, kp2], axis=-1)
        warn = ['False', 'False']
        if warn == ['True', 'True']:
            pars1 = cameras[idx1].params
            pars2 = cameras[idx2].params
            k1 = np.array([[pars1[0], 0, pars1[2]], [0, pars1[1], pars1[3]], [0, 0, 1]])
            k2 = np.array([[pars2[0], 0, pars2[2]], [0, pars2[1], pars2[3]], [0, 0, 1]])

            kp1_n = (kp1 - np.array([[pars1[2], pars1[3]]])) / np.asarray([[pars1[0], pars1[1]]])
            kp2_n = (kp2 - np.array([[pars2[2], pars2[3]]])) / np.asarray([[pars2[0], pars2[1]]])

            kps_n = np.concatenate([kp1_n, kp2_n], axis=-1)

        elif warn == ['False', 'True']:
            f, cx, cy = norm_input(kp1)
            k1 = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            kp1_n = (kp1 - np.array([[cx, cy]])) / np.asarray([[f, f]])

            pars2 = cameras[idx2].params
            k2 = np.array([[pars2[0], 0, pars2[2]], [0, pars2[1], pars2[3]], [0, 0, 1]])
            kp2_n = (kp2 - np.array([[pars2[2], pars2[3]]])) / np.asarray([[pars2[0], pars2[1]]])

            kps_n = np.concatenate([kp1_n, kp2_n], axis=-1)
        elif warn == ['True', 'False']:
            f, cx, cy = norm_input(kp2)
            k2 = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            kp2_n = (kp2 - np.array([[cx, cy]])) / np.asarray([[f, f]])

            pars1 = cameras[idx1].params
            k1 = np.array([[pars1[0], 0, pars1[2]], [0, pars1[1], pars1[3]], [0, 0, 1]])
            kp1_n = (kp1 - np.array([[pars1[2], pars1[3]]])) / np.asarray([[pars1[0], pars1[1]]])

            kps_n = np.concatenate([kp1_n, kp2_n], axis=-1)
        elif warn == ['False', 'False']:
            f, cx, cy = norm_input(kp2)
            k2 = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            kp2_n = (kp2 - np.array([[cx, cy]])) / np.asarray([[f, f]])

            f, cx, cy = norm_input(kp1)
            k1 = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            kp1_n = (kp1 - np.array([[cx, cy]])) / np.asarray([[f, f]])

            kps_n = np.concatenate([kp1_n, kp2_n], axis=-1)

        config, _ = get_nmnet_config()
        model2 = NM_Net(config)
        model2.to(device)
        checkpoint_path = os.path.join(
            'H_align/pretrained/nmnet/model_best.pth')
        checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
        model2.load_state_dict(checkpoint['state_dict'])

        nmnet_input = torch.from_numpy(kps_n).float().to(device)
        output, A, in_ratio, e_hat, x_new = model2(nmnet_input.unsqueeze(0))

        match_pred = output[2][1]
        match_pred = match_pred > 0


        temp = list()
        temp2 = list()
        match_ind = list()
        for i in range(match_pred.shape[0]):
            for j in range(match_pred.shape[1]):
                if match_pred[i][j]:
                    temp.append(kps[j])
                    match_ind.append(j)
                else:
                    temp2.append(kps[j])

        if len(temp2) == 0:
            temp2.append([0, 0, 0, 0])

        t1_1 = time.time()
        t1 = t1_1 - t1_0


        ####绘制gt图片####
        temp = np.asarray(temp).astype(int)
        temp2 = np.asarray(temp2).astype(int)

        #### compute Homography Matrix ####
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1_crop = cv2.cvtColor(img1_crop, cv2.COLOR_BGR2RGB)
        img2_low = cv2.cvtColor(img2_low, cv2.COLOR_BGR2RGB)

        # homography based on fg/bg(blur)
        H,w = cv2.findHomography(kp1[match_ind], kp2[match_ind],cv2.RANSAC, 1)
        
        
        ## if the input of main and wide image is not the same size, we need to adjust the homography matrix
        # H[:2,2]=4*H[:2,2]
        # H[:2,2]=24*2+H[:2,2]
        # H[2,:2]=0.25*H[2,:2]


        #if the input of main and wide image is not the same size
        # img1 = cv2.warpPerspective(img1_crop, H, (img2.shape[1]+24*4, img2.shape[0]+24*4),  borderMode = cv2.BORDER_CONSTANT,borderValue=[255,255,255])
        
        img1=cv2.warpPerspective(img1_crop, H, (img2.shape[1], img2.shape[0]),  borderMode = cv2.BORDER_CONSTANT,borderValue=[255,255,255])
        if rot==1:
            img1=np.rot90(img1,3)


        
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(outfilepath, img1)
 

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

    root = '/'

    wide_root = os.path.join(root, 'wide_match_fg') ## ultra-wide camera aligned by homography

    main_root = os.path.join(root, 'original/fg')  # main camera

    save_warp_root = os.path.join(root, 'wide_warp_fg')  # fg, bg

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
        # wide=exposure.match_histograms(wide,main,multichannel=multi)#multichannel=True if RGB images

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


            # if wide and main image are not the same size
            # d_top = (h_wide_low - h_main_low) // 2
            # d_bottom = h_wide_low - h_main_low - d_top
            # d_left = (w_wide_low - w_main_low) // 2
            # d_right = w_wide_low - w_main_low - d_left
            # print(d_top)


            # wide_warp = wide_warp[..., d_top:-d_bottom, d_left:-d_right]
            # confidence = confidence[..., d_top:-d_bottom, d_left:-d_right]



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


if __name__ == '__main__':
    load_testing_data('sift_2048_hardnet')
    demo(args)