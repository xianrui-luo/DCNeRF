import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np



# sys.path.append('/home/zhaoweiyue/image-matching-benchmark-zhaochen')
from third_party.colmap.scripts.python.read_write_model import read_model, qvec2rotmat
from third_party.colmap.scripts.python.read_dense import read_array
from imageio import imread
import h5py

import json
import cv2
import pickle
from tqdm import trange
from extract_patches.core import extract_patches
import time
import tqdm
from model.HardNet import HardNet
from model.Utils import cv2_scale, np_reshape
import torchvision.transforms as transforms

from config import get_config, print_usage
from methods import feature_matching as matching

from core.config import get_config as get_nmnet_config
from model.model_new1 import NM_Net_v2 as NM_Net
from tqdm import trange
# from core.utils import tocuda
from matplotlib.patches import ConnectionPatch

from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt\


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    original_dir='/data2/luoxianrui/data/registration_deblur/focal_stack_final/original/bg_re_match_flow/'#3648*2736
    # original_dir='G:/data/registration_deblur/focal_stack/original/fg/'#3648*2736
    wide_dir='/data2/luoxianrui/data/registration_deblur/focal_stack_final/wide_raw/'#3840*2592

    new_wide_dir='/data2/luoxianrui/data/registration_deblur/focal_stack_final/wide_match_bg/'
    if not os.path.exists(new_wide_dir):
        os.makedirs(new_wide_dir)
    filelist=os.listdir(original_dir)
    # filelist.sort(key = lambda x:int(x[:-4]))
    for name in filelist:
        rot=0
        img2_path = os.path.join(original_dir,name)
        img1_path = os.path.join(wide_dir,name)
        # mask_path=os.path.join(mask_dir,name)

        outFilename = name
        
        outfilepath=os.path.join(new_wide_dir,outFilename)
        # print("Saving aligned image : ", outfilepath); 
        print("Saving aligned image : ", outfilepath); 
        # if os.path.exists(outfilepath):
        #     continue


        print('extract keypoints and descriptors')
        
        #### extract keypoints and descriptors ####

        sift2 = cv2.xfeatures2d.SIFT_create(
            contrastThreshold=-10000, edgeThreshold=-10000)
        # sift2 = cv2.SIFT_create(
        #     contrastThreshold=-10000, edgeThreshold=-10000)
        img2 = cv2.imread(img2_path)
        img1 = cv2.imread(img1_path)
        # mask=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        # ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        # mask=np.expand_dims(mask,axis=2)
        # mask=np.stack((mask,mask,mask),axis=2)
        # mask=mask/255.0


        h,w,_=img2.shape
        if h>w:
            rot=1
            img1=np.rot90(img1,1)
            img2=np.rot90(img2,1)
        w=3648
        h=2736
        new_h=h//4
        new_w=w//4
        # new_h=h
        # new_w=w
        img2_low=cv2.resize(img2,(new_w,new_h))
        # mask_low=cv2.resize(mask,(new_w,new_h)).astype(np.uint8)

        
        # img2_low=img2_low*mask_low

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


        sift1 = cv2.xfeatures2d.SIFT_create(
            contrastThreshold=-10000, edgeThreshold=-10000)
        # sift1 = cv2.SIFT_create(
        #     contrastThreshold=-10000, edgeThreshold=-10000)
        # img1 = cv2.imread(img1_path)

        #for wide
        #ensure the ratio of h/w same
        img1_crop=img1[:,192:]
        img1_crop=img1_crop[:,:-192]
        # img1_crop=cv2.resize(img1_crop,(w,h))

        img1_low=cv2.resize(img1_crop,(new_w,new_h))
        # img1_low=img1_low*mask_low
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
        hardnet_pretrained_path = './pretrained/hardnet/HardNet++.pth'
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        model1 = HardNet()
        # model1.load_state_dict(torch.load(hardnet_pretrained_path, map_location=device)['state_dict'])
        model1.load_state_dict(torch.load(hardnet_pretrained_path)['state_dict'])
        # model1 = model1.to(device)
        model1 = model1.cuda()
        model1.eval()

        des1 = np.zeros((patches1.shape[0],128))
        des2 = np.zeros((patches2.shape[0],128))
        transforms = get_transforms(False)
        # data_1 = torch.stack([transforms(patch) for patch in patches1]).to(device)
        # data_2 = torch.stack([transforms(patch) for patch in patches2]).to(device)
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
            './pretrained/nmnet/model_best.pth')
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

        #single homography
        # H,w = cv2.findHomography(kp1[match_ind], kp2[match_ind],cv2.RANSAC, 1)

        # homography based on fg/bg(blur)
        H,w = cv2.findHomography(kp1[match_ind], kp2[match_ind],cv2.RANSAC, 1)
        # H_bg,w = cv2.findHomography(kp1[match_ind], kp2[match_ind],cv2.RANSAC, 1)
        H[:2,2]=4*H[:2,2]
        H[:2,2]=24*2+H[:2,2]
        H[2,:2]=0.25*H[2,:2]


        #wide
        img1 = cv2.warpPerspective(img1_crop, H, (img2.shape[1]+24*4, img2.shape[0]+24*4),  borderMode = cv2.BORDER_CONSTANT,borderValue=[255,255,255])
        # img1_bg = cv2.warpPerspective(img1_crop, H_bg, (img2.shape[1]+24*4, img2.shape[0]+24*4),  borderMode = cv2.BORDER_CONSTANT,borderValue=[255,255,255])
        # else:
            # img1 = cv2.warpPerspective(img1_crop, H, (img2.shape[0]+24*4, img2.shape[1]+24*4),  borderMode = cv2.BORDER_CONSTANT,borderValue=[255,255,255])
        if rot==1:
            img1=np.rot90(img1,3)


        
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(outfilepath, img1)
 


if __name__ == '__main__':
    load_testing_data('sift_2048_hardnet')