## 输入是经过配准的广角和主摄的融合图，用的pose是主摄fg or bg的
## 加入confidence

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import imageio
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import nerf_model
from nerf_utils import *
from utils_eval import *

from load_llff import load_llff_data

from log import LOG_Limited
from bokeh_utils import render_bokeh_no_01_disp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def compute_depth_loss(pred_depth, gt_depth): 
    # pred_depth_e = NDC2Euclidean(pred_depth_ndc)
    t_pred = torch.median(pred_depth)
    s_pred = torch.mean(torch.abs(pred_depth - t_pred))

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))

    pred_depth_n = (pred_depth - t_pred)/s_pred
    gt_depth_n = (gt_depth - t_gt)/s_gt

    # return torch.mean(torch.abs(pred_depth_n - gt_depth_n))
    return torch.mean(torch.pow(pred_depth_n - gt_depth_n, 2))



def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    
    parser.add_argument("--lrate_bokeh",          type=float, default=2e-4, 
                        help='learning rate')
    
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--N_iters",      type=int,   default=200000, 
                        help='Iters to Train Nerf Model')
    parser.add_argument("--bokeh_iters",      type=int,   default=0, 
                        help='Extra Iters to Train Bokeh Parameters')
    
    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=2, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=200000, 
                        help='frequency of render_poses video saving')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    # if args.dataset_type == 'llff':
    images, main_images,poses, bds, render_poses, i_test,confidence_gt,depth_gt= load_llff_data(args.datadir, args.factor,
                                                                recenter=True, bd_factor=.75,
                                                                spherify=args.spherify,logname=args.expname)
    # images,main_images, poses, bds, render_poses, i_test,depth_gt = load_llff_data(args.datadir, args.factor,
    #                                                           recenter=True, bd_factor=.75,
    #                                                           spherify=args.spherify,logname=args.expname)
    
    confidence_gt=np.expand_dims(confidence_gt,3)
    confidence_gt=np.tile(confidence_gt,(1,1,1,3))
    depth_gt=np.expand_dims(depth_gt,3)
    depth_gt=np.tile(depth_gt,(1,1,1,3))
    
    
    hwf = poses[0,:3,-1]
    poses = poses[:,:3,:4]
    print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
    if not isinstance(i_test, list):
        i_test = [i_test]

    if args.llffhold > 0:
        print('Auto LLFF holdout,', args.llffhold)
        i_test = np.arange(images.shape[0])[::args.llffhold]

    i_val = i_test
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                    (i not in i_test and i not in i_val)])

    print('DEFINING BOUNDS')
    if args.no_ndc:
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.
        
    else:
        near = 0.
        far = 1.
    print('NEAR FAR', near, far)



    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    # log = LOG_Limited(level='DEBUG',file=args.expname, max=5000)
    log = LOG_Limited(name='TRAIN',
                     dir=os.path.join(args.basedir, args.expname),
                     file='TrainLog.txt',
                     level='INFO')
    logger_para = LOG_Limited(name='PARAM',
                        dir=os.path.join(args.basedir, args.expname),
                        file='ParaLog.txt',
                        level='INFO')
    log.output_debug('This is a debug message')


    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model +bokeh_param
    # render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    
    N_image=images.shape[0]
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer,bokeh_param = create_nerf_test(args,N_image)
    # render_kwargs_train_main, render_kwargs_test_main, start_main, grad_vars_main, optimizer_main= create_nerf(args)
    global_step = start
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    render_kwargs_test_2={'network_query_fn':render_kwargs_test['network_query_fn'],
                                   'perturb' :render_kwargs_test['perturb'],\
                                   'N_importance':render_kwargs_test['N_importance'],\
                                    'network_fine_main':render_kwargs_test['network_fine_main'],
                                    'N_samples':render_kwargs_test['N_samples'],\
                                    'network_fn_main':render_kwargs_test['network_fn_main'],
                                    'use_viewdirs':render_kwargs_test['use_viewdirs'],\
                                    'white_bkgd':render_kwargs_test['white_bkgd'],
                                    'raw_noise_std':render_kwargs_test['raw_noise_std']    }

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            # para_K_bokeh,para_disp_focus=bokeh_param()
            
            # dispdir = os.path.join(basedir, expname, 'testset_disp')
            # depthfiles = [os.path.join(dispdir, f) for f in sorted(os.listdir(dispdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
            # for f in sorted(os.listdir(dispdir)):
            #     depth_file=   os.path.join(dispdir, f)        
            #     depth_imgs=imageio.imread(depth_file, ignoregamma=True)/255
            #     index=f[:3]
            #     number=int(index)
            #     disp_focus=para_disp_focus[number]
            #     # depth_imgs = np.moveaxis(depth_imgs, -1, 0).astype(np.float32)
            #     # depth_imgs=np.expand_dims(depth_imgs,3)
            #     # depth_imgs=np.tile(depth_imgs,(1,1,1,3))
            #     depth_imgs = torch.Tensor(depth_imgs).to(device)
            #     target_disp=depth_imgs
            #     target_focus=torch.abs(target_disp-disp_focus)
            #     defocus_max=torch.max(target_focus)
            #     bokeh_range=1/(para_K_bokeh[0]*10)
            #     zeros=torch.zeros_like(target_focus)
            #     ones=torch.ones_like(target_focus)
            #     # binary_mask=torch.where(target_focus<(defocus_max*0.2),zeros,target_focus)
            #     binary_mask=torch.where(target_focus<(defocus_max*bokeh_range),zeros,target_focus)
            #     binary_mask=binary_mask.cpu().numpy()
            #     mask = nerf_model.to8b(binary_mask)
            #     newdir_name=os.path.join(basedir, expname, 'testset_defocus')
            #     os.makedirs(newdir_name,exist_ok=True)
            #     filename = os.path.join(newdir_name ,f)
            #     imageio.imwrite(filename, mask)
            # depth= depth= [imageio.imread(f, ignoregamma=True)/255. for f in depthfiles]
            # depth_imgs = np.stack(depth, -1)  
            # depth_imgs = np.moveaxis(depth_imgs, -1, 0).astype(np.float32)
            # depth_imgs=np.expand_dims(depth_imgs,3)
            # depth_imgs=np.tile(depth_imgs,(1,1,1,3))
            # depth_imgs = torch.Tensor(depth_imgs).to(device)
            # target_disp=depth_imgs[5]
            # target_focus=torch.abs(target_disp-para_disp_focus)
            # defocus_max=torch.max(target_focus)
            # bokeh_range=1/(para_K_bokeh[0]*10)
            # print(bokeh_range)
            # zeros=torch.zeros_like(target_focus)
            # ones=torch.ones_like(target_focus)
            # # binary_mask=torch.where(target_focus<(defocus_max*0.2),zeros,target_focus)
            # binary_mask=torch.where(target_focus<(defocus_max*bokeh_range),zeros,target_focus)
            # binary_mask=binary_mask.cpu().numpy()
            # mask = nerf_model.to8b(binary_mask)
            # # newdir_name=os.path.join(basedir, expname, 'testset_defocus')
            # # # filename = os.path.join(basedir, expname, './mask_bear_bg.jpg')
            # # imageio.imwrite(filename, mask)
            # if args.render_test:
                # render_test switches to test poses
            # images = images[i_test]
            # else:
                # Default is smoother render_poses path
                # images = None
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            if args.render_test:
                rgbs, _ = render_path_test(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
                print('Done rendering', testsavedir)


# use
            # testsavedir = os.path.join(basedir, expname, 'renderonly1_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            # os.makedirs(testsavedir, exist_ok=True)
            # print('test poses shape', render_poses.shape)
            # i='360'
            # # rgbs, _ = render_path_mask(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            # # rgbs, _ = render_path_mask(torch.Tensor(poses).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            # rgbs, disps = render_path_test(render_poses, hwf, K, args.chunk, render_kwargs_test,savedir=testsavedir)
            # # rgbs, disps = render_path_main(render_poses, hwf, K, args.chunk, render_kwargs_test_2,savedir=testsavedir)
            # print('Done, saving', rgbs.shape, disps.shape)
            # # moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            # moviebase = os.path.join(basedir, expname, '{}_spiral_main_{}_'.format(expname, i))
            # imageio.mimwrite(moviebase + 'rgb.mp4', nerf_model.to8b(rgbs), fps=30, quality=8)
            # imageio.mimwrite(moviebase + 'disp.mp4', nerf_model.to8b(disps / np.max(disps)), fps=30, quality=8)
            
            
            
            # print('Done rendering', testsavedir)
            # imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), nerf_model.to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    
    # ones=np.ones_like(confidence_gt[:,0,:])
    # zeros=np.zeros_like(confidence_gt[:,0,:])
    # rays_confidence_mask=np.where(confidence_gt[:,0,:]>0.5,ones,zeros)
    # rays_confidence_mask=np.expand_dims(rays_confidence_mask,axis=1)
    # confidence_mask=np.tile(rays_confidence_mask,(1,3,1,1,1))
    # input_images=images*confidence_gt+main_images*(1-confidence_gt)
    if use_batching:
    #     # For random ray batching
        print('get rays')
        rays = np.stack([nerf_model.get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        
        print('done, concats')
        rays_rgb_patch = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb_patch = np.concatenate([rays_rgb_patch, confidence_gt[:,None]], 1) # [N, ro+rd+rgb+c, H, W, 3]
        # rays_confidence_patch = np.concatenate([rays, confidence_gt[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
    
        # byte_mask=rays_confidence_mask.byte().bool()
        
        # rays_confidence_patch = confidence_gt[:,None] # [N, ro+rd+rgb, H, W, 3]
        # rays_confidence_patch = rays_confidence_mask[:,None] # [N, ro+rd+rgb, H, W, 3]
        
        rays_rgb_main_patch= np.concatenate([rays, main_images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]

        rays_depth_patch= np.concatenate([rays, depth_gt[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        ##depth
        # rays_rgbd_patch= np.concatenate([rays_rgb_patch, depth_gt[:,None]], 1) # [N, ro+rd+rgb+d, H, W, 3]
        # rays_rgbf_patch= np.concatenate([rays_rgb_patch, defocus_gt[:,None]], 1) # [N, ro+rd+rgb+d+df, H, W, 3]
        # rays_rgbfc_patch= np.concatenate([rays_rgbf_patch, rays_confidence_mask[:,None]], 1) # [N, ro+rd+rgb+d+df, H, W, 3]
        # rays_rgbfm_patch= np.concatenate([rays_rgbf_patch, main_images[:,None]], 1) # [N, ro+rd+rgb+d+df+main, H, W, 3]
        
        # rays_rgbfc_patch = np.transpose(rays_rgbfc_patch, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb_patch = np.transpose(rays_rgb_patch, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb_main_patch = np.transpose(rays_rgb_main_patch, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        # rays_confidence_patch = np.transpose(rays_confidence_patch, [0,2,3,1,4]) # [N, H, W, rgb, 3]
        rays_depth_patch = np.transpose(rays_depth_patch, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
    #     # rays_rgbd_patch = np.transpose(rays_rgbd_patch, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
    #     # rays_defocus_patch = np.transpose(rays_defocus_patch, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        
    #     # rays_rgbd = np.stack([rays_rgbd_patch[i] for i in i_train], 0) # train images only
        rays_rgb = np.stack([rays_rgb_patch[i] for i in i_train], 0) # train images only
    #     rays_rgbfc = np.stack([rays_rgbfc_patch[i] for i in i_train], 0) # train images only
        rays_rgb_main = np.stack([rays_rgb_main_patch[i] for i in i_train], 0) # train images only
    #     # rays_confidence = np.stack([rays_confidence_patch[i] for i in i_train], 0) 
    #     # rays_defocus = np.stack([rays_defocus_patch[i] for i in i_train], 0) 
        rays_depth = np.stack([rays_depth_patch[i] for i in i_train], 0) 
        

        rays_rgb = np.reshape(rays_rgb, [-1,4,3]) # [(N-1)*H*W, ro+rd+rgb+c, 3]

        rays_rgb_main = np.reshape(rays_rgb_main, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
    #     # rays_confidence = np.reshape(rays_confidence, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
    #     # rays_defocus = np.reshape(rays_defocus, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_depth = np.reshape(rays_depth, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        
        
    #     rays_rgbfc = rays_rgbfc.astype(np.float32)
        rays_depth = rays_depth.astype(np.float32)
        rays_rgb = rays_rgb.astype(np.float32)
        rays_rgb_main = rays_rgb_main.astype(np.float32)
    #     # rays_confidence = rays_confidence.astype(np.float32)
    #     # rays_defocus = rays_defocus.astype(np.float32)
    #     print('shuffle rays')
    #     print('done')
        i_batch = 0
    ones=np.ones_like(rays_rgb[:,3,:])
    zeros=np.zeros_like(rays_rgb[:,3,:])
    rays_confidence_mask=np.where(rays_rgb[:,3,:]>0.5,ones,zeros)
    rays_confidence_mask=np.expand_dims(rays_confidence_mask,axis=1)
    rays_confidence_mask=np.tile(rays_confidence_mask,(1,3,1))
    byte_mask=np.array(rays_confidence_mask,dtype=bool)
    
    N,r,c=rays_rgb.shape
    
    rays_rgb_1=rays_rgb[:,0:3,:]  
    # rays_rgb_1=np.expand_dims(rays_rgb_1,axis=1)  
    rays_rgb_1=rays_rgb_1[byte_mask]
    rays_rgb_1=rays_rgb_1.reshape(-1,r-1,c)
    
    state=np.random.get_state()
    np.random.shuffle(rays_rgb_1)
    np.random.set_state(state)
    np.random.shuffle(rays_rgb)
    np.random.set_state(state)
    np.random.shuffle(rays_rgb_main)
    # np.random.set_state(state)
    # np.random.shuffle(rays_depth)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)
        rays_rgb_1 = torch.Tensor(rays_rgb_1).to(device)
        rays_rgb_main = torch.Tensor(rays_rgb_main).to(device)
    #     # rays_rgb_main_fine = torch.Tensor(rays_rgb_main_fine).to(device)
    #     # rays_confidence = torch.Tensor(rays_confidence).to(device)
        # rays_depth = torch.Tensor(rays_depth).to(device)
    images = torch.Tensor(images).to(device)
    main_images = torch.Tensor(main_images).to(device)
    confidence_gt = torch.Tensor(confidence_gt).to(device)
        ## refine edge
    N_patch_sample = 36
    N_pixel=96
    patch_rays = []
    patch_main_rays = []
    patch_depth_rays = []
    image_list = []
    print('Preparing Patching Rays')
    # rays_rgb_patch = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
    # rays_rgb_patch = np.transpose(rays_rgb_patch, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3] 前面已有
    # rays_rgb_patch = np.stack([rays_rgb_patch[i] for i in i_train], 0) # train images only [N-?, H, W, ro+rd+rgb, 3]
    # H_patch = int((H-N_pixel)/N_patch_sample)
    # W_patch = int((W-N_pixel)/N_patch_sample)
    # patch_rays = []
    # patch_main_rays = []
    # # patch_confidence_rays = []
    # patch_depth_rays = []
    # image_list = []
    # for i in i_train:
    #     for h in range(H_patch+2):
    #         for w in range(W_patch+2):
    #             h0, w0 = h * N_patch_sample, w * N_patch_sample
    #             if h0 + N_pixel >= H:
    #                 h0 = H - N_pixel
    #             if w0 + N_pixel >= W:
    #                 w0 = W - N_pixel
    #             # if np.any(confidence_byte[i, h0:h0+N_pixel, w0:w0+N_pixel,:]==0):  
    #             #     continue
    #             else:
    #                 # patch_rays.append(rays_rgb_patch[i, h0:h0+N_pixel, w0:w0+N_pixel, ...])
    #                 patch_main_rays.append(rays_rgb_main_patch[i, h0:h0+N_pixel, w0:w0+N_pixel, ...])
    #                 # patch_confidence_rays.append(rays_confidence_patch[i, h0:h0+N_pixel, w0:w0+N_pixel, ...])
    #                 # patch_depth_rays.append(rays_depth_patch[i, h0:h0+N_pixel, w0:w0+N_pixel, ...])
    #                 image_list.append(i)

                
    # print(len(patch_main_rays))
    # patch_main_rays = np.stack(patch_main_rays)
    # # patch_depth_rays = np.stack(patch_depth_rays)
    # patch_main_rays = patch_main_rays.astype(np.float32)
    # # patch_depth_rays = patch_depth_rays.astype(np.float32)
    # train_index = np.stack(list(range(len(patch_main_rays))))
    # print('shuffle index') 
    # np.random.shuffle(train_index)
    # print('done')          
    i_patch = 0

    # patch_main_rays=torch.Tensor(patch_main_rays).to(device)
    # patch_depth_rays=torch.Tensor(patch_depth_rays).to(device)         

    N_iters = args.N_iters+1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    # log.output_debug('HOLDOUT view is:{%s}'.format(i_test))
    log.output_debug('HOLDOUT view is:')
    log.output_debug(i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1

    bce_loss_func = nn.BCELoss()
    # for i in trange(start, N_iters):
    
    #test
    # N_iters=1
    pre=False
    pre_depth=False
    fg=True
    frozen=False
    render_kwargs_train_1={'network_query_fn':render_kwargs_train['network_query_fn'],
                                   'perturb' :render_kwargs_train['perturb'],\
                                   'N_importance':render_kwargs_train['N_importance'],\
                                    'network_fine':render_kwargs_train['network_fine'],
                                    'N_samples':render_kwargs_train['N_samples'],\
                                    'network_fn':render_kwargs_train['network_fn'],
                                    'use_viewdirs':render_kwargs_train['use_viewdirs'],\
                                    'white_bkgd':render_kwargs_train['white_bkgd'],
                                    'raw_noise_std':render_kwargs_train['raw_noise_std']    }
    render_kwargs_train_2={'network_query_fn':render_kwargs_train['network_query_fn_main'],
                                   'perturb' :render_kwargs_train['perturb'],\
                                   'N_importance':render_kwargs_train['N_importance'],\
                                    'network_fine_main':render_kwargs_train['network_fine_main'],
                                    'N_samples':render_kwargs_train['N_samples'],\
                                    'network_fn_main':render_kwargs_train['network_fn_main'],
                                    'use_viewdirs':render_kwargs_train['use_viewdirs'],\
                                    'white_bkgd':render_kwargs_train['white_bkgd'],
                                    'raw_noise_std':render_kwargs_train['raw_noise_std']    }
    render_kwargs_test_2={'network_query_fn':render_kwargs_test['network_query_fn'],
                                   'perturb' :render_kwargs_test['perturb'],\
                                   'N_importance':render_kwargs_test['N_importance'],\
                                    'network_fine_main':render_kwargs_test['network_fine_main'],
                                    'N_samples':render_kwargs_test['N_samples'],\
                                    'network_fn_main':render_kwargs_test['network_fn_main'],
                                    'use_viewdirs':render_kwargs_test['use_viewdirs'],\
                                    'white_bkgd':render_kwargs_test['white_bkgd'],
                                    'raw_noise_std':render_kwargs_test['raw_noise_std']    }
    render_kwargs_test_1={'network_query_fn':render_kwargs_test['network_query_fn'],
                                   'perturb' :render_kwargs_test['perturb'],\
                                   'N_importance':render_kwargs_test['N_importance'],\
                                    'network_fine':render_kwargs_test['network_fine'],
                                    'N_samples':render_kwargs_test['N_samples'],\
                                    'network_fn':render_kwargs_test['network_fn'],
                                    'use_viewdirs':render_kwargs_test['use_viewdirs'],\
                                    'white_bkgd':render_kwargs_test['white_bkgd'],
                                    'raw_noise_std':render_kwargs_test['raw_noise_std']    }
    for i in trange(start, N_iters + args.bokeh_iters):
        time0 = time.time()
       # Random from one image
        para_K_bokeh,para_disp_focus=bokeh_param()
        
        #test
        # i=N_iters+200000

        # Sample random ray batch
        if i<N_iters:
            # Random over all images
            # batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = rays_rgb_1[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            # batch_main = rays_rgb_main[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            # batch_depth = rays_depth[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            
            batch = torch.transpose(batch, 0, 1)
            # batch_main = torch.transpose(batch_main, 0, 1)
            # batch_confidence = torch.transpose(batch_confidence, 0, 1)
            # batch_depth= torch.transpose(batch_depth, 0, 1)
            # batch_confidence = batch_confidence[2]
            # batch_main = batch_main[2]
            batch_rays, target_s = batch[:2], batch[2]                                                                                                             
            # batch_depth_rays =  batch_depth[2]                                                                                                             

            i_batch += N_rand
            if i_batch >= rays_rgb_1.shape[0]:
                print("Shuffle data after an epoch!")
                # rand_idx = torch.randperm(rays_rgb.shape[0])
                rand_idx = np.arange(0,rays_rgb_1.shape[0],1,dtype=np.int64)
                np.random.shuffle(rand_idx)
                rays_rgb_1 = rays_rgb_1[torch.as_tensor(rand_idx)]
                i_batch = 0
        elif i<N_iters+200000:
            batch_main = rays_rgb_main[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]        
            # batch_depth= rays_depth[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]        
            batch_main = torch.transpose(batch_main, 0, 1)
            # batch_depth = torch.transpose(batch_depth, 0, 1)
            batch_rays, target_main= batch_main[:2], batch_main[2]                                                                                                             
            # batch_depth_rays =  batch_depth[2]                                                                                                             

            i_batch += N_rand
            if i_batch >= rays_rgb_main.shape[0]:
                print("Shuffle data after an epoch!")
                # rand_idx = torch.randperm(rays_rgb.shape[0])
                rand_idx = np.arange(0,rays_rgb_main.shape[0],1,dtype=np.int64)
                np.random.shuffle(rand_idx)
                rays_rgb_main = rays_rgb_main[torch.as_tensor(rand_idx)]
                # rays_depth = rays_depth[torch.as_tensor(rand_idx)]
                i_batch = 0
        elif i<N_iters+400000:
            # test
            # i_patch=0
            if patch_main_rays==[]:
                dispdir = os.path.join(basedir, expname, 'testset_disp')
                depthfiles = [os.path.join(dispdir, f) for f in sorted(os.listdir(dispdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
                depth= depth= [imageio.imread(f, ignoregamma=True)/255. for f in depthfiles]
                depth_imgs = np.stack(depth, -1)  
                depth_imgs = np.moveaxis(depth_imgs, -1, 0).astype(np.float32)
                depth_imgs=np.expand_dims(depth_imgs,3)
                depth_imgs=np.tile(depth_imgs,(1,1,1,3))
                rays_depth_patch= np.concatenate([rays, depth_imgs[:,None]], 1)
                depth_imgs = torch.Tensor(depth_imgs).to(device)
                rays_depth_patch = np.transpose(rays_depth_patch, [0,2,3,1,4])
                H_patch = int((H-N_pixel)/N_patch_sample)
                W_patch = int((W-N_pixel)/N_patch_sample)
                
                for itrain in i_train:
                    for h in range(H_patch+2):
                        for w in range(W_patch+2):
                            h0, w0 = h * N_patch_sample, w * N_patch_sample
                            if h0 + N_pixel >= H:
                                h0 = H - N_pixel
                            if w0 + N_pixel >= W:
                                w0 = W - N_pixel
                            # if np.any(confidence_byte[i, h0:h0+N_pixel, w0:w0+N_pixel,:]==0):  
                            #     continue
                            else:
                                # patch_rays.append(rays_rgb_patch[i, h0:h0+N_pixel, w0:w0+N_pixel, ...])
                                patch_main_rays.append(rays_rgb_main_patch[itrain, h0:h0+N_pixel, w0:w0+N_pixel, ...])
                                # patch_confidence_rays.append(rays_confidence_patch[i, h0:h0+N_pixel, w0:w0+N_pixel, ...])
                                patch_depth_rays.append(rays_depth_patch[itrain, h0:h0+N_pixel, w0:w0+N_pixel, ...])
                                image_list.append(itrain)

                            
                print(len(patch_main_rays))
                patch_main_rays = np.stack(patch_main_rays)
                patch_depth_rays = np.stack(patch_depth_rays)
                patch_main_rays = patch_main_rays.astype(np.float32)
                patch_depth_rays = patch_depth_rays.astype(np.float32)
                train_index = np.stack(list(range(len(patch_main_rays))))
                print('shuffle index') 
                np.random.shuffle(train_index)
                print('done')          
                patch_main_rays=torch.Tensor(patch_main_rays).to(device)
                patch_depth_rays=torch.Tensor(patch_depth_rays).to(device)
                
            index = train_index[i_patch]
            # patch = patch_rays[index]
            patch_main = patch_main_rays[index]
            patch_depth = patch_depth_rays[index]
            
            img_i = image_list[index]
            patch_main = torch.reshape(patch_main, [-1, patch_main.shape[-2], patch_main.shape[-1]])
            patch_main = torch.transpose(patch_main, 0, 1)
            batch_rays, target_main = patch_main[:2], patch_main[2]
            # batch_rays.requires_grad=False
            # target_main = patch_main[2]
            patch_depth= torch.reshape(patch_depth, [-1, patch_depth.shape[-2], patch_depth.shape[-1]])
            patch_depth= torch.transpose(patch_depth, 0, 1)
            target_disp = patch_depth[2]
            
            i_patch += 1
            if i_patch >= patch_main_rays.shape[0]:
                log.output_info("Shuffle data after an epoch!")
                i_patch = 0
                np.random.shuffle(train_index)
                print('Done')

        else:
            # if pre_depth==False:
            #     dispdir = os.path.join(basedir, expname, 'testset_disp')
            #     depthfiles = [os.path.join(dispdir, f) for f in sorted(os.listdir(dispdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
            #     depth= depth= [imageio.imread(f, ignoregamma=True)/255. for f in depthfiles]
            #     depth_imgs = np.stack(depth, -1)  
            #     depth_imgs = np.moveaxis(depth_imgs, -1, 0).astype(np.float32)
            #     depth_imgs=np.expand_dims(depth_imgs,3)
            #     depth_imgs=np.tile(depth_imgs,(1,1,1,3))
            #     depth_imgs = torch.Tensor(depth_imgs).to(device)
                # pre_depth=True

            if i==N_iters+400000 or pre==False:
                n,h,w,_,c=rays_rgb_patch.shape
                with torch.no_grad():

                    para_disp_focus=para_disp_focus.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    para_disp_focus_all=para_disp_focus.repeat(1,h,w,c)
                    # para_disp_focus_all = torch.stack([para_disp_focus_all[i] for i in i_train], 0) # train images only
                    para_disp_focus_all=torch.clamp(para_disp_focus_all,0,1)
                    # para_disp_focus_all = torch.reshape(para_disp_focus_all, [-1,1,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
                    # defocus_map=rays_depth[:,2,:].unsqueeze(1)
                    # rays_defocus=torch.abs(defocus_map-para_disp_focus_all)
                    # min_defocus=torch.min(rays_defocus)
                    # max_defocus=torch.max(rays_defocus)
                    # rays_defocus=(rays_defocus-min_defocus)/(max_defocus-min_defocus)
                pre=True
            # Random over all images
            img_i = np.random.choice(i_train)
            target_wide = images[img_i]
            target_main = main_images[img_i]
            target_conf = confidence_gt[img_i]
            target_disp= depth_imgs[img_i]
            
            para_disp_focus=para_disp_focus_all[img_i]
            target_disp=torch.Tensor(target_disp).to(device)
            target_defocus=torch.abs(target_disp-para_disp_focus)
            # target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]
            
            
            if N_rand is not None:
                rays_o, rays_d = nerf_model.get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target_wide[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_confidence = target_conf[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                target_main = target_main[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                # target_focus = para_disp_focus[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                target_focus = target_defocus[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

          

        #####  Core optimization loop  #####
        # rgb, disp, confidence,acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
        #                                         verbose=i < 10, retraw=True,
        #                                         **render_kwargs_train)
        if i <N_iters:
            frozen=False
            rgb_ori, disp, acc, depth,extras = render(H, W, K,chunk=args.chunk, rays=batch_rays,
                                            verbose=i < 10, retraw=True,**render_kwargs_train_1)
            
        elif i<N_iters+200000:
            # frozen=True
            # perturb=render_kwargs_train['perturb'] 
            # raw_noise_std=render_kwargs_train['raw_noise_std']
            # render_kwargs_train['perturb'] = False
            # render_kwargs_train['raw_noise_std'] = 0.
            # for name,param in render_kwargs_train_2['network_fn_main'].named_parameters():
            #     param.requires_grad = False
            # for name,param in render_kwargs_train_2['network_fine_main'].named_parameters():
            #     param.requires_grad = False
            # render_kwargs_train_2={'network_query_fn':render_kwargs_train['network_query_fn'],
            #                        'perturb' :render_kwargs_train['perturb'],\
            #                        'N_importance':render_kwargs_train['N_importance'],\
            #                         'network_fine_main':render_kwargs_train['network_fine_main'],
            #                         'N_samples':render_kwargs_train['N_samples'],\
            #                         'network_fn_main':render_kwargs_train['network_fn_main'],
            #                         'use_viewdirs':render_kwargs_train['use_viewdirs'],\
            #                         'white_bkgd':render_kwargs_train['white_bkgd'],
            #                         'raw_noise_std':render_kwargs_train['raw_noise_std']    }
            
            rgb_ori, disp, acc, depth,extras = render_main(H, W, K,chunk=args.chunk, rays=batch_rays,
                                            verbose=i < 10, retraw=True,**render_kwargs_train_2)
        elif i<N_iters+400000:
            frozen=True
            for name,param in render_kwargs_train['network_fn_main'].named_parameters():
                param.requires_grad = False
            for name,param in render_kwargs_train['network_fine_main'].named_parameters():
                param.requires_grad = False
            # render_kwargs_train_2['perturb'] = 0.
            # render_kwargs_train_2['raw_noise_std'] = 0.

            # _, disp, _, _,_ = render_main(H, W, K,chunk=args.chunk, rays=batch_rays,
            #                             verbose=i < 10, retraw=True,**render_kwargs_train_2)
            
            for name,param in render_kwargs_train_1['network_fn'].named_parameters():
                param.requires_grad = False
            for name,param in render_kwargs_train_1['network_fine'].named_parameters():
                param.requires_grad = False
            render_kwargs_train_1['perturb'] = 0.
            render_kwargs_train_1['raw_noise_std'] = 0.
            rgb_ori, _, acc, depth,extras = render(H, W, K,chunk=args.chunk, rays=batch_rays,
                                        verbose=i < 10, retraw=True,**render_kwargs_train_1)
        else:
            # if frozen==True:               
            #     render_kwargs_train['perturb']=perturb
            #     render_kwargs_train['raw_noise_std']=raw_noise_std
            for name,param in render_kwargs_train['network_fn'].named_parameters():
                param.requires_grad = False
            for name,param in render_kwargs_train['network_fine'].named_parameters():
                param.requires_grad = False
            for name,param in render_kwargs_train['network_fn_main'].named_parameters():
                param.requires_grad = True
            for name,param in render_kwargs_train['network_fine_main'].named_parameters():
                param.requires_grad = True
            rgb_ori, disp, acc, depth,rgb_main,extras = render_test(H, W, K,chunk=args.chunk, rays=batch_rays,
                                            verbose=i < 10, retraw=True,**render_kwargs_train)
            
        # rgb_ori, disp, acc, depth,extras = render_test(H, W, K,chunk=args.chunk, rays=batch_rays,
        #                                     verbose=i < 10, retraw=True,**render_kwargs_train)
        if i <N_iters+200000:
            rgb=rgb_ori
        elif i<N_iters+400000:
            rgb=torch.reshape(rgb_ori,[N_pixel,-1,3])
            disp_input=target_disp[:,0]
            disp_input=torch.reshape(disp_input,[N_pixel,-1])
            # disp_input=torch.reshape(disp,[N_pixel,-1])
            # disp_max=disp_input.max()
            # disp_min=disp_input.min()
            # disp_input=(disp_input-disp_min)/(disp_max-disp_min) 
            
            # disp_input=target_depth[:,0]
            # disp_input=torch.reshape(disp_input,[N_pixel,-1])
            rgb = render_bokeh_no_01_disp(rgb, 
                                          disp_input, # disp / max_disp, 
                                        #   disp / math.ceil(disp_max), # disp / max_disp, 
                                        #   K_bokeh=para_K_bokeh[img_i] * 10,
                                          K_bokeh=para_K_bokeh[0] * 10,
                                        #   K_bokeh=5,
                                          gamma=2,
                                          disp_focus=para_disp_focus[img_i],
                                          defocus_scale=1)
            rgb = torch.reshape(rgb, [-1, 3])
        else:
            rgb=rgb_ori
            # disp_input=disp_main


        optimizer.zero_grad(set_to_none=True)
        
        # disp=disp/(disp.max()+1e-5)
        # t_pred = torch.median(depth)
        # s_pred = torch.mean(torch.abs(depth - t_pred))
        # depth=(depth-t_pred)/(s_pred+1e-5)
        # if i<N_iters:
        # depth=depth.unsqueeze(1).repeat(1,3)
        # disp=disp.unsqueeze(1).repeat(1,3)
        decay_iteration=25
        # if i>N_iters+200000:
        divsor = (i-N_iters) // (decay_iteration * 1000)
        # else:
        #     divsor = i // (decay_iteration * 1000)

        depth_decay_rate = 10

        w_depth = 0.1/(depth_decay_rate ** divsor)

        
        

        if i<N_iters:
            img_loss = nerf_model.img2mse(rgb, target_s)
            # depth_loss=compute_depth_loss(disp,batch_depth_rays)
            loss = img_loss
            # loss = img_loss+depth_loss*w_depth
        # else:
        elif i<N_iters+200000:
            # rgb=rgb.transpose(0,1).reshape(-1,3,N_pixel,N_pixel)
            # target_main=target_main.transpose(0,1).reshape(-1,3,N_pixel,N_pixel)
            # img_loss = ssim_loss_func(rgb, target_main)
            disp=disp.unsqueeze(1).repeat(1,3)
            img_loss = nerf_model.img2mse(rgb, target_main)
            # depth_loss=compute_depth_loss(disp,batch_depth_rays)
            # loss = img_loss+w_depth*depth_loss
            loss = img_loss
        elif i<N_iters+400000:
            rgb=rgb.transpose(0,1).reshape(-1,3,N_pixel,N_pixel)
            target_main=target_main.transpose(0,1).reshape(-1,3,N_pixel,N_pixel)
            img_loss = ssim_loss_func(rgb, target_main)
            loss = img_loss
        # elif i<N_iters+150000:
        #     img_loss = nerf_model.img2mse(rgb_main, target_main)
        else:
            # batch_defocus_rays=torch.abs(target_focus-disp_input)
            # defocus_max=torch.max(batch_defocus_rays)
            target_focus.requires_grad=False
            para_K_bokeh.detach()
            defocus_max=torch.max(target_focus)
            bokeh_range=1/(para_K_bokeh[0]*10)
            
            # if fg:
                # a=batch_defocus_rays
            zeros=torch.zeros_like(target_focus)
            ones=torch.ones_like(target_focus)
            
            # binary_mask=torch.where(target_focus<(defocus_max*0.4),zeros,target_focus)
            # binary_mask=torch.where(target_focus<(defocus_max*0.6),zeros,target_focus)
            binary_mask=torch.where(target_focus<(defocus_max*bokeh_range),zeros,target_focus)
            # binary_mask=torch.where(target_focus<(defocus_max*bokeh_range),zeros,ones)
            
            #nothres 则不写
            # binary_mask=torch.where(binary_mask>(defocus_max*0.9),ones,binary_mask)

            binary_mask=batch_confidence*binary_mask

                
            #     # weight_defocus_main=batch_defocus_rays
            # else:
            #     zeros=torch.zeros_like(target_focus)
            #     ones=torch.ones_like(target_focus)
            #     binary_mask=torch.where(target_focus<(defocus_max*0.6),zeros,target_focus)
            #     # binary_mask=torch.where(binary_mask>0.8,ones,binary_mask)
            #     binary_mask=torch.where(binary_mask>(defocus_max*0.8),ones,binary_mask)
            #     binary_mask=batch_confidence*binary_mask
            target_final=binary_mask*target_s+(1-binary_mask)*target_main
            # img_loss_main = nerf_model.img2mse(rgb_main, target_main)
            img_loss_main = nerf_model.img2mse(rgb_main, target_main)
            # img_loss_wide = nerf_model.img2mse(rgb_wide*batch_confidence, target_s*batch_confidence)
            img_loss = nerf_model.img2mse(rgb, target_final)
            # depth_loss=compute_depth_loss(disp,batch_depth_rays)

            # loss = img_loss+depth_loss*w_depth+img_loss_main
            loss = img_loss+img_loss_main
        

        # patch_main=torch.reshape(patch_main, [-1, patch_main.shape[-2], patch_main.shape[-1]])
        # patch_main = torch.transpose(patch_main, 0, 1)
        
        
        #edge refine
        # grad_loss=1 * pyramid_image_guidance_regularizer(defocus, image, 4)
        # loss = img_loss+grad_loss
        assert torch.isnan(loss).sum() == 0, print(loss)
        
        # log.output_debug('nan:%s',loss)
        # loss = img_loss+0.01*mask_loss
        psnr = nerf_model.mse2psnr(img_loss)
        # assert psnr<40, print(psnr)

        if i >= N_iters+200000 and i<N_iters+400000:
            rgb_0=extras['rgb0']
            # del(extras['rgb0'])
            rgb_0 = torch.reshape(rgb_0, [N_pixel, -1, 3])
            rgb_0 = render_bokeh_no_01_disp(rgb_0, 
                                            disp_input , # disp / max_disp, 
                                            # disp0 / 4. , # disp / max_disp, 
                                            # disp / math.ceil(disp_max) , # disp / max_disp, 
                                            # K_bokeh=5,
                                            K_bokeh=para_K_bokeh[0] * 10,
                                            # K_bokeh=5,
                                            gamma=2,
                                            disp_focus=para_disp_focus[img_i],
                                            defocus_scale=1)
            rgb_0 = torch.reshape(rgb_0, [-1, 3])
            # rgb_0=rgb_0[byte_mask]
            # rgb_0 = torch.reshape(rgb_0, [-1, 3])
            
            # img_loss0 = nerf_model.img2mse(rgb_0, target_s)
            rgb_0=rgb_0.transpose(0,1).reshape(-1,3,N_pixel,N_pixel)

            img_loss0 = ssim_loss_func(rgb_0, target_main)
            
            # defocus=depth0.transpose(0,1).reshape(-1,patch_h,patch_w).unsqueeze(0)
            # defocus=disp0.transpose(0,1).reshape(-1,patch_h,patch_w).unsqueeze(0)
            # grad_loss0=1 * pyramid_image_guidance_regularizer(defocus, image, 4)
            # depth_loss0=compute_depth_loss(depth0,target_depth)
            loss = loss + img_loss0

            # loss = loss + img_loss0+depth_loss0*w_depth

        elif i<N_iters:
            img_loss0 = nerf_model.img2mse(extras['rgb0'], target_s)
            # depth0 = torch.reshape(depth0, [-1, 3])
            # depth_loss0=compute_depth_loss(disp,batch_depth_rays)
            # loss=loss+img_loss0+depth_loss0*w_depth
            loss=loss+img_loss0
        elif i<N_iters+200000:
            img_loss0 = nerf_model.img2mse(extras['rgb0'], target_main)
            # depth0 = torch.reshape(extras['disp0'], [-1, 3])
            # disp=extras['disp0'].unsqueeze(1).repeat(1,3)
            # depth_loss0=compute_depth_loss(disp,batch_depth_rays)
            # loss=loss+img_loss0+depth_loss0*w_depth
            loss=loss+img_loss0
        else:
            # if fg:
                
            #     target_final=batch_defocus_rays
            # else:
            #     weight_defocus_main=
            img_loss0_main = nerf_model.img2mse(extras['rgb_main0'], target_main)
            # img_loss0_wide = nerf_model.img2mse(extras['rgb_wide0']*batch_confidence, target_s*batch_confidence)
            img_loss0 = nerf_model.img2mse(extras['rgb0'], target_final)
            # depth_loss0=compute_depth_loss(disp,batch_depth_rays)
            # loss=loss+img_loss0+depth_loss0*w_depth+img_loss0_main
            loss=loss+img_loss0+img_loss0_main
            # loss=loss+img_loss0+img_loss0_wide



        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###

        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        if i < N_iters:
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        elif i<N_iters+200000:
            new_lrate = args.lrate_bokeh * (decay_rate ** ((global_step-N_iters) / decay_steps))
        elif i<N_iters+400000:
            new_lrate = args.lrate_bokeh * (decay_rate ** ((global_step-N_iters-200000) / decay_steps))
        else:
            new_lrate = args.lrate * (decay_rate ** ((global_step-N_iters-400000) / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
            # naive learning rate
        # new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'network_fn_main_state_dict': render_kwargs_train['network_fn_main'].state_dict(),
                'network_fine_main_state_dict': render_kwargs_train['network_fine_main'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)
            os.makedirs(os.path.join(basedir, expname, 'param'), exist_ok=True)
            path_param = os.path.join(basedir, expname, 'param', '{:06d}_param.tar'.format(i))
            torch.save(bokeh_param.state_dict(), path_param)
            log.output_info(' '.join(['Saved checkpoints at', str(path)]))
            log.output_info(' '.join(['Saved param checkpoints at', str(path_param)]))

        # if i%args.i_video==0 and i > 0:
        #     # Turn on testing mode
        #     with torch.no_grad():
        #         if i<N_iters:

        #             rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test_1)
        #         elif i<N_iters+400000:
        #             # rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
                    
        #             if i==N_iters-1+200000:
        #                 testsavedir = os.path.join(basedir, expname, 'testset_disp')
        #                 os.makedirs(testsavedir, exist_ok=True)
        #                 render_path_disp(torch.Tensor(poses).to(device), hwf, K, args.chunk, render_kwargs_test_2, gt_imgs=images[i_test], savedir=testsavedir)
        #             # else:
        #             rgbs, disps = render_path_main(render_poses, hwf, K, args.chunk, render_kwargs_test_2)
        #         else:
        #             rgbs, disps = render_path_test(render_poses, hwf, K, args.chunk, render_kwargs_test)
        #     print('Done, saving', rgbs.shape, disps.shape)
        #     moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
        #     imageio.mimwrite(moviebase + 'rgb.mp4', nerf_model.to8b(rgbs), fps=30, quality=8)
        #     imageio.mimwrite(moviebase + 'disp.mp4', nerf_model.to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                if i<N_iters:
                    
                # render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
                    render_path(torch.Tensor(poses).to(device), hwf, K, args.chunk, render_kwargs_test_1, gt_imgs=images[i_test], savedir=testsavedir)
                elif i<N_iters+400000:
                    
                # render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
                    render_path_main(torch.Tensor(poses).to(device), hwf, K, args.chunk, render_kwargs_test_2, gt_imgs=images[i_test], savedir=testsavedir)
                else:
                    render_path_test(torch.Tensor(poses).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            log.output_debug(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            
            if i > N_iters+200000:
                logger_para.log_info(' '.join(['Logged at Iters: ', str(i)]))
                logger_para.log_info(' '.join(['Train param focus: ', str(para_disp_focus)]))
                logger_para.log_info(' '.join(['Train param aperture: ', str(para_K_bokeh)]))

        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()