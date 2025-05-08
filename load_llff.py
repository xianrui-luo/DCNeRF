import numpy as np
import os, imageio

########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
        
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):

    if 'fg' in basedir.split('/')[-1]:
        newbasedir=basedir.replace('fg','wide_match_flow_color_fg')
        ## none
        # newbasedir=basedir.replace('fg','wide_raw')
        print(newbasedir)

    elif 'bg' in basedir.split('/')[-1]:
        newbasedir=basedir.replace('bg_flow','wide_match_flow_color_bg')
    # basedir='/data4/luoxianrui/data/nerf/nerf_aif/iphone/fg'
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])


    
    # img0 = [os.path.join(newbasedir, 'images', f) for f in sorted(os.listdir(os.path.join(newbasedir, 'images'))) \
    #         if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    main_img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(main_img0).shape
    if basedir.split('/')[-1]=='fg':
        confidence_index='consistency_fg'
        # defocus_index='fg_defocus'
    elif basedir.split('/')[-1]=='bg_flow':
        confidence_index='consistency_bg'
        # defocus_index='bg_defocus'
    else:
        print('wrong input')
        return
    if 'fg' in basedir.split('/')[-1]:
        confidence_dir=basedir.replace('fg',confidence_index)
        # defocus_dir=basedir.replace('fg',defocus_index)
    elif 'bg' in basedir.split('/')[-1]:
        confidence_dir=basedir.replace('bg_flow',confidence_index)
        # defocus_dir=basedir.replace('bg_flow',defocus_index)
    # confidence_dir=newbasedir.replace()
    
    sfx = ''
    sfx_main = '_{}'.format(factor)
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        _minify(newbasedir, factors=[factor])
        _minify(confidence_dir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        _minify(newbasedir, resolutions=[[height, width]])
        _minify(confidence_dir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        _minify(newbasedir, resolutions=[[height, width]])
        _minify(confidence_dir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    main_imgdir = os.path.join(basedir, 'images' + sfx)
    # main_imgdir = os.path.join(basedir, 'images' + sfx_main)
    wide_imgdir = os.path.join(newbasedir, 'images' + sfx)
    confidencedir = os.path.join(confidence_dir, 'images' + sfx)
    if basedir.split('/')[-1]=='fg':
        depthdir = basedir.replace('fg','midas_fg')
        # depthdir = basedir.replace('fg','depth_fg')
    elif basedir.split('/')[-1]=='bg_flow':
        depthdir = basedir.replace('bg_flow','midas_bg')
        # depthdir = basedir.replace('bg_flow','depth_bg')
    # # depthdir = '/data4/luoxianrui/data/nerf/nerf_aif/iphone/'
    if not os.path.exists(main_imgdir):
        print( main_imgdir, 'does not exist, returning' )
        return
    if not os.path.exists(confidencedir):
        print( confidencedir, 'does not exist, returning' )
        return
    
    main_imgfiles = [os.path.join(main_imgdir, f) for f in sorted(os.listdir(main_imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    wide_imgfiles = [os.path.join(wide_imgdir, f) for f in sorted(os.listdir(wide_imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    confiles = [os.path.join(confidencedir, f) for f in sorted(os.listdir(confidencedir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    depthfiles = [os.path.join(depthdir, f) for f in sorted(os.listdir(depthdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    # defocusfiles = [os.path.join(defocus_dir, f) for f in sorted(os.listdir(defocus_dir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(main_imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(main_imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(main_imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
    
    
    imgs = imgs = [imread(f)[...,:3]/255. for f in wide_imgfiles]
    imgs = np.stack(imgs, -1)  
    main_imgs = main_imgs = [imread(f)[...,:3]/255. for f in main_imgfiles]
    main_imgs = np.stack(main_imgs, -1)  
    # confidence = confidence = [imageio.imread(f,as_gray=True)/255. for f in confiles]
    confidence = confidence = [imread(f)/255. for f in confiles]
    confidence_imgs = np.stack(confidence, -1)  
    
    
    depth= depth= [imread(f)/65535. for f in depthfiles]
    depth_imgs = np.stack(depth, -1)  
    # defocus= defocus= [imread(f)/255. for f in defocusfiles]
    # defocus_imgs = np.stack(defocus, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    # return poses, bds, imgs
    # return poses, bds, imgs,main_imgs,confidence_imgs
    return poses, bds, imgs,main_imgs,confidence_imgs,depth_imgs

    
            
            
    

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    


def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds
    

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, logname='log.log',path_zflat=False):
    
    # poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    poses, bds, imgs,main_imgs,confidence_imgs,depth_imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    # poses, bds, imgs,main_imgs,confidence_imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    main_imgs = np.moveaxis(main_imgs, -1, 0).astype(np.float32)
    confidence_imgs = np.moveaxis(confidence_imgs, -1, 0).astype(np.float32)
    depth_imgs = np.moveaxis(depth_imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    
    if recenter:
        poses = recenter_poses(poses)
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        # N_views = 360
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    # print('HOLDOUT view is', i_test)
    # log.output_debug('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    main_images = main_imgs.astype(np.float32)

    poses = poses.astype(np.float32)
    confidence_gt = confidence_imgs.astype(np.float32)
    depth_imgs = depth_imgs.astype(np.float32)

    # return images, poses, bds, render_poses, i_test,confidence_gt
    return images, main_images,poses, bds, render_poses, i_test,confidence_gt,depth_imgs


