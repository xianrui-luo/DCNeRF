import os
import cv2
import time
import imageio
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

import nerf_utils
import nerf_model as nerf_model


from bokeh_renderer.scatter import ModuleRenderScatter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian_blur(x, r, sigma=None):
    r = int(round(r))
    if sigma is None:
        sigma = 0.3 * (r - 1) + 0.8
    x_grid, y_grid = torch.meshgrid(torch.arange(-int(r), int(r) + 1), torch.arange(-int(r), int(r) + 1))
    kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / 2 / sigma ** 2)
    kernel = kernel.float() / kernel.sum()
    kernel = kernel.expand(1, 1, 2*r+1, 2*r+1).to(x.device)
    x = F.pad(x, pad=(r, r, r, r), mode='replicate')
    x = F.conv2d(x, weight=kernel, padding=0)
    return x


def render_bokeh(rgbs, 
                 disps, 
                 K_bokeh=20, 
                 gamma=4, 
                 disp_focus=90/255, 
                 defocus_scale=1):
    
    classical_renderer = ModuleRenderScatter().to(device)

    # disps =  (disps - disps.min()) / (disps.ma x()- disps.min())
    disps = disps / disps.max()
    
    signed_disp = disps - disp_focus
    defocus = (K_bokeh) * signed_disp / defocus_scale

    defocus = defocus.unsqueeze(0).unsqueeze(0).contiguous()
    rgbs = rgbs.permute(2, 0, 1).unsqueeze(0).contiguous()

    bokeh_classical = classical_renderer(rgbs**gamma, defocus*defocus_scale)
    bokeh_classical = bokeh_classical ** (1/gamma)
    bokeh_classical = bokeh_classical[0].permute(1, 2, 0)
    return bokeh_classical

def render_bokeh_no_01_disp(rgbs, 
                            disps,
                            K_bokeh=20, 
                            gamma=4, 
                            disp_focus=90/255, 
                            defocus_scale=1):
    
    classical_renderer = ModuleRenderScatter().to(device)

    # disps =  (disps - disps.min()) / (disps.ma x()- disps.min())
    # disps = disps / disps.max()
    
    signed_disp = disps - disp_focus
    defocus = (K_bokeh) * signed_disp / defocus_scale

    defocus = defocus.unsqueeze(0).unsqueeze(0).contiguous()
    rgbs = rgbs.permute(2, 0, 1).unsqueeze(0).contiguous()

    bokeh_classical = classical_renderer(rgbs**gamma, defocus*defocus_scale)
    bokeh_classical = bokeh_classical ** (1/gamma)
    bokeh_classical = bokeh_classical[0].permute(1, 2, 0)
    return bokeh_classical


