import os
import imageio
import time
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange


from run_dnerf_helpers import *

from load_blender import load_blender_data
import numpy as np

try:
    from apex import amp
except ImportError:
    pass
#os.environ["CUDA_VISIBLE_DEVICES"]=0,1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False
from run_dnerf_helpers import LatentNetwork
import time

import pandas as pd

#from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from create_latent_ode_model import create_LatentODE_model
from torch.distributions.normal import Normal

import seaborn as sns

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs_pos, inputs_time):
        num_batches = inputs_pos.shape[0]

        out_list = []
        #dx_list = []
        for i in range(0, num_batches, chunk):
            #out, dx = fn(inputs_pos[i:i+chunk], [inputs_time[0][i:i+chunk], inputs_time[1][i:i+chunk]])
            #out, dx = fn(inputs_pos[i:i+chunk], inputs_time[i:i+chunk])
            out = fn(inputs_pos[i:i+chunk], inputs_time[i:i+chunk])
            out_list += [out]
            #dx_list += [dx]
        return torch.cat(out_list, 0)#, torch.cat(dx_list, 0)
    return ret


def run_network(inputs, viewdirs, frame_time, fn, embed_fn, embeddirs_fn, embedtime_fn, netchunk=1024*64,
                embd_time_discr=True):
    """Prepares inputs and applies network 'fn'.
    inputs: N_rays x N_points_per_ray x 3
    viewdirs: N_rays x 3
    frame_time: N_rays x 1
    """
    assert len(torch.unique(frame_time)) == 1, "Only accepts all points from same time"
    cur_time = torch.unique(frame_time)[0]

    # embed position
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    # embed time
    if embd_time_discr:
        B, N, _ = inputs.shape
        input_frame_time = frame_time[:, None].expand([B, N, 1])
        input_frame_time_flat = torch.reshape(input_frame_time, [-1, 1])
        #embedded_time = embedtime_fn(input_frame_time_flat)
        #embedded_times = [embedded_time, embedded_time]

    else:
        assert NotImplementedError

    # embed views
    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    #outputs_flat, position_delta_flat = batchify(fn, netchunk)(embedded, embedded_times)
    outputs_flat = batchify(fn, netchunk)(embedded, input_frame_time_flat)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    #position_delta = torch.reshape(position_delta_flat, list(inputs.shape[:-1]) + [position_delta_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1., frame_time=None,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    frame_time = frame_time * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far, frame_time], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(args, render_poses, render_times, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None,
                render_factor=0, save_also_gt=False, i_offset=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    if savedir is not None:
        save_dir_estim = os.path.join(savedir, "estim")
        save_dir_gt = os.path.join(savedir, "gt")
        if not os.path.exists(save_dir_estim):
            os.makedirs(save_dir_estim)
        if save_also_gt and not os.path.exists(save_dir_gt):
            os.makedirs(save_dir_gt)

    rgbs = []
    disps = []

    for i, (c2w, frame_time) in enumerate(zip(tqdm(render_poses), render_times)):
        #rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], frame_time=frame_time, **render_kwargs)
        rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], frame_time=int(frame_time * args.num_frames), **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if savedir is not None:
            rgb8_estim = to8b(rgbs[-1])
            filename = os.path.join(save_dir_estim, '{:03d}.png'.format(i+i_offset))
            imageio.imwrite(filename, rgb8_estim)
            if save_also_gt:
                rgb8_gt = to8b(gt_imgs[i])
                filename = os.path.join(save_dir_gt, '{:03d}.png'.format(i+i_offset))
                imageio.imwrite(filename, rgb8_gt)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, 3, args.i_embed)
    embedtime_fn, input_ch_time = get_embedder(args.multires, 1, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, 3, args.i_embed)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF.get_by_name(args.nerf_type, num_frames = args.num_frames + 1, D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, input_ch_time=input_ch_time,
                 use_viewdirs=args.use_viewdirs, embed_fn=embed_fn,
                 zero_canonical=not args.not_zero_canonical).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.use_two_models_for_fine:

        model_fine = NeRF.get_by_name(args.nerf_type, num_frames = args.num_frames + 1, D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, input_ch_time=input_ch_time,
                          use_viewdirs=args.use_viewdirs, embed_fn=embed_fn,
                          zero_canonical=not args.not_zero_canonical).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, ts, network_fn : run_network(inputs, viewdirs, ts, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                embedtime_fn=embedtime_fn,
                                                                netchunk=args.netchunk,
                                                                embd_time_discr=args.nerf_type!="temporal")

    """grad_vars=[]
    for k,v in model.named_parameters():

      if "latent_time_net" in k:
        continue
      grad_vars += [{"params":v, 'lr':args.lrate}]

    grad_vars += [{"params":model.latent_time_net.parameters(),
                'lr':args.lrate*0.5}]"""

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    if args.do_half_precision:
        print("Run model at half precision")
        if model_fine is not None:
            [model, model_fine], optimizers = amp.initialize([model, model_fine], optimizer, opt_level='O1')
        else:
            model, optimizers = amp.initialize(model, optimizer, opt_level='O1')

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        if args.do_half_precision:
            amp.load_state_dict(ckpt['amp'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine': model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'use_two_models_for_fine' : args.use_two_models_for_fine,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])
        # rgb_map = rgb_map + torch.cat([acc_map[..., None] * 0, acc_map[..., None] * 0, (1. - acc_map[..., None])], -1)

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                z_vals=None,
                use_two_models_for_fine=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 9 else None
    bounds = torch.reshape(ray_batch[...,6:9], [-1,1,3])
    near, far, frame_time = bounds[...,0], bounds[...,1], bounds[...,2] # [-1,1]
    z_samples = None
    rgb_map_0, disp_map_0, acc_map_0, position_delta_0 = None, None, None, None

    if z_vals is None:
        t_vals = torch.linspace(0., 1., steps=N_samples)
        if not lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


        if N_importance <= 0:
            #raw, position_delta = network_query_fn(pts, viewdirs, frame_time, network_fn)
            raw = network_query_fn(pts, viewdirs, frame_time, network_fn)
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

        else:
            if use_two_models_for_fine:
                #raw, position_delta_0 = network_query_fn(pts, viewdirs, frame_time, network_fn)
                raw = network_query_fn(pts, viewdirs, frame_time, network_fn)
                rgb_map_0, disp_map_0, acc_map_0, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

            else:
                with torch.no_grad():
                    #raw, _ = network_query_fn(pts, viewdirs, frame_time, network_fn)
                    raw = network_query_fn(pts, viewdirs, frame_time, network_fn)
                    _, _, _, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
    run_fn = network_fn if network_fine is None else network_fine
    #raw, position_delta = network_query_fn(pts, viewdirs, frame_time, run_fn)
    raw = network_query_fn(pts, viewdirs, frame_time, run_fn)
    rgb_map, disp_map, acc_map, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'z_vals' : z_vals}
           #'position_delta' : position_delta}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        if rgb_map_0 is not None:
            ret['rgb0'] = rgb_map_0
        if disp_map_0 is not None:
            ret['disp0'] = disp_map_0
        if acc_map_0 is not None:
            ret['acc0'] = acc_map_0
        if position_delta_0 is not None:
            ret['position_delta_0'] = position_delta_0
        if z_samples is not None:
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


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
    parser.add_argument("--nerf_type", type=str, default="original",
                        help='nerf network type')
    parser.add_argument("--N_iter", type=int, default=500000,
                        help='num training iterations')
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    #parser.add_argument("--netwidth", type=int, default=512, 
    #                    help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    #parser.add_argument("--netwidth_fine", type=int, default=512, 
    #                    help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--do_half_precision", action='store_true',
                        help='do half precision training and inference')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    #parser.add_argument("--lrate", type=float, default=1e-4, 
    #                    help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    #parser.add_argument("--chunk", type=int, default=1024*32, 
    #                    help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--chunk", type=int, default=512*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--not_zero_canonical", action='store_true',
                        help='if set zero time is not the canonic space')
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
    parser.add_argument("--use_two_models_for_fine", action='store_true',
                        help='use two models for fine results')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_iters_time", type=int, default=0,
                        help='number of steps to train on central time')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')
    parser.add_argument("--add_tv_loss", action='store_true',
                        help='evaluate tv loss')
    parser.add_argument("--tv_loss_weight", type=float,
                        default=1.e-4, help='weight of tv loss')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=2,
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
    parser.add_argument("--factor", type=int, default=8, 
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
    parser.add_argument("--i_img",     type=int, default=10000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=5000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=5000,
                        help='frequency of render_poses video saving')
    
    parser.add_argument('--irregular', action='store_true', default=False, help="Train with irregular time-steps")
    parser.add_argument('--extrap', action='store_true', default=True, help="Set extrapolation mode. If this flag is not set, run interpolation mode.")
    parser.add_argument('--window_size', type=int, default=20, help="Window size to sample")
    parser.add_argument('--sample_size', type=int, default=15, help="Number of time points to sub-sample")
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('--init_dim', type=int, default=32)
    parser.add_argument('--dec_diff', type=str, default='dopri5', choices=['dopri5', 'euler', 'adams', 'rk4'])
    parser.add_argument('--n_layers', type=int, default=2, help='A number of layer of ODE func')
    parser.add_argument('--n_downs', type=int, default=2)
    parser.add_argument('--input_size', type=int, default=128)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--run_backwards', action='store_true', default=True)

    return parser

def interpolate_latents(latents):
    latents_new = []
    for i in range(len(latents)):
        #print(len(latents_new))
        latents_new.append(latents[i])
        if i < (len(latents)-1):
            latents_new.append((latents[i]+latents[i+1])/2)
    #print(np.array(latents_new).shape)
    #exit()
    return np.array(latents_new)

def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data

    if args.dataset_type == 'blender':
        #images, poses, times, render_poses, render_times, hwf, i_split = load_blender_data(args, args.datadir, args.half_res, args.testskip)
        dataloaders, times, render_poses, render_times, hwf, i_split = load_blender_data(args, args.datadir, args.half_res, args.testskip, device)
        print('Loaded blender', render_poses.shape, hwf, args.datadir)
        #i_train, i_val, i_test = i_split

        # Number of training views
        args.num_frames = np.unique(times).shape[0] - 1

        near = 0.
        far = 4.

        """if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]"""

        # images = [rgb2hsv(img) for img in images]

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    #min_time, max_time = times[i_train[0]], times[i_train[-1]]
    #assert min_time == 0., "time must start at 0"
    #assert max_time == 1., "max time must be 1"

    # Cast intrinsics to right types
    H, W, focal = hwf
    #H, W = int(H), int(W)
    hwf = [H, W, focal]


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

    # Create nerf model
    #render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    #latent_net = LatentNetwork(1200)
    load_dict = torch.load('/hdd5/hiran/NeRF_ODE/DyNeRF-ODE_latent_dyn_var_position1/logs_ball_roll_together_diff_starting_points_wo_enc/ss_100_euler_all_pred_n_obs_1_15motions_loc_linear_latent_nODE_loc_supervised_l2/pitcher_base/080000.tar')#["network_fine_state_dict"]
    ode_dict = load_dict["vid_ode"]
    #print(load_dict.keys())
    #exit()
    #new_state_dict = {'fc.weight':nerf_load_dict["latent_time_net.fc.weight"]}
    #latent_net.load_state_dict(new_state_dict)
    #latent_net.requires_grad = False
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
    latent_ode_model = create_LatentODE_model(512, z0_prior, 0.01, device, num_frames=args.num_frames+1)
    #latent_ode_model.load_state_dict(torch.load('log_damp_once/logs_sample_100_randfirst_rnn_s10_pred_all_l2_0/pitcher_base/050000.tar')['vid_ode'])
    latent_ode_model.load_state_dict(ode_dict)
    #latents_true = latent_net(torch.arange(0,100).int())
    #latents_true = latents_true.cpu().detach().numpy()
    #time_to_predict = torch.arange(0,13)[1:]/12
    #latents = []
    #for t in time_to_predict:
    #    print(t)
    # latents, _ = latent_ode_model.next_latent(torch.squeeze(latent_net(torch.tensor([[0,1,2,3,4,5,6,7,8,9]]))),
    #                                         torch.tensor([0,1,2,3,4,5,6,7,8,9]),
    #                                         torch.arange(0,300)/99)
    angles = [[0.32490700483322144, 0.4572400450706482, 0.10000000149011612], 
            [-0.1850930154323578, 0.16724005341529845, 0.10000000149011612],
            [0.3649072051048279, 1.0572359561920166, 0.10000000149011612],
            [-0.16509287059307098, 0.8372400403022766, 0.10000000149011612],
            [-0.06509285420179367, 0.34724003076553345, 0.10000000149011612],
            [0.12490701675415039, 1.0372400283813477, 0.10000000149011612]]
    #angles = [-0.6, -0.4, -0.2, 0.2, 0.4, 0.6]
    #angles = [-0.2, 0.4, -0.6, 0.2, 0.6, -0.4]
    times = torch.arange(0,1500)/100
    time_obs = times[:1]
    time_to_pred = times
    latents, _ = latent_ode_model.next_latent(None,
                                            torch.tensor(time_obs),
                                            torch.tensor(time_to_pred), loc = angles[0])
    
    all_latents = latents.cpu().detach().numpy()
    for angle in angles[1:]:
        
        latents, _ = latent_ode_model.next_latent(None,
                                            torch.tensor(time_obs),
                                            torch.tensor(time_to_pred), loc = angle)
        
        latents = latents.cpu().detach().numpy()
        all_latents= np.concatenate((all_latents, latents), axis=0)
        #latents.append(latent.cpu().detach().numpy())
        #latents_new = interpolate_latents(latents)
        #latents = latents.reshape([-1,512])
        
        #total_latents = np.concatenate((latents, latents_true))
        #total_latents = latents
        #latents = np.stack(latents)
    #print("Latents shape: ", latents[-10:-6])
    #exit()
    total_latents = all_latents
    #print(total_latents.shape)
    time_start = time.time()
    tsne = TSNE(n_components=2, perplexity=50)
    tsne_result = tsne.fit_transform(total_latents)
    #print("TSNE SHAPE: ", tsne_result)
    #exit()
    #tsne_results = tsne_results_total[:len(latents)]
    #tsne_true_results = tsne_results_total[len(latents):]


    #tsne_true = TSNE(n_components=2, perplexity=10)
    #tsne_true_results = tsne_true.fit_transform(latents_true)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
    
    """def create_colormap(color, num_steps):
        colors = [(0, 0, 0, 1)]  # Start with black
        for i in range(1, num_steps + 1):
            alpha = i / (num_steps + 1)
            colors.append((*color, alpha))
        return LinearSegmentedColormap.from_list('custom_cmap', colors)"""
    
    def create_colormap(color, num_steps):
        cmap = plt.cm.get_cmap(color)
        colors = cmap(np.linspace(0, 1, num_steps))
        return colors
    
    plt.figure(figsize=(10, 8))
    #plt.scatter(tsne_result[:,0], tsne_result[:,1], label='nODE')
    c = ['Greys', 'Greens', 'Blues', 'Purples', 'pink', 'Reds']
    for i in range(6):
        #cmap = create_colormap(plt.cm.tab10(i), 7)
        print(i)
        cmap = create_colormap(c[i], 1500)
        plt.scatter(tsne_result[i*1500:(i+1)*1500, 0], 
                    tsne_result[i*1500:(i+1)*1500, 1], 
                    label=f'Cluster {i}', color=cmap)
    
    #labels = [0,1]*75# + [2]*(400-150)
    """labels = [0]*100 + [1]*100 + [2]*100 + [3]*100 + [4]*100 + [5]*100 + [6]*100
    labels_req = labels
    #colours = ['b' if label == 0 else 'r' for label in labels_req]
    colours = []
    colormap = get_cmap('plasma')
    colors = [colormap(i / (300-100)) for i in range(300-100)]
    for i, label in enumerate(labels_req):
        if label == 0:
            colours.append('b')
        #elif label == 1:
        #    colours.append('r')
        elif label == 2:
            #colours.append('y')
            colours.append(colors[i-100])
    legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10, label='nODE'),
    #Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='nODE Interpolated'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='nODE Extrapolated'),
    #Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='NeRF')
    ]
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], label='nODE', c=colours)"""


    legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=10, label='Unseen1'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Unseen2'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Seen1'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Seen2'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='pink', markersize=10, label='Senn3'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Unseen3')
    ]
    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(handles=legend_elements, loc="best")
    plt.savefig('tsne_ball_roll_15motions_loc_linear_latent_nODE_loc_supervised_l2/nODE_extrap_1500_80k_ckpt.png')

    """df_subset = {}
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]

    plt.figure(figsize=(16,10))
    scatterplot = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        #hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    fig = scatterplot.get_figure()
    fig.savefig("tsne.png") """


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
