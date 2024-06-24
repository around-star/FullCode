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

import utils
#from vid_ode import VidODE
from rnn_vae import RNN_VAE
from create_latent_ode_model import create_LatentODE_model
import pickle
from torch.distributions.normal import Normal
from torch.nn.functional import l1_loss, cosine_similarity
from torch.nn import MSELoss, HuberLoss
from run_dnerf_helpers import LatentNetwork

#from robust_loss_pytorch import AdaptiveLossFunction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs_pos, inputs_time, latent):
        num_batches = inputs_pos.shape[0]

        out_list = []
        dx_list = []
        for i in range(0, num_batches, chunk):
            #out, dx = fn(inputs_pos[i:i+chunk], [inputs_time[0][i:i+chunk], inputs_time[1][i:i+chunk]])
            out, dx = fn(inputs_pos[i:i+chunk], inputs_time[i:i+chunk], latent)
            out_list += [out]
            dx_list += [dx]
        return torch.cat(out_list, 0), torch.cat(dx_list, 0)
    return ret


def run_network(inputs, viewdirs, frame_time, fn, embed_fn, embeddirs_fn, embedtime_fn, netchunk=1024*64,
                embd_time_discr=True, latent=None):
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
    outputs_flat, position_delta_flat = batchify(fn, netchunk)(embedded, input_frame_time_flat, latent)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    position_delta = torch.reshape(position_delta_flat, list(inputs.shape[:-1]) + [position_delta_flat.shape[-1]])
    return outputs, position_delta


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


def render_path(args, render_poses, render_times, hwf, latent_store, chunk, render_kwargs, gt_imgs=None, savedir=None,
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
        latent_frame_time = str(int(frame_time * args.num_frames))
        if not latent_frame_time in latent_store.keys():
            continue
        latent = latent_store[latent_frame_time]
        render_kwargs["latent"] = latent
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

def render_path_extrapolate(args, render_poses, rnn_model, render_times, hwf, latent_store, chunk, render_kwargs, gt_imgs=None, savedir=None,
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

    """for i, (c2w, frame_time) in enumerate(zip(tqdm(render_poses), render_times)):
        latent_frame_time = str(int(frame_time * args.num_frames))
        if not latent_frame_time in latent_store.keys():
            continue
        latent = latent_store[latent_frame_time]
        render_kwargs["latent"] = latent
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
                imageio.imwrite(filename, rgb8_gt)"""

    #print(latent_store.keys())
    #latents = torch.stack([latent_store[str(int(render_times[-2]*args.num_frames))], latent_store[str(int(render_times[-1]*args.num_frames))]])
    latents = torch.stack([latent_store[str(0)], latent_store[str(1)], latent_store[str(2)]])
    c2w = render_poses[-1]
    for j in range(13):
        print(j,'/[13]')
        latent_nexts, _ = rnn_model.next_latent(latents)
        for latent_next in latent_nexts:
          render_kwargs["latent"] = torch.squeeze(latent_next)
          #rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], frame_time=frame_time, **render_kwargs)
          rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], frame_time=int(1 * args.num_frames), **render_kwargs)
          rgbs.append(rgb.cpu().numpy())
          disps.append(disp.cpu().numpy())
        latents = latent_nexts
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
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
    latent_ode_model = create_LatentODE_model(512, z0_prior, 0.01, device, num_frames=args.num_frames+1)

    grad_vars = list(latent_ode_model.parameters())

    """model = NeRF.get_by_name(args.nerf_type, num_frames = args.num_frames + 1, D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, input_ch_time=input_ch_time,
                 use_viewdirs=args.use_viewdirs, embed_fn=embed_fn,
                 zero_canonical=not args.not_zero_canonical,
                 latent_dim = latent_ode_model.latent_embedder_out_dim).to(device)"""
    model = LatentNetwork(1200)
    #grad_vars += list(model.parameters())

    model_fine = None
    if args.use_two_models_for_fine:
        model_fine = NeRF.get_by_name(args.nerf_type, num_frames = args.num_frames + 1, D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, input_ch_time=input_ch_time,
                          use_viewdirs=args.use_viewdirs, embed_fn=embed_fn,
                          zero_canonical=not args.not_zero_canonical,
                          latent_dim = latent_ode_model.latent_embedder_out_dim).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, ts, network_fn, latent : run_network(inputs, viewdirs, ts, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                embedtime_fn=embedtime_fn,
                                                                netchunk=args.netchunk,
                                                                embd_time_discr=args.nerf_type!="temporal",
                                                                latent = latent)

    #grad_vars=[]
    #for k,v in model.named_parameters():
    #  grad_vars += [{"params":v, 'lr':args.lrate}]

    #for k,v in rnn_vae_model.named_parameters():
    #  grad_vars += [{"params":v, 'lr':args.lrate*5}]
    # Create optimizer
    #optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    optimizer = torch.optim.Adamax(params=grad_vars, lr=args.lrate)
    ## Load Latent ODE Model Model
    """ode_ckpt = torch.load('/hdd4/hiran/DyNeRF-ODE_latent_wo_encdec/logs_single_vpt_big_pendulam_nODE_sample_size_75_org_time_l2_lr_5e-4/pitcher_base/040000.tar')
    latent_ode_model.load_state_dict(ode_ckpt["vid_ode"])
    optimizer.load_state_dict(ode_ckpt['optimizer_state_dict'])"""

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

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, model, latent_ode_model


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
                use_two_models_for_fine=False,
                latent=None):
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
            raw, position_delta = network_query_fn(pts, viewdirs, frame_time, network_fn)
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

        else:
            if use_two_models_for_fine:
                raw, position_delta_0 = network_query_fn(pts, viewdirs, frame_time, network_fn)
                rgb_map_0, disp_map_0, acc_map_0, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

            else:
                with torch.no_grad():
                    raw, _ = network_query_fn(pts, viewdirs, frame_time, network_fn, latent)
                    _, _, _, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
    run_fn = network_fn if network_fine is None else network_fine
    raw, position_delta = network_query_fn(pts, viewdirs, frame_time, run_fn, latent)
    rgb_map, disp_map, acc_map, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'z_vals' : z_vals,
           'position_delta' : position_delta}
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
    #parser.add_argument("--lrate", type=float, default=5e-4, 
    #                    help='learning rate')
    parser.add_argument("--lrate", type=float, default=1e-2, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    #parser.add_argument("--chunk", type=int, default=512*32, 
    #                    help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    #parser.add_argument("--netchunk", type=int, default=512*64, 
    #                    help='number of pts sent through network in parallel, decrease if running out of memory')
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
    parser.add_argument('--sample_size', type=int, default=100, help="Number of time points to sub-sample")
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('--init_dim', type=int, default=32)
    parser.add_argument('--dec_diff', type=str, default='dopri5', choices=['dopri5', 'euler', 'adams', 'rk4'])
    parser.add_argument('--n_layers', type=int, default=2, help='A number of layer of ODE func')
    parser.add_argument('--n_downs', type=int, default=2)
    parser.add_argument('--input_size', type=int, default=128)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--run_backwards', action='store_true', default=True)
    return parser


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
        print("Num Frames: ", args.num_frames)

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

    if args.render_test:
        render_poses = np.array(poses[i_test])
        render_times = np.array(times[i_test])

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
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, latent_net, vid_ode_model = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    render_times = torch.Tensor(render_times).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        pickle_file = open('logs_single_vpt_latent_ode/pitcher_base/005000.pkl', 'rb')
        latent_store = pickle.load(pickle_file)
        vid_ode_model.load_state_dict(torch.load('logs_single_vpt_latent_ode/pitcher_base/005000.tar')['vid_ode'])
        #print(latent_store)
        #exit()
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path_extrapolate(args, render_poses, vid_ode_model, render_times, hwf, latent_store, args.chunk, render_kwargs_test, gt_imgs=images,
                                  savedir=testsavedir, render_factor=args.render_factor, save_also_gt=False)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, focal, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    #images = torch.Tensor(images).to(device)
    #poses = torch.Tensor(poses).to(device)
    #times = torch.Tensor(times).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    #N_iters = args.N_iter + 1
    N_iters = 50001
    args.i_weights = (N_iters - 1)//10
    #print(args.i_weights)
    #exit()
    print('Begin')

    # Summary writers
    #writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    train_dataloader = dataloaders["train_dataloader"]
    #load_dict = torch.load('/hdd4/hiran/DyNeRF_latent_embedding/logs_single_vpt_512_coarse/pitcher_base/110000.tar')["network_fine_state_dict"]
    load_dict = torch.load('../DyNeRF_latent_embedding_copy/logs_big_pendulam_single_vpt_512_coarse_n2f15_once/pitcher_base/400000.tar')["network_fine_state_dict"]
    #load_dict = torch.load('../DyNeRF_latent_embedding_copy/logs_big_pendulam_single_vpt_512_coarse_n2f15_one/pitcher_base/160000.tar')["network_fine_state_dict"]
    #print(latent_net.state_dict().keys())
    #exit()
    new_state_dict = {'fc.weight':load_dict["latent_time_net.fc.weight"]}
    latent_net.load_state_dict(new_state_dict)
    latent_net.requires_grad = False
    #exit()
    """latent_store = {}
    for frame in range(args.num_frames+1):
        latent_store.update({str(frame): torch.zeros([vid_ode_model.latent_embedder_out_dim ])})"""
    l2_loss = MSELoss()
    #huber_loss = HuberLoss(delta=0.2)
    #robust_loss = AdaptiveLossFunction(512, torch.float32, device)
    #time_all = torch.arange(0,15)/14
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        """if use_batching:
            raise NotImplementedError("Time not implemented")
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            if i >= args.precrop_iters_time:
                img_i = np.random.choice(i_train)
            else:
                skip_factor = i / float(args.precrop_iters_time) * len(i_train)
                max_sample = max(int(skip_factor), 3)
                img_i = np.random.choice(i_train[:max_sample])"""

            #target = images[img_i]
            #pose = poses[img_i, :3, :4]
            #frame_time = times[img_i]
        optimizer.zero_grad()
        utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lrate / 10)
        data_dict = utils.get_data_dict(train_dataloader)
        batch_dict = utils.get_next_batch(data_dict)
        org_times_all = batch_dict["times_all"]
        times_all = batch_dict["tp_all"]
        #print(org_times_all.shape)
        #print(org_times_all[org_times_all!=3].shape)
        #print(times_all.shape)
        #exit()
        seen = np.random.choice(torch.arange(0,80).cpu().detach().numpy())
        #print(seen)
        #org_time = batch_dict["times"]
        #org_time = torch.unsqueeze(org_times_all[:,seen],0)
        org_time = org_times_all[:,seen:seen+10]
        #org_time = torch.unsqueeze(torch.unsqueeze(time_all[0], 0), 0)
        #org_time = torch.tensor([[0]])
        #print(org_time)
        #exit()
        #org_times_to_pred = batch_dict["times_to_pred"]
        org_times_to_pred = batch_dict["times_all"]
        #org_times_to_pred = torch.cat((org_times_all[:,:seen],org_times_all[:,seen+1:]), dim=1)
        #org_times_to_pred = torch.cat((org_times_all[:,:seen],org_times_all[:,seen+10:]), dim=1)
        #org_times_to_pred = torch.unsqueeze(time_all[1:], 0)
        #all_times = torch.arange(0, len(org_times_to_pred[0])+1)/(len(org_times_to_pred[0])+1)
        #times_obs = torch.squeeze(batch_dict["observed_tp"])
        #times_obs = torch.squeeze(times_all)[seen:seen+10]
        times_obs = torch.squeeze(times_all)[seen:seen+10]
        #times_obs = torch.squeeze(org_time)
        #times_obs = all_times[0]
        if len(times_obs.shape) == 0:
            times_obs = torch.unsqueeze(times_obs,0)
        #times_to_pred = torch.squeeze(batch_dict["tp_to_predict"])
        #times_to_pred = torch.squeeze(org_times_to_pred)
        #times_to_pred = all_times[1:]
        #times_to_pred = torch.cat((times_all[:seen],times_all[seen+10:]), dim=0)
        times_to_pred = torch.squeeze(times_all)
        if len(times_to_pred.shape) == 0:
            times_to_pred = torch.unsqueeze(times_to_pred,0)
        #print(times_obs)
        #print(times_to_pred)
        #print(args.num_frames * org_time)
        #print(args.num_frames * org_times_to_pred)
        #exit()
        #print((args.num_frames * batch_dict["times"]).int())
        latents_gt = torch.squeeze(latent_net((args.num_frames * org_time).int()))
        #latents_gt = latent_net((args.num_frames * org_time).int()))
        #print(latents_gt.shape)
        #exit()
        #latents_gt = torch.squeeze(latent_net((torch.tensor([[0]]))))
        latents_gt_to_pred = torch.squeeze(latent_net((args.num_frames * org_times_to_pred).int()))
        #print(latents_gt_to_pred.shape)
        #print((args.num_frames/(args.num_frames + 0)) * batch_dict["times"])
        #print((args.num_frames/(args.num_frames + 0)) * batch_dict["times_to_pred"])
        #exit()
        latents_to_pred, _ = vid_ode_model.next_latent(latents_gt, 
                                                       times_obs,
                                                       times_to_pred)
        #print(latents_to_pred.shape)
        #print(latents_gt_to_pred.shape)
        #exit()
        #total_loss = l1_loss(latents_to_pred, latents_gt_to_pred)
        #total_loss = 1 - torch.mean(cosine_similarity(torch.unsqueeze(latents_to_pred,0),
        #                                              torch.unsqueeze(latents_gt_to_pred,0)))
        total_loss = l2_loss(latents_to_pred, latents_gt_to_pred)
        #print("Shape: ", total_loss.shape)
        #print("Dtype: ", total_loss.dtype)
        #total_loss = torch.mean(robust_loss.lossfun(total_loss))
        #total_loss = huber_loss(latents_to_pred, latents_gt_to_pred)
        #rgbs = []
        """total_loss = 0
        optimizer.zero_grad()
        all_targets = torch.squeeze(batch_dict["data_to_predict"])
        all_poses = torch.squeeze(batch_dict["poses_to_pred"])
        all_frame_times = torch.squeeze(batch_dict["times_to_pred"])
        for i_idx in range(len(all_targets)):
            target = all_targets[i_idx]
            pose = all_poses[i_idx, :3, :4]
            frame_time = all_frame_times[i_idx]
            frame_time = (args.num_frames * frame_time).int().item()
            latent = latents[i_idx]
            latent_to_pred = latents_to_pred[i_idx]
            latent_store.update({str(frame_time):latent})

            render_kwargs_train["latent"] = latent
            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

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
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            #####  Core optimization loop  #####
            #rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, frame_time=frame_time,
            #                                        verbose=i < 10, retraw=True,
            #                                        **render_kwargs_train)
            rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, frame_time=int(frame_time*args.num_frames),
                                                    verbose=i < 10, retraw=True,
                                                    **render_kwargs_train)
            #rgbs.append(rgb)

        
            if args.add_tv_loss:
                #frame_time_prev = times[img_i - 1] if img_i > 0 else None
                #frame_time_next = times[img_i + 1] if img_i < times.shape[0] - 1 else None
                latent_prev = latents[i_idx - 1] if i_idx > 0 else None
                latent_next = latents[i_idx + 1] if i_idx < len(latents) -1 else None

                if frame_time_prev is not None and frame_time_next is not None:
                    if np.random.rand() > .5:
                        frame_time_prev = None
                    else:
                        frame_time_next = None

                #if frame_time_prev is not None:
                if latent_prev is not None:
                    #rand_time_prev = frame_time_prev + (frame_time - frame_time_prev) * torch.rand(1)[0]
                    rand_time_prev = 0
                    #_, _, _, extras_prev = render(H, W, focal, chunk=args.chunk, rays=batch_rays, frame_time=rand_time_prev,
                    #                                verbose=i < 10, retraw=True, z_vals=extras['z_vals'].detach(),
                    #                                **render_kwargs_train)
                    render_kwargs_train["latent"] = latent_prev
                    _, _, _, extras_prev = render(H, W, focal, chunk=args.chunk, rays=batch_rays, frame_time=int(rand_time_prev*args.num_frames),
                                                    verbose=i < 10, retraw=True, z_vals=extras['z_vals'].detach(),
                                                    **render_kwargs_train)

                #if frame_time_next is not None:
                if latent_next is not None:
                    #rand_time_next = frame_time + (frame_time_next - frame_time) * torch.rand(1)[0]
                    rand_time_next = 0
                    #_, _, _, extras_next = render(H, W, focal, chunk=args.chunk, rays=batch_rays, frame_time=rand_time_next,
                    #                                verbose=i < 10, retraw=True, z_vals=extras['z_vals'].detach(),
                    #                                **render_kwargs_train)
                    render_kwargs_train["latent"] = latent_next
                    _, _, _, extras_next = render(H, W, focal, chunk=args.chunk, rays=batch_rays, frame_time=int(rand_time_next*args.num_frames),
                                                    verbose=i < 10, retraw=True, z_vals=extras['z_vals'].detach(),
                                                    **render_kwargs_train)

            
            img_loss = img2mse(rgb, target_s)

            tv_loss = 0
            if args.add_tv_loss:
                #if frame_time_prev is not None:
                if latent_prev is not None:
                    tv_loss += ((extras['position_delta'] - extras_prev['position_delta']).pow(2)).sum()
                    if 'position_delta_0' in extras:
                        tv_loss += ((extras['position_delta_0'] - extras_prev['position_delta_0']).pow(2)).sum()
                #if frame_time_next is not None:
                if latent_next is not None:
                    tv_loss += ((extras['position_delta'] - extras_next['position_delta']).pow(2)).sum()
                    if 'position_delta_0' in extras:
                        tv_loss += ((extras['position_delta_0'] - extras_next['position_delta_0']).pow(2)).sum()
                tv_loss = tv_loss * args.tv_loss_weight
            #latent_loss = 0
            #if i > 200:
            #    latent_loss = l1_loss(latent, latent_to_pred)
            loss = img_loss + tv_loss# + 0.05*latent_loss
            total_loss += loss
        
        #optimizer.zero_grad()
        #psnr = mse2psnr(img_loss)
        total_loss = total_loss/(i_idx+1)
        psnr = mse2psnr(total_loss)
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        if args.do_half_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if args.do_half_precision:
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()"""        
        total_loss.backward()

        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        """decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        #new_lrate_embedding = 5 * args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = new_lrate"""
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            save_dict = {
                'global_step': global_step,
                #'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if render_kwargs_train['network_fine'] is not None:
                save_dict['network_fine_state_dict'] = render_kwargs_train['network_fine'].state_dict()
            ### Save Vid-ODE
            save_dict['vid_ode'] = vid_ode_model.state_dict()
            #pickle.dump(latent_store, latent_file)
            torch.save(save_dict, path)
            print('Saved checkpoints at', path)

        if i % args.i_print == 0:
            tqdm_txt = f"[TRAIN] Iter: {i} Loss_fine: {total_loss.item()}"
            if args.add_tv_loss:
                tqdm_txt += f" TV: {tv_loss.item()}"
            tqdm.write(tqdm_txt)

            #writer.add_scalar('loss', img_loss.item(), i)
            #print('loss', img_loss.item(), i)
            #writer.add_scalar('psnr', psnr.item(), i)
            #print('psnr', psnr.item(), i)
            #if 'rgb0' in extras:
                #writer.add_scalar('loss0', img_loss0.item(), i)
                #print('loss0', img_loss0.item(), i)
                #writer.add_scalar('psnr0', psnr0.item(), i)
                #print('psnr0', psnr0.item(), i)
            #if args.add_tv_loss:
                #writer.add_scalar('tv', tv_loss.item(), i)
                #print('tv', tv_loss.item(), i)
        #del total_loss, latents_gt_to_pred, latents_to_pred, batch_dict, data_dict, latents_gt
        """del loss, img_loss, psnr, target_s
        if 'rgb0' in extras:
            del img_loss0, psnr0
        if args.add_tv_loss:
            del tv_loss
        del rgb, disp, acc, extras"""

        """if i%args.i_img==0:
            torch.cuda.empty_cache()
            # Log a rendered validation view to Tensorboard
            img_i=np.random.choice(i_val)
            target = images[img_i]
            pose = poses[img_i, :3,:4]
            frame_time = times[img_i]
            with torch.no_grad():
                #rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose, frame_time=frame_time,
                #                                    **render_kwargs_test)
                rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose, frame_time=int(frame_time*args.num_frames),
                                                    **render_kwargs_test)

            psnr = mse2psnr(img2mse(rgb, target))
            writer.add_image('gt', to8b(target.cpu().numpy()), i, dataformats='HWC')
            writer.add_image('rgb', to8b(rgb.cpu().numpy()), i, dataformats='HWC')
            writer.add_image('disp', disp.cpu().numpy(), i, dataformats='HW')
            writer.add_image('acc', acc.cpu().numpy(), i, dataformats='HW')

            if 'rgb0' in extras:
                writer.add_image('rgb_rough', to8b(extras['rgb0'].cpu().numpy()), i, dataformats='HWC')
            if 'disp0' in extras:
                writer.add_image('disp_rough', extras['disp0'].cpu().numpy(), i, dataformats='HW')
            if 'z_std' in extras:
                writer.add_image('acc_rough', extras['z_std'].cpu().numpy(), i, dataformats='HW')

            print("finish summary")
            writer.flush()"""

        """if i%args.i_video==0 or i == 5000:
            # Turn on testing mode
            print("Rendering video...")
            with torch.no_grad():
                savedir = os.path.join(basedir, expname, 'frames_{}_spiral_{:06d}_time/'.format(expname, i))
                rgbs, disps = render_path(args, render_poses, render_times, hwf, latent_store, args.chunk, render_kwargs_test, savedir=savedir)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)"""

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)
        """if i%args.i_testset==0  or i == 5000:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            print('Testing poses shape...', poses[i_test].shape)
            with torch.no_grad():
                render_path(args, torch.Tensor(poses[i_test]).to(device), torch.Tensor(times[i_test]).to(device),
                            hwf, latent_store, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')"""

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()