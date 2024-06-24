###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import numpy as np
import sklearn as sk
import numpy as np
#import gc
import torch
import torch.nn as nn
from torch.nn.functional import relu

import rnn_utils as utils
from rnn_utils import get_device
from encoder_decoder import *
#from lib.likelihood_eval import *

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent
from run_dnerf_helpers import LatentNetwork, get_embedder
#from lib.base_models import VAE_Baseline
import random
import torch.nn.functional as F


class LatentODE(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver, 
        z0_prior, device, obsrv_std = None, 
        use_binary_classif = False, use_poisson_proc = False,
        linear_classifier = False,
        classif_per_tp = False,
        n_labels = 1,
        train_classif_w_reconstr = False,
        num_frames=80,
        latent_embedder_out_dim=512,
        latent_embedder = None,
        embed_angle=None,
        decoder_pose = None,
        z0_encoder_type = "odernn"):

        super(LatentODE, self).__init__()
        self.latent_embedder_out_dim = latent_embedder_out_dim
        #self.latent_embedder = latent_embedder

        self.encoder_z0 = encoder_z0
        self.diffeq_solver = diffeq_solver
        self.device = device
        self.decoder = decoder
        self.use_poisson_proc = use_poisson_proc
        self.num_frames = num_frames
        self.z0_prior = z0_prior
        self.latent_dim = latent_dim
        print("Num Frames: ", num_frames)
        self.latent_net = LatentNetwork(input_size=12, 
                                        latent_size=512)
        self.embed_angle = embed_angle
        self.decoder_pose = decoder_pose
        self.z0_encoder_type = z0_encoder_type

    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, latent_truth_time_steps,
        latent_time_steps_to_pred=None, angle=None, mask = None, n_traj_samples = 1, run_backwards = True, mode = None,
        warmup=False):
        
        angle = self.embed_angle(angle[0]).squeeze()
        if len(angle.shape) == 1:  ## REPLACE ANGLE WITH LATENT EMBEDDINGS
            truth = angle.unsqueeze(0).unsqueeze(0)
        else:
            truth = angle.unsqueeze(0)
        

        if self.z0_encoder_type == 'odernn':
            first_point_mu, first_point_std = self.encoder_z0(
                    truth, truth_time_steps, run_backwards = run_backwards)
            #print(first_point_mu.shape)
            means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
            sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
            first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)
            first_point_enc_aug = first_point_enc
            
        if self.z0_encoder_type == 'mlp':
            h = truth
            for i, l in enumerate(self.encoder_z0):
                h = self.encoder_z0[i](h)
                if i < 8:
                    h = F.relu(h)
                if i == 4:
                    h = torch.cat([truth, h], -1)
                    
            first_point_enc_aug = h
        #print(first_point_enc.shape)
        
        #angle = angle.unsqueeze(0).unsqueeze(0)
        #extra = torch.zeros([1,1,6])
        #first_point_enc_aug = torch.cat([first_point_enc, angle], dim=-1)

        #print(angle.shape)
        sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)
        #print(sol_y.shape)
        

        pred_x = self.decoder(sol_y)
        pred_pose = self.decoder_pose(sol_y)

        return torch.squeeze(pred_x), torch.squeeze(pred_pose)


    def sample_traj_from_prior(self, time_steps_to_predict, n_traj_samples = 1):
        # input_dim = starting_point.size()[-1]
        # starting_point = starting_point.view(1,1,input_dim)

        # Sample z0 from prior
        starting_point_enc = self.z0_prior.sample([n_traj_samples, 1, self.latent_dim]).squeeze(-1)

        starting_point_enc_aug = starting_point_enc
        if self.use_poisson_proc:
            n_traj_samples, n_traj, n_dims = starting_point_enc.size()
            # append a vector of zeros to compute the integral of lambda
            zeros = torch.zeros(n_traj_samples, n_traj,self.input_dim).to(self.device)
            starting_point_enc_aug = torch.cat((starting_point_enc, zeros), -1)

        sol_y = self.diffeq_solver.sample_traj_from_prior(starting_point_enc_aug, time_steps_to_predict, 
            n_traj_samples = 3)

        if self.use_poisson_proc:
            sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)
        
        return torch.squeeze(self.decoder(sol_y)), None

    # def forward(self, batch_dict):
    #     batch_dict["tp_to_predict"] = batch_dict["tp_to_predict"].to(self.device)
    #     batch_dict["observed_data"] = batch_dict["observed_data"].to(self.device)
    #     batch_dict["observed_tp"] = batch_dict["observed_tp"].to(self.device)
    #     batch_dict["observed_mask"] = batch_dict["observed_mask"].to(self.device)
    #     batch_dict["mask_predicted_data"] = batch_dict["mask_predicted_data"].to(self.device)
    #     batch_dict["times"] = ((self.num_frames-1)*batch_dict["times"].to(self.device)).int()
    #     batch_dict["times_to_pred"] = ((self.num_frames-1)*batch_dict["times_to_pred"].to(self.device)).int()
    #     #batch_dict["times"] = batch_dict["times"].to(self.device)
    #     #batch_dict["times_to_pred"] = batch_dict["times_to_pred"].to(self.device)
    #     latents, latents_to_pred, _ = self.get_reconstruction(
    #         time_steps_to_predict=batch_dict["tp_to_predict"],
    #         truth=batch_dict["observed_data"],
    #         truth_time_steps=batch_dict["observed_tp"],
    #         latent_truth_time_steps=batch_dict["times"],
    #         latent_time_steps_to_pred = batch_dict["times_to_pred"],
    #         mask=None)

    #     return latents, latents_to_pred

    def forward(self, batch_dict, warmup=False, freeze=False, div=False):
        # batch_dict["tp_to_predict"] = batch_dict["tp_to_predict"].to(self.device)
        #batch_dict["observed_data"] = batch_dict["observed_data"].to(self.device)
        #batch_dict["data_to_predict"] = batch_dict["data_to_predict"].to(self.device)
        # batch_dict["observed_tp"] = batch_dict["observed_tp"].to(self.device)
        #batch_dict["observed_mask"] = batch_dict["observed_mask"].to(self.device)
        #batch_dict["mask_predicted_data"] = batch_dict["mask_predicted_data"].to(self.device)
        #batch_dict["times"] = ((self.num_frames-1)*batch_dict["times"].to(self.device)).int()
        #batch_dict["times_to_pred"] = ((self.num_frames-1)*batch_dict["times_to_pred"].to(self.device)).int()
        angle = batch_dict["angle"].to(self.device)
        angle_to_pred = batch_dict["angle_to_pred"].to(self.device).squeeze()
        angle_latent_to_pred = self.embed_angle(angle_to_pred)
        
        batch_dict["times"] = batch_dict["times"].to(self.device)
        batch_dict["times_to_pred"] = batch_dict["times_to_pred"].to(self.device)
        # print(len(batch_dict["times"]))
        # exit()
        #print(batch_dict["times_to_pred"])
        win_start = int(batch_dict["win_start"].cpu().item())
        if warmup:
            # seen = random.randint(0,1)
            seen = 0
            # seen = random.randint(0,49)
            #self.latent_net.train()
            self.latent_net.requires_grad = True
            self.diffeq_solver.requires_grad = False
        else:
            seen = np.random.choice(torch.arange(0,batch_dict["times_to_pred"].shape[-1]).cpu().detach().numpy())
            #self.latent_net.eval()
            self.latent_net.requires_grad = False
            # self.latent_net.requires_grad = True  ## for start_obs
            self.diffeq_solver.requires_grad = True

        latent_truth_time_steps = batch_dict["times"]
        # latent_truth_time_steps = torch.unsqueeze(batch_dict["observed_tp"], dim=0)
        truth_time_steps = torch.squeeze(latent_truth_time_steps)
        latent_time_steps_to_pred = batch_dict["times_to_pred"]#[:, seen]
        # latent_time_steps_to_pred = torch.unsqueeze(batch_dict["tp_to_predict"], dim=0)
        time_steps_to_predict = torch.squeeze(latent_time_steps_to_pred)#[seen]

        if len(time_steps_to_predict.shape) == 0:
            time_steps_to_predict = torch.unsqueeze(time_steps_to_predict, 0)
        if len(truth_time_steps.shape) == 0:
            truth_time_steps = torch.unsqueeze(truth_time_steps, 0)
        #print(time_steps_to_predict)
        #exit()
        #print(latent_truth_time_steps)
        #print(truth_time_steps)
        latents, latents_pose = self.get_reconstruction(
            time_steps_to_predict=time_steps_to_predict,
            truth=None,
            truth_time_steps=truth_time_steps,
            latent_truth_time_steps=latent_truth_time_steps,
            #latent_truth_time_steps=(win_start)/self.num_frames + latent_truth_time_steps,
            angle = angle,
            mask=None,
            warmup = warmup)
        
        divergence = None
        if warmup:
            #return latents[seen], seen, None, None  ## THIS
            return latents, seen, None, None
        #return latents[seen], seen, latents_pose, angle_latent_to_pred.squeeze()
        return latents, seen, latents_pose, angle_latent_to_pred.squeeze(), divergence

    def next_latent(self, latent, times_obs, times_to_pred, angle=0.2, run_backwards=True, n_traj_samples=1):
        #time_steps_to_predict = torch.tensor([0.5, 0.75])
        #truth_time_steps = torch.tensor([0, 0.25])
        """len_latent = 2*len(latent)
        time_latent = torch.arange(0, len_latent)/len_latent
        truth_time_steps = time_latent[:len_latent//2]
        time_steps_to_predict = time_latent[len_latent//2:]"""
        truth_time_steps = torch.squeeze(times_obs)
        if len(truth_time_steps.shape) == 0:
            truth_time_steps = torch.unsqueeze(truth_time_steps, 0)
        time_steps_to_predict = torch.squeeze(times_to_pred)

        if latent==None:

            
            angle = self.embed_angle(torch.tensor([angle])).squeeze()
            if len(angle.shape) == 1:  ## REPLACE ANGLE WITH LATENT EMBEDDINGS
                truth = angle.unsqueeze(0).unsqueeze(0)
            else:
                truth = angle.unsqueeze(0)

            truth_w_mask = truth

        if self.z0_encoder_type == 'odernn':
            first_point_mu, first_point_std = self.encoder_z0(
                truth_w_mask, truth_time_steps, run_backwards = run_backwards)

            means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
            sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
            first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0) 
            first_point_enc_aug = first_point_enc
        
        if self.z0_encoder_type == 'mlp':
            h = truth_w_mask
            for i, l in enumerate(self.encoder_z0):
                h = self.encoder_z0[i](h)
                if i < 8:
                    h = F.relu(h)
                if i == 4:
                    h = torch.cat([truth_w_mask, h], -1)
                    
            first_point_enc_aug = h
        
        #angle = angle.unsqueeze(0).unsqueeze(0)
        #first_point_enc_aug = torch.cat([first_point_enc, angle], dim=-1)
    

        # Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
        
        sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)
        #return torch.squeeze(sol_y), None

        pred_x = self.decoder(sol_y)
        return torch.squeeze(pred_x), None

