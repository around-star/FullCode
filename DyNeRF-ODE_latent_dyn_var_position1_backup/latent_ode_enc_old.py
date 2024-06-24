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
        decoder_pose=None,
        z0_encoder_type = "odernn",
        linear = None):

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
        self.latent_net = LatentNetwork(input_size=20, 
                                        latent_size=512)
        self.embed_angle = embed_angle
        self.decoder_pose = decoder_pose
        self.z0_encoder_type = z0_encoder_type
        print("Encoder: ", z0_encoder_type)
        #self.linear = linear

    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, latent_truth_time_steps,
        latent_time_steps_to_pred=None, angle=None, mask = None, n_traj_samples = 1, run_backwards = True, mode = None,
        warmup=False):
        
        #if warmup:
        #    latent_time_input = torch.squeeze((self.num_frames * latent_truth_time_steps)).int()
        #else:
        #    latent_time_input = torch.squeeze((torch.arange(0,15))).int()
            
        latent_time_input = torch.squeeze((self.num_frames * latent_truth_time_steps)).int()
        #print(latent_time_input)
        if len(latent_time_input.shape) == 0:
            latent_time_input = torch.unsqueeze(latent_time_input, 0)
        latent_embeddings = self.latent_net(latent_time_input).float().squeeze()
        if warmup:
            return latent_embeddings, None
        

        angle = self.embed_angle(angle[0]).squeeze()      
        #angle = angle[0]  ## THIS
        #latent_embeddings = torch.cat([angle, latent_embeddings], dim=-1)
        #latent_embeddings = angle
        #latent_embeddings = torch.unsqueeze(latent_embeddings, 0)
        if len(angle.shape) == 1:  ## REPLACE ANGLE WITH LATENT EMBEDDINGS
            truth = angle.unsqueeze(0).unsqueeze(0)
        else:
            truth = angle.unsqueeze(0)
        
        #angle = angle.unsqueeze(0).unsqueeze(0)  ## THIS
        if len(latent_embeddings.shape) == 1:  ## REPLACE ANGLE WITH LATENT EMBEDDINGS
            latent_embeddings = latent_embeddings.unsqueeze(0).unsqueeze(0)
        else:
            latent_embeddings = latent_embeddings.unsqueeze(0)
        #print(truth.shape)
        #latent_embeddings = self.latent_embedder(torch.unsqueeze(latent_truth_time_steps, -1))
        #latent_embeddings_to_pred = self.latent_embedder(torch.unsqueeze(latent_time_steps_to_pred, -1))
        #truth = latent_embeddings
        truth_w_mask = truth
        
        
        if isinstance(self.encoder_z0, Encoder_z0_ODE_RNN) or \
            isinstance(self.encoder_z0, Encoder_z0_RNN):
            
            #latent_embeddings_to_pred = self.latent_net(torch.squeeze(latent_time_steps_to_pred).int()).float()
            #print(latent_embeddings.shape)
            #print(angle)
            
            #if mask is not None:
            #    truth_w_mask = torch.cat((truth, mask), -1)
            #truth_w_mask = torch.cat([truth_w_mask, latent_embeddings], dim=-1)
            first_point_mu, first_point_std = self.encoder_z0(
                truth_w_mask, truth_time_steps, run_backwards = run_backwards)
            means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
            sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
            first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)


        
            first_point_std = first_point_std.abs()
            assert(torch.sum(first_point_std < 0) == 0.)

            first_point_enc_aug = first_point_enc
            means_z0_aug = means_z0

        if self.z0_encoder_type == 'mlp':
            h = truth_w_mask
            for i, l in enumerate(self.encoder_z0):
                h = self.encoder_z0[i](h)
                h = F.relu(h)
                if i == 3:
                    h = torch.cat([truth_w_mask, h], -1)
                    
            first_point_enc_aug = h

        
        #angle = torch.zeros([1,1,21])
        #print(angle.shape)
        #exit()
        #first_point_enc_aug = torch.cat([angle, first_point_enc_aug], dim=-1)  ## THIS
        first_point_enc_aug = torch.cat([latent_embeddings, first_point_enc_aug], dim=-1)
        #first_point_enc_aug = torch.cat([first_point_enc_aug, latent_embeddings], dim=-1)
        #first_point_enc_aug = self.linear(first_point_enc_aug)  ## LINEAR LAYER ADDED
        # Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
        sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)

        pred_x = self.decoder(sol_y)
        pred_pose = self.decoder_pose(sol_y)
        #print(pred_x.shape)
        #pred_x = sol_y

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


    def forward(self, batch_dict, warmup=False, freeze=False):
        # batch_dict["tp_to_predict"] = batch_dict["tp_to_predict"].to(self.device)
        #batch_dict["observed_data"] = batch_dict["observed_data"].to(self.device)
        #batch_dict["data_to_predict"] = batch_dict["data_to_predict"].to(self.device)
        # batch_dict["observed_tp"] = batch_dict["observed_tp"].to(self.device)
        #batch_dict["observed_mask"] = batch_dict["observed_mask"].to(self.device)
        #batch_dict["mask_predicted_data"] = batch_dict["mask_predicted_data"].to(self.device)
        #batch_dict["times"] = ((self.num_frames-1)*batch_dict["times"].to(self.device)).int()
        #batch_dict["times_to_pred"] = ((self.num_frames-1)*batch_dict["times_to_pred"].to(self.device)).int()
        angle = batch_dict["angle"].to(self.device)
        #print(angle)
        angle_to_pred = batch_dict["angle_to_pred"].to(self.device).squeeze()
        angle_latent_to_pred = self.embed_angle(angle_to_pred)
        
        batch_dict["times"] = batch_dict["times"].to(self.device)
        batch_dict["times_to_pred"] = batch_dict["times_to_pred"].to(self.device)
        # print(len(batch_dict["times"]))
        # exit()
        #print(batch_dict["times_to_pred"])
        win_start = int(batch_dict["win_start"].cpu().item())
        if warmup:
            #seen = random.randint(0,1)  ## Not used because there is only one latent
            seen = 0
            # seen = random.randint(0,49)
            #self.latent_net.train()
            self.latent_net.requires_grad = True
            self.diffeq_solver.requires_grad = False
        else:
            #win_start = torch.randint(0,15,(1,))
            seen = np.random.choice(torch.arange(0,batch_dict["times_to_pred"].shape[-1]).cpu().detach().numpy())
            #print(batch_dict["times_to_pred"].shape[-1])
            #self.latent_net.eval()
            #if freeze:
            #    self.latent_net.requires_grad = True
            #else:    
            self.latent_net.requires_grad = False 
            self.diffeq_solver.requires_grad = True

        latent_truth_time_steps = batch_dict["times"]
        # latent_truth_time_steps = torch.unsqueeze(batch_dict["observed_tp"], dim=0)
        truth_time_steps = torch.squeeze(latent_truth_time_steps)
        latent_time_steps_to_pred = batch_dict["times_to_pred"]#[:, seen]
        # latent_time_steps_to_pred = torch.unsqueeze(batch_dict["tp_to_predict"], dim=0)
        time_steps_to_predict = torch.squeeze(latent_time_steps_to_pred)#[seen]
        #print(latent_truth_time_steps)
        #print(truth_time_steps)
        #print(time_steps_to_predict)
        #print("Win Start:" , win_start)
        #print("Angle: ", angle)
        # print(seen)
        # exit()
        
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
        #print(latents.shape)
        #exit()
        #return latents, seen
        if warmup:
            #return latents[seen], seen, None, None  ## THIS
            return latents, seen, None, None
        return latents[seen], seen, latents_pose, angle_latent_to_pred.squeeze()

    def next_latent(self, latent, times_obs, times_to_pred, angle=None, loc=None, run_backwards=True, n_traj_samples=1):
        #time_steps_to_predict = torch.tensor([0.5, 0.75])
        #truth_time_steps = torch.tensor([0, 0.25])
        """len_latent = 2*len(latent)
        time_latent = torch.arange(0, len_latent)/len_latent
        truth_time_steps = time_latent[:len_latent//2]
        time_steps_to_predict = time_latent[len_latent//2:]"""
        if not angle:
            angle=loc
        truth_time_steps = torch.squeeze(times_obs)
        if len(truth_time_steps.shape) == 0:
            truth_time_steps = torch.unsqueeze(truth_time_steps, 0)
        time_steps_to_predict = torch.squeeze(times_to_pred)
        #print(truth_time_steps)
        #print(time_steps_to_predict)
        #exit()
        if latent==None:
            # win_start = random.randint(0,3)
            # win_start = 0
            #latents = self.latent_net(torch.tensor(list(range(20)))).float()  ## THIS
            latents = self.latent_net(torch.tensor([0])).float()
            #return torch.squeeze(latents), None
            #latents = self.latent_net(torch.arange(0,15)).float()
            #latents = latents.mean(0)
            #latent = torch.tensor([0]*512).float()
            #latent = torch.nn.init.normal_(latent, mean=0.0, std = 0.01/(512**(1/2)))

            
            angle = self.embed_angle(torch.tensor([angle])).squeeze()
            if len(angle.shape) == 1:   ## REPLACE WITH LATENT EMBEDDINGS
                truth = angle.unsqueeze(0).unsqueeze(0)
            else:
                truth = angle.unsqueeze(0)
            #latent = angle
            #latent = torch.cat([angle, latent], dim=-1)
            #latent = torch.unsqueeze(latent, 0)
        
            #truth = latents

            truth_w_mask = truth
            #angle = angle.unsqueeze(0).unsqueeze(0)  ## THIS
            if len(latents.shape) == 1:  ## REPLACE ANGLE WITH LATENT EMBEDDINGS
                latents = latents.unsqueeze(0).unsqueeze(0)
            else:
                latents = latents.unsqueeze(0)
            #latents = latents.unsqueeze(0).unsqueeze(0)
        #print(truth_w_mask.shape)
        #exit()
            #if mask is not None:
            #    truth_w_mask = torch.cat((truth, mask), -1)
        if self.z0_encoder_type == "odernn":
            first_point_mu, first_point_std = self.encoder_z0(
                truth_w_mask, truth_time_steps, run_backwards = run_backwards)

            means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
            sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
            first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)

            
            first_point_std = first_point_std.abs()
            assert(torch.sum(first_point_std < 0) == 0.)

            first_point_enc_aug = first_point_enc
            means_z0_aug = means_z0
            
        if self.z0_encoder_type == 'mlp':
            h = truth_w_mask
            for i, l in enumerate(self.encoder_z0):
                h = self.encoder_z0[i](h)
                h = F.relu(h)
                if i == 3:
                    h = torch.cat([truth_w_mask, h], -1)
                    
            first_point_enc_aug = h

        # Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
        
        
        #first_point_enc_aug = torch.cat([angle, first_point_enc_aug], dim=-1)  ## THIS
        first_point_enc_aug = torch.cat([latents, first_point_enc_aug], dim=-1)
        #first_point_enc_aug = self.linear(first_point_enc_aug)
        sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)


        pred_x = self.decoder(sol_y)
        return torch.squeeze(pred_x), None

    def next_latent_batch(self, latent, times_obs, times_to_pred, angle=None, loc=None, run_backwards=True, n_traj_samples=1):

        if not angle:
            angle=loc
        #truth_time_steps = torch.squeeze(times_obs)
        truth_time_steps = torch.tensor([0.], requires_grad=True)
        if len(truth_time_steps.shape) == 0:
            truth_time_steps = torch.unsqueeze(truth_time_steps, 0)
        time_steps_to_predict = torch.squeeze(times_to_pred)
        time_steps_to_predict.requires_grad = True
        
        
        angle_input = torch.tensor(angle)
        angle_input.requires_grad = True
        if latent==None:

            latents = self.latent_net(torch.tensor([0])).float()

            
            angle = self.embed_angle(angle_input).squeeze()
            if len(angle.shape) == 1:   ## REPLACE WITH LATENT EMBEDDINGS
                truth = angle.unsqueeze(0).unsqueeze(0)
            else:
                truth = angle.unsqueeze(1)

            truth_w_mask = truth
            print(truth_w_mask.shape)
            
            latents = latents.repeat(angle.shape[0], 1)
            if len(latents.shape) == 1:  ## REPLACE ANGLE WITH LATENT EMBEDDINGS
                latents = latents.unsqueeze(0).unsqueeze(0)
            else:
                latents = latents.unsqueeze(0)

        if self.z0_encoder_type == "odernn":
            #print("CHECK")
            #truth_w_mask = torch.cat([truth_w_mask, latents], dim=-1)
            first_point_mu, first_point_std = self.encoder_z0(
                truth_w_mask, truth_time_steps, run_backwards = run_backwards)

            #output_scalar = first_point_mu.sum()  
            #output_scalar.backward(retain_graph=True)
              
            means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
            sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
            first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)

            
            first_point_std = first_point_std.abs()
            assert(torch.sum(first_point_std < 0) == 0.)

            first_point_enc_aug = first_point_enc
            means_z0_aug = means_z0
            
        if self.z0_encoder_type == 'mlp':
            print("Truth Shape: ", truth_w_mask.shape)
            h = truth_w_mask
            for i, l in enumerate(self.encoder_z0):
                h = self.encoder_z0[i](h)
                h = F.relu(h)
                if i == 4:
                    h = torch.cat([truth_w_mask, h], -1)
                    
            first_point_enc_aug = h
            
            ### [40,1,63] --> [1,40,63]
            first_point_enc_aug = first_point_enc_aug.permute(1,0,2)


        first_point_enc_aug = torch.cat([latents, first_point_enc_aug], dim=-1)
        #first_point_enc_aug = self.linear(first_point_enc_aug)
        sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)
        
        pred_x = torch.squeeze(self.decoder(sol_y))
        print("Pred X: ", pred_x.shape)
        #output_scalar = pred_x.sum()
        #output_scalar.backward(retain_graph=True)
        #print(time_steps_to_predict.grad.shape)
        divergences = []
        for i in range(pred_x.shape[0]):
            
            div = torch.autograd.grad(outputs=pred_x[i], inputs=time_steps_to_predict, grad_outputs=torch.ones_like(pred_x[i]), create_graph=True, retain_graph=True)[0]
            print("Grad: ", div.shape)
            divergences.append(div)
        
        return torch.squeeze(pred_x), torch.stack(divergences)

