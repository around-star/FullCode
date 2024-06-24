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
from run_dnerf_helpers import LatentNetwork
#from lib.base_models import VAE_Baseline
import random



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
        latent_embedder = None):

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
        self.latent_net = LatentNetwork(input_size=2, 
                                        latent_size=latent_embedder_out_dim)

    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, latent_truth_time_steps,
        latent_time_steps_to_pred=None ,mask = None, n_traj_samples = 1, run_backwards = True, mode = None,
        warmup=False):
        if isinstance(self.encoder_z0, Encoder_z0_ODE_RNN) or \
            isinstance(self.encoder_z0, Encoder_z0_RNN):
            # print((self.num_frames * latent_truth_time_steps).int())
            latent_embeddings = self.latent_net(torch.squeeze(self.num_frames * latent_truth_time_steps).round().int()).float()
            if warmup:
                return latent_embeddings
            #latent_embeddings_to_pred = self.latent_net(torch.squeeze(latent_time_steps_to_pred).int()).float()
            truth = torch.unsqueeze(latent_embeddings, 0)
            #print(truth.shape)
            #latent_embeddings = self.latent_embedder(torch.unsqueeze(latent_truth_time_steps, -1))
            #latent_embeddings_to_pred = self.latent_embedder(torch.unsqueeze(latent_time_steps_to_pred, -1))
            #truth = latent_embeddings
            truth_w_mask = truth
            #if mask is not None:
            #    truth_w_mask = torch.cat((truth, mask), -1)
            first_point_mu, first_point_std = self.encoder_z0(
                truth_w_mask, truth_time_steps, run_backwards = run_backwards)

            means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
            sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
            first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)

        else:
            raise Exception("Unknown encoder type {}".format(type(self.encoder_z0).__name__))
        
        first_point_std = first_point_std.abs()
        assert(torch.sum(first_point_std < 0) == 0.)

        if self.use_poisson_proc:
            n_traj_samples, n_traj, n_dims = first_point_enc.size()
            # append a vector of zeros to compute the integral of lambda
            zeros = torch.zeros([n_traj_samples, n_traj,self.input_dim]).to(get_device(truth))
            first_point_enc_aug = torch.cat((first_point_enc, zeros), -1)
            means_z0_aug = torch.cat((means_z0, zeros), -1)
        else:
            first_point_enc_aug = first_point_enc
            means_z0_aug = means_z0
            
        assert(not torch.isnan(time_steps_to_predict).any())
        assert(not torch.isnan(first_point_enc).any())
        assert(not torch.isnan(first_point_enc_aug).any())

        # Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
        sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)

        if self.use_poisson_proc:
            sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)

            assert(torch.sum(int_lambda[:,:,0,:]) == 0.)
            assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)

        pred_x = self.decoder(sol_y)

        all_extra_info = {
            "first_point": (first_point_mu, first_point_std, first_point_enc),
            "latent_traj": sol_y.detach()
        }

        if self.use_poisson_proc:
            # intergral of lambda from the last step of ODE Solver
            all_extra_info["int_lambda"] = int_lambda[:,:,-1,:]
            all_extra_info["log_lambda_y"] = log_lambda_y

        """if self.use_binary_classif:
            if self.classif_per_tp:
                all_extra_info["label_predictions"] = self.classifier(sol_y)
            else:
                all_extra_info["label_predictions"] = self.classifier(first_point_enc).squeeze(-1)"""
        #print(pred_x.shape)
        #exit()
        return torch.squeeze(pred_x)#, torch.squeeze(latent_embeddings_to_pred), all_extra_info


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

    def forward(self, batch_dict, warmup=False):
        # batch_dict["tp_to_predict"] = batch_dict["tp_to_predict"].to(self.device)
        #batch_dict["observed_data"] = batch_dict["observed_data"].to(self.device)
        #batch_dict["data_to_predict"] = batch_dict["data_to_predict"].to(self.device)
        # batch_dict["observed_tp"] = batch_dict["observed_tp"].to(self.device)
        #batch_dict["observed_mask"] = batch_dict["observed_mask"].to(self.device)
        #batch_dict["mask_predicted_data"] = batch_dict["mask_predicted_data"].to(self.device)
        #batch_dict["times"] = ((self.num_frames-1)*batch_dict["times"].to(self.device)).int()
        #batch_dict["times_to_pred"] = ((self.num_frames-1)*batch_dict["times_to_pred"].to(self.device)).int()
        batch_dict["times"] = batch_dict["times"].to(self.device)
        batch_dict["times_to_pred"] = batch_dict["times_to_pred"].to(self.device)
        # print(len(batch_dict["times"]))
        # exit()
        #print(batch_dict["times_to_pred"])
        # win_start = int(batch_dict["win_start"].cpu().item())
        if warmup:
            seen = random.randint(0,1)
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
        # print(truth_time_steps)
        # print(time_steps_to_predict)
        # print(seen)
        # exit()
        
        if len(time_steps_to_predict.shape) == 0:
            time_steps_to_predict = torch.unsqueeze(time_steps_to_predict, 0)
        if len(truth_time_steps.shape) == 0:
            truth_time_steps = torch.unsqueeze(truth_time_steps, 0)
        #print(time_steps_to_predict)
        #exit()
        latents = self.get_reconstruction(
            time_steps_to_predict=time_steps_to_predict,
            truth=None,
            truth_time_steps=truth_time_steps,
            latent_truth_time_steps=latent_truth_time_steps,
            # latent_truth_time_steps=(2*win_start)/self.num_frames + latent_truth_time_steps,
            latent_time_steps_to_pred = latent_time_steps_to_pred,
            mask=None,
            warmup = warmup)
        #print(latents.shape)
        #exit()
        return latents[seen], seen

    def next_latent(self, latent, times_obs, times_to_pred, run_backwards=True, n_traj_samples=1):
        #time_steps_to_predict = torch.tensor([0.5, 0.75])
        #truth_time_steps = torch.tensor([0, 0.25])
        """len_latent = 2*len(latent)
        time_latent = torch.arange(0, len_latent)/len_latent
        truth_time_steps = time_latent[:len_latent//2]
        time_steps_to_predict = time_latent[len_latent//2:]"""
        truth_time_steps = torch.squeeze(times_obs)
        time_steps_to_predict = torch.squeeze(times_to_pred)
        #print(truth_time_steps)
        #print(time_steps_to_predict)
        #exit()
        if latent==None:
            # win_start = random.randint(0,3)
            # win_start = 0
            # latent = self.latent_net(torch.tensor([2*win_start, 2*win_start+1])).float()
            latent = self.latent_net(torch.tensor([0,1])).float()
        truth = torch.unsqueeze(latent, 0)

        truth_w_mask = truth
        #print(truth_w_mask.shape)
        #exit()
            #if mask is not None:
            #    truth_w_mask = torch.cat((truth, mask), -1)
        first_point_mu, first_point_std = self.encoder_z0(
            truth_w_mask, truth_time_steps, run_backwards = run_backwards)

        means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
        sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
        first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)

         
        first_point_std = first_point_std.abs()
        assert(torch.sum(first_point_std < 0) == 0.)

        if self.use_poisson_proc:
            n_traj_samples, n_traj, n_dims = first_point_enc.size()
            # append a vector of zeros to compute the integral of lambda
            zeros = torch.zeros([n_traj_samples, n_traj,self.input_dim]).to(get_device(truth))
            first_point_enc_aug = torch.cat((first_point_enc, zeros), -1)
            means_z0_aug = torch.cat((means_z0, zeros), -1)
        else:
            first_point_enc_aug = first_point_enc
            means_z0_aug = means_z0
            
        assert(not torch.isnan(time_steps_to_predict).any())
        assert(not torch.isnan(first_point_enc).any())
        assert(not torch.isnan(first_point_enc_aug).any())

        # Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
        sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)

        if self.use_poisson_proc:
            sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)

            assert(torch.sum(int_lambda[:,:,0,:]) == 0.)
            assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)

        pred_x = self.decoder(sol_y)
        return torch.squeeze(pred_x), None

