# ******************************************************************************
# Copyright 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from core import mod_utils as utils
from core.models import Actor, Critic


class Off_Policy_Algo(object):
    """Classes implementing TD3 and DDPG off-policy learners

         Parameters:
               args (object): Parameter class


     """
    def __init__(self, wwid, algo_name, state_dim, action_dim, actor_lr, critic_lr, gamma, tau, init_w = True):

        self.algo_name = algo_name; self.gamma = gamma; self.tau = tau

        #Initialize actors
        self.actor = Actor(state_dim, action_dim, wwid)
        if init_w: self.actor.apply(utils.init_weights)
        self.actor_target = Actor(state_dim, action_dim, wwid)
        utils.hard_update(self.actor_target, self.actor)
        self.actor_optim = Adam(self.actor.parameters(), actor_lr)


        self.critic = Critic(state_dim, action_dim)
        if init_w: self.critic.apply(utils.init_weights)
        self.critic_target = Critic(state_dim, action_dim)
        utils.hard_update(self.critic_target, self.critic)
        self.critic_optim = Adam(self.critic.parameters(), critic_lr)

        self.loss = nn.MSELoss()

        self.actor_target.cuda(); self.critic_target.cuda(); self.actor.cuda(); self.critic.cuda()
        self.num_critic_updates = 0

        #Statistics Tracker
        self.action_loss = {'min':[], 'max': [], 'mean':[], 'std':[]}
        self.policy_loss = {'min':[], 'max': [], 'mean':[], 'std':[]}
        self.critic_loss = {'mean':[]}
        self.q = {'min':[], 'max': [], 'mean':[], 'std':[]}
        self.val = {'min':[], 'max': [], 'mean':[], 'std':[]}

    def compute_stats(self, tensor, tracker):
        """Computes stats from intermediate tensors

             Parameters:
                   tensor (tensor): tensor
                   tracker (object): logger

             Returns:
                   None


         """
        tracker['min'].append(torch.min(tensor).item())
        tracker['max'].append(torch.max(tensor).item())
        tracker['mean'].append(torch.mean(tensor).item())
        tracker['mean'].append(torch.mean(tensor).item())

    def update_parameters(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch, num_epoch=1, **kwargs):
        """Runs a step of Bellman upodate and policy gradient using a batch of experiences

             Parameters:
                  state_batch (tensor): Current States
                  next_state_batch (tensor): Next States
                  action_batch (tensor): Actions
                  reward_batch (tensor): Rewards
                  done_batch (tensor): Done batch
                  num_epoch (int): Number of learning iteration to run with the same data

             Returns:
                   None

         """

        if isinstance(state_batch, list): state_batch = torch.cat(state_batch); next_state_batch = torch.cat(next_state_batch); action_batch = torch.cat(action_batch); reward_batch = torch.cat(reward_batch). done_batch = torch.cat(done_batch)

        for _ in range(num_epoch):
            ########### CRITIC UPDATE ####################

            #Compute next q-val, next_v and target
            with torch.no_grad():
                #Policy Noise
                policy_noise = np.random.normal(0, kwargs['policy_noise'], (action_batch.size()[0], action_batch.size()[1]))
                policy_noise = torch.clamp(torch.Tensor(policy_noise), -kwargs['policy_noise_clip'], kwargs['policy_noise_clip'])

                #Compute next action_bacth
                next_action_batch = self.actor_target.forward(next_state_batch) + policy_noise.cuda()
                next_action_batch = torch.clamp(next_action_batch, 0,1)

                #Compute Q-val and value of next state masking by done
                q1, q2, _ = self.critic_target.forward(next_state_batch, next_action_batch)
                q1 = (1 - done_batch) * q1
                q2 = (1 - done_batch) * q2

                #Select which q to use as next-q (depends on algo)
                if self.algo_name == 'TD3' or self.algo_name == 'TD3_actor_min': next_q = torch.min(q1, q2)
                elif self.algo_name == 'DDPG': next_q = q1
                elif self.algo_name == 'TD3_max': next_q = torch.max(q1, q2)

                #Compute target q and target val
                target_q = reward_batch + (self.gamma * next_q)


            self.critic_optim.zero_grad()
            current_q1, current_q2, current_val = self.critic.forward((state_batch), (action_batch))
            self.compute_stats(current_q1, self.q)

            dt = self.loss(current_q1, target_q)

            if self.algo_name == 'TD3' or self.algo_name == 'TD3_max': dt = dt + self.loss(current_q2, target_q)
            self.critic_loss['mean'].append(dt.item())

            dt.backward()

            self.critic_optim.step()
            self.num_critic_updates += 1


            #Delayed Actor Update
            if self.num_critic_updates % kwargs['policy_ups_freq'] == 0:

                actor_actions = self.actor.forward(state_batch)
                Q1, Q2, val = self.critic.forward(state_batch, actor_actions)

                # if self.args.use_advantage: policy_loss = -(Q1 - val)
                policy_loss = -Q1

                self.compute_stats(policy_loss,self.policy_loss)
                policy_loss = policy_loss.mean()


                self.actor_optim.zero_grad()



                policy_loss.backward(retain_graph=True)
                self.actor_optim.step()


            if self.num_critic_updates % kwargs['policy_ups_freq'] == 0: utils.soft_update(self.actor_target, self.actor, self.tau)
            utils.soft_update(self.critic_target, self.critic, self.tau)










