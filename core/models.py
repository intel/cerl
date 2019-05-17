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
import torch.nn.functional as F
from torch.distributions import Normal


class Actor(nn.Module):
    """Actor model

        Parameters:
              args (object): Parameter class
    """

    def __init__(self, state_dim, action_dim, wwid):
        super(Actor, self).__init__()

        self.wwid = torch.Tensor([wwid])
        l1 = 400; l2 = 300

        # Construct Hidden Layer 1
        self.f1 = nn.Linear(state_dim, l1)
        self.ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.f2 = nn.Linear(l1, l2)
        self.ln2 = nn.LayerNorm(l2)

        #Out
        self.w_out = nn.Linear(l2, action_dim)

    def forward(self, input):
        """Method to forward propagate through the actor's graph

            Parameters:
                  input (tensor): states

            Returns:
                  action (tensor): actions


        """
        #Hidden Layer 1
        out = F.elu(self.f1(input))
        out = self.ln1(out)

        #Hidden Layer 2
        out = F.elu(self.f2(out))
        out = self.ln2(out)

        #Out
        return torch.sigmoid(self.w_out(out))


class Critic(nn.Module):

    """Critic model

        Parameters:
              args (object): Parameter class

    """

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        l1 = 400; l2 = 300

        ######################## Q1 Head ##################
        # Construct Hidden Layer 1 with state
        self.q1f1 = nn.Linear(state_dim + action_dim, l1)
        self.q1ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.q1f2 = nn.Linear(l1, l2)
        self.q1ln2 = nn.LayerNorm(l2)

        #Out
        self.q1out = nn.Linear(l2, 1)


        ######################## Q2 Head ##################
        # Construct Hidden Layer 1 with state
        self.q2f1 = nn.Linear(state_dim + action_dim, l1)
        self.q2ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.q2f2 = nn.Linear(l1, l2)
        self.q2ln2 = nn.LayerNorm(l2)

        #Out
        self.q2out = nn.Linear(l2, 1)

        ######################## Value Head ##################  [NOT USED IN CERL]
        # Construct Hidden Layer 1 with
        self.vf1 = nn.Linear(state_dim, l1)
        self.vln1 = nn.LayerNorm(l1)

        # Hidden Layer 2
        self.vf2 = nn.Linear(l1, l2)
        self.vln2 = nn.LayerNorm(l2)

        # Out
        self.vout = nn.Linear(l2, 1)





    def forward(self, obs, action):
        """Method to forward propagate through the critic's graph

             Parameters:
                   input (tensor): states
                   input (tensor): actions

             Returns:
                   Q1 (tensor): Qval 1
                   Q2 (tensor): Qval 2
                   V (tensor): Value



         """

        #Concatenate observation+action as critic state
        state = torch.cat([obs, action], 1)

        ###### Q1 HEAD ####
        q1 = F.elu(self.q1f1(state))
        q1 = self.q1ln1(q1)
        q1 = F.elu(self.q1f2(q1))
        q1 = self.q1ln2(q1)
        q1 = self.q1out(q1)

        ###### Q2 HEAD ####
        q2 = F.elu(self.q2f1(state))
        q2 = self.q2ln1(q2)
        q2 = F.elu(self.q2f2(q2))
        q2 = self.q2ln2(q2)
        q2 = self.q2out(q2)

        ###### Value HEAD ####
        v = F.elu(self.vf1(obs))
        v = self.vln1(v)
        v = F.elu(self.vf2(v))
        v = self.vln2(v)
        v = self.vout(v)


        return q1, q2, v



# Initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)

