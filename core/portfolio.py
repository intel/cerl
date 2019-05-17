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

from core.learner import Learner


def initialize_portfolio(portfolio, args, genealogy, portfolio_id):
	"""Portfolio of learners

        Parameters:
            portfolio (list): Incoming list
            args (object): param class

        Returns:
            portfolio (list): Portfolio of learners
    """


	if portfolio_id == 10:
		td3args = {'policy_noise': 0.2, 'policy_noise_clip': 0.5, 'policy_ups_freq': 2, 'action_low': args.action_low, 'action_high': args.action_high}

		# Learner 1
		wwid = genealogy.new_id('learner_1')
		portfolio.append(
			Learner(wwid, 'TD3', args.state_dim, args.action_dim, actor_lr=1e-3, critic_lr=1e-3, gamma=0.9, tau=5e-3,
			        init_w=True, **td3args))

		# Learner 3
		wwid = genealogy.new_id('learner_3')
		portfolio.append(
			Learner(wwid, 'TD3', args.state_dim, args.action_dim, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=5e-3,
			        init_w=True, **td3args))

		# Learner 4
		wwid = genealogy.new_id('learner_4')
		portfolio.append(
			Learner(wwid, 'TD3', args.state_dim, args.action_dim, actor_lr=1e-3, critic_lr=1e-3, gamma=0.997, tau=5e-3,
			        init_w=True, **td3args))

		# Learner 4
		wwid = genealogy.new_id('learner_4')
		portfolio.append(
			Learner(wwid, 'TD3', args.state_dim, args.action_dim, actor_lr=1e-3, critic_lr=1e-3, gamma=0.9995, tau=5e-3,
			        init_w=True, **td3args))

	if portfolio_id == 11:
		td3args = {'policy_noise': 0.2, 'policy_noise_clip': 0.5, 'policy_ups_freq': 2, 'action_low': args.action_low, 'action_high': args.action_high}

		# Learner 1
		wwid = genealogy.new_id('learner_1')
		portfolio.append(
			Learner(wwid, 'TD3', args.state_dim, args.action_dim, actor_lr=1e-3, critic_lr=1e-3, gamma=0.9, tau=5e-3,
			        init_w=True, **td3args))

	if portfolio_id == 12:
		td3args = {'policy_noise': 0.2, 'policy_noise_clip': 0.5, 'policy_ups_freq': 2, 'action_low': args.action_low, 'action_high': args.action_high}

		# Learner 1
		wwid = genealogy.new_id('learner_1')
		portfolio.append(
			Learner(wwid, 'TD3', args.state_dim, args.action_dim, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=5e-3,
			        init_w=True, **td3args))

	if portfolio_id == 13:
		td3args = {'policy_noise': 0.2, 'policy_noise_clip': 0.5, 'policy_ups_freq': 2, 'action_low': args.action_low, 'action_high': args.action_high}

		# Learner 1
		wwid = genealogy.new_id('learner_1')
		portfolio.append(
			Learner(wwid, 'TD3', args.state_dim, args.action_dim, actor_lr=1e-3, critic_lr=1e-3, gamma=0.997, tau=5e-3,
			        init_w=True, **td3args))

	if portfolio_id == 14:
		td3args = {'policy_noise': 0.2, 'policy_noise_clip': 0.5, 'policy_ups_freq': 2, 'action_low': args.action_low, 'action_high': args.action_high}

		# Learner 1
		wwid = genealogy.new_id('learner_1')
		portfolio.append(
			Learner(wwid, 'TD3', args.state_dim, args.action_dim, actor_lr=1e-3, critic_lr=1e-3, gamma=0.9995, tau=5e-3,
			        init_w=True, **td3args))



	##############MOTIVATING EXAMPLE #######
	if portfolio_id == 100:

		td3args = {'policy_noise': 0.2, 'policy_noise_clip': 0.5, 'policy_ups_freq': 2, 'action_low': args.action_low, 'action_high': args.action_high}



		# Learner 1
		wwid = genealogy.new_id('learner_1')
		portfolio.append(
			Learner(wwid, 'TD3', args.state_dim, args.action_dim, actor_lr=1e-3, critic_lr=1e-3, gamma=0.0, tau=5e-3,
			        init_w=True, **td3args))

		# Learner 2
		wwid = genealogy.new_id('learner_2')
		portfolio.append(
			Learner(wwid, 'TD3', args.state_dim, args.action_dim, actor_lr=1e-3, critic_lr=1e-3, gamma=1.0, tau=5e-3, init_w=True,
			        **td3args))

	if portfolio_id == 101:
		td3args = {'policy_noise': 0.2, 'policy_noise_clip': 0.5, 'policy_ups_freq': 2, 'action_low': args.action_low, 'action_high': args.action_high}


		# Learner 3
		wwid = genealogy.new_id('learner_3')
		portfolio.append(
			Learner(wwid, 'TD3', args.state_dim, args.action_dim, actor_lr=1e-3, critic_lr=1e-3, gamma=0.0, tau=5e-3,
			        init_w=True, **td3args))

	if portfolio_id == 102:
		td3args = {'policy_noise': 0.2, 'policy_noise_clip': 0.5, 'policy_ups_freq': 2, 'action_low': args.action_low, 'action_high': args.action_high}


		# Learner 1
		wwid = genealogy.new_id('learner_1')
		portfolio.append(
			Learner(wwid, 'TD3', args.state_dim, args.action_dim, actor_lr=1e-3, critic_lr=1e-3, gamma=1.0, tau=5e-3,
			        init_w=True, **td3args))



	return portfolio
