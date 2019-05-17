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

import gym


class EnvironmentWrapper:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, env_name, ALGO):
		"""
		A base template for all environment wrappers.
		"""
		self.env = gym.make(env_name)
		self.action_low = float(self.env.action_space.low[0])
		self.action_high = float(self.env.action_space.high[0])
		self.ALGO = ALGO




	def reset(self):
		"""Method overloads reset
			Parameters:
				None

			Returns:
				next_obs (list): Next state
		"""
		return self.env.reset()


	def step(self, action: object): #Expects a numpy action
		"""Take an action to forward the simulation

			Parameters:
				action (ndarray): action to take in the env

			Returns:
				next_obs (list): Next state
				reward (float): Reward for this step
				done (bool): Simulation done?
				info (None): Template from OpenAi gym (doesnt have anything)
		"""

		action = self.action_low + action * (self.action_high - self.action_low)
		return self.env.step(action)

	def render(self):
		self.env.render()



