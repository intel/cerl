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

import math, random


def ucb(allocation_size, portfolio, c):
	"""Upper Confidence Bound implementation to pick learners

        Parameters:
            allocation_size (int): Size of allocation (num of resources)
			portfolio (list): List of learners
			c (float): Exploration coefficient in UCB

        Returns:
            allocation (list): List of learner ids formulating the resource allocation
	"""


	values = [learner.value for learner in portfolio]
	#Normalize values
	values = [val - min(values)  for val in values]
	values = [val/(sum(values)+0.1) for val in values]

	visit_counts = [learner.visit_count for learner in portfolio]
	total_visit = sum(visit_counts)

	######## Implement UCB ########
	ucb_scores = [(values[i]) + c * math.sqrt( math.log(total_visit)/visit_counts[i]) for i in range(len(portfolio))]


	########## Use UCB scores to perform probabilistic resource allocation (different from making one choice) ##########
	allocation = roulette_wheel(ucb_scores, allocation_size)




	return allocation



def roulette_wheel(probs, num_samples):
	"""Roulette_wheel selection from a prob. distribution

        Parameters:
            probs (list): Probability distribution
			num_samples (int): Num of iterations to sample from distribution

        Returns:
            out (list): List of samples based on incoming distribution
	"""

	#Normalize
	probs = [prob - min(probs) + abs(min(probs)) for prob in probs] #Biased translation (to positive axis) to ensure the lowest does not end up with a probability of zero

	####### HACK FOR ROLLOUT_SIZE = 1 #####
	if sum(probs) != 0:
		probs = [prob / sum(probs) for prob in probs]
	else:
		probs = [1.0 for _ in probs]
	####### END HACK #####


	#Selection
	out = []
	for _ in range(num_samples):
		rand = random.random()

		for i in range(len(probs)):
			if rand < sum(probs[0:i+1]):
				out.append(i)
				break

	print('UCB_prob_mass', ["%.2f" %i for i in probs])
	print('Allocation', out)
	print()

	return out
