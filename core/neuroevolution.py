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

import random
import numpy as np
import math
import core.mod_utils as utils



class SSNE:
	"""Neuroevolution object that contains all then method to run SUb-structure based Neuroevolution (SSNE)

		Parameters:
			  args (object): parameter class


	"""

	def __init__(self, args):
		self.gen = 0
		self.args = args;
		self.population_size = self.args.pop_size;
		#RL TRACKERS
		self.rl_sync_pool = []; self.all_offs = []; self.rl_res = {"elites":0.0, 'selects': 0.0, 'discarded':0.0}; self.num_rl_syncs = 0.0001
		self.lineage = [0.0 for _ in range(self.population_size)]; self.lineage_depth = 10

	def selection_tournament(self, index_rank, num_offsprings, tournament_size):
		"""Conduct tournament selection

			Parameters:
				  index_rank (list): Ranking encoded as net_indexes
				  num_offsprings (int): Number of offsprings to generate
				  tournament_size (int): Size of tournament

			Returns:
				  offsprings (list): List of offsprings returned as a list of net indices

		"""


		total_choices = len(index_rank)
		offsprings = []
		for i in range(num_offsprings):
			winner = np.min(np.random.randint(total_choices, size=tournament_size))
			offsprings.append(index_rank[winner])

		offsprings = list(set(offsprings))  # Find unique offsprings
		if len(offsprings) % 2 != 0:  # Number of offsprings should be even
			offsprings.append(index_rank[winner])
		return offsprings

	def list_argsort(self, seq):
		"""Sort the list

			Parameters:
				  seq (list): list

			Returns:
				  sorted list

		"""
		return sorted(range(len(seq)), key=seq.__getitem__)

	def regularize_weight(self, weight, mag):
		"""Clamps on the weight magnitude (reguralizer)

			Parameters:
				  weight (float): weight
				  mag (float): max/min value for weight

			Returns:
				  weight (float): clamped weight

		"""
		if weight > mag: weight = mag
		if weight < -mag: weight = -mag
		return weight

	def crossover_inplace(self, gene1, gene2):
		"""Conduct one point crossover in place

			Parameters:
				  gene1 (object): A pytorch model
				  gene2 (object): A pytorch model

			Returns:
				None

		"""


		keys1 =  list(gene1.state_dict())
		keys2 = list(gene2.state_dict())

		for key in keys1:
			if key not in keys2: continue

			# References to the variable tensors
			W1 = gene1.state_dict()[key]
			W2 = gene2.state_dict()[key]

			if len(W1.shape) == 2: #Weights no bias
				num_variables = W1.shape[0]
				# Crossover opertation [Indexed by row]
				try: num_cross_overs = random.randint(0, int(num_variables * 0.3))  # Number of Cross overs
				except: num_cross_overs = 1
				for i in range(num_cross_overs):
					receiver_choice = random.random()  # Choose which gene to receive the perturbation
					if receiver_choice < 0.5:
						ind_cr = random.randint(0, W1.shape[0]-1)  #
						W1[ind_cr, :] = W2[ind_cr, :]
					else:
						ind_cr = random.randint(0, W1.shape[0]-1)  #
						W2[ind_cr, :] = W1[ind_cr, :]

			elif len(W1.shape) == 1: #Bias or LayerNorm
				if random.random() <0.8: continue #Crossover here with low frequency
				num_variables = W1.shape[0]
				# Crossover opertation [Indexed by row]
				#num_cross_overs = random.randint(0, int(num_variables * 0.05))  # Crossover number
				for i in range(1):
					receiver_choice = random.random()  # Choose which gene to receive the perturbation
					if receiver_choice < 0.5:
						ind_cr = random.randint(0, W1.shape[0]-1)  #
						W1[ind_cr] = W2[ind_cr]
					else:
						ind_cr = random.randint(0, W1.shape[0]-1)  #
						W2[ind_cr] = W1[ind_cr]

	def mutate_inplace(self, gene):
		"""Conduct mutation in place

			Parameters:
				  gene (object): A pytorch model

			Returns:
				None

		"""
		mut_strength = 0.1
		num_mutation_frac = 0.05
		super_mut_strength = 10
		super_mut_prob = 0.05
		reset_prob = super_mut_prob + 0.02

		num_params = len(list(gene.parameters()))
		ssne_probabilities = np.random.uniform(0, 1, num_params) * 2

		for i, param in enumerate(gene.parameters()):  # Mutate each param

			# References to the variable keys
			W = param.data
			if len(W.shape) == 2:  # Weights, no bias

				num_weights = W.shape[0] * W.shape[1]
				ssne_prob = ssne_probabilities[i]

				if random.random() < ssne_prob:
					num_mutations = random.randint(0,
						int(math.ceil(num_mutation_frac * num_weights)))  # Number of mutation instances
					for _ in range(num_mutations):
						ind_dim1 = random.randint(0, W.shape[0]-1)
						ind_dim2 = random.randint(0, W.shape[-1]-1)
						random_num = random.random()

						if random_num < super_mut_prob:  # Super Mutation probability
							W[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength * W[ind_dim1, ind_dim2])
						elif random_num < reset_prob:  # Reset probability
							W[ind_dim1, ind_dim2] = random.gauss(0, 0.1)
						else:  # mutauion even normal
							W[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * W[ind_dim1, ind_dim2])

						# Regularization hard limit
						W[ind_dim1, ind_dim2] = self.regularize_weight(W[ind_dim1, ind_dim2],
																	   self.args.weight_magnitude_limit)

			elif len(W.shape) == 1:  # Bias or layernorm
				num_weights = W.shape[0]
				ssne_prob = ssne_probabilities[i]*0.04 #Low probability of mutation here

				if random.random() < ssne_prob:
					num_mutations = random.randint(0,
						int(math.ceil(num_mutation_frac * num_weights)))  # Number of mutation instances
					for _ in range(num_mutations):
						ind_dim = random.randint(0, W.shape[0]-1)
						random_num = random.random()

						if random_num < super_mut_prob:  # Super Mutation probability
							W[ind_dim] += random.gauss(0, super_mut_strength * W[ind_dim])
						elif random_num < reset_prob:  # Reset probability
							W[ind_dim] = random.gauss(0, 1)
						else:  # mutauion even normal
							W[ind_dim] += random.gauss(0, mut_strength * W[ind_dim])

						# Regularization hard limit
						W[ind_dim] = self.regularize_weight(W[ind_dim], self.args.weight_magnitude_limit)



	def reset_genome(self, gene):
		"""Reset a model's weights in place

			Parameters:
				  gene (object): A pytorch model

			Returns:
				None

		"""
		for param in (gene.parameters()):
			param.data.copy_(param.data)

	def epoch(self, gen, genealogy, pop, net_inds, fitness_evals, migration):
		"""Method to implement a round of selection and mutation operation

			Parameters:
				  pop (shared_list): Population of models
				  net_inds (list): Indices of individuals evaluated this generation
				  fitness_evals (list): Fitness values for evaluated individuals
				  **migration (object): Policies from learners to be synced into population

			Returns:
				None

		"""

		self.gen+= 1; num_elitists = int(self.args.elite_fraction * len(fitness_evals))
		if num_elitists < 2: num_elitists = 2


		# Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
		index_rank = self.list_argsort(fitness_evals); index_rank.reverse()
		elitist_index = index_rank[:num_elitists]  # Elitist indexes safeguard

		# Selection step
		offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - len(elitist_index) - len(migration), tournament_size=3)

		#Transcripe ranked indexes from now on to refer to net indexes
		elitist_index = [net_inds[i] for i in elitist_index]
		offsprings = [net_inds[i] for i in offsprings]

		#Figure out unselected candidates
		unselects = []; new_elitists = []
		for net_i in net_inds:
			if net_i in offsprings or net_i in elitist_index:
				continue
			else:
				unselects.append(net_i)
		random.shuffle(unselects)

		#Inheritance step (sync learners to population)
		for policy in migration:
			replacee = unselects.pop(0)
			utils.hard_update(target=pop[replacee], source=policy)
			wwid = genealogy.asexual(int(policy.wwid.item()))
			pop[replacee].wwid[0] = wwid

		# Elitism step, assigning elite candidates to some unselects
		for i in elitist_index:
			try: replacee = unselects.pop(0)
			except: replacee = offsprings.pop(0)
			new_elitists.append(replacee)
			utils.hard_update(target=pop[replacee], source=pop[i])
			wwid = genealogy.asexual(int(pop[i].wwid.item()))
			pop[replacee].wwid[0] = wwid
			genealogy.elite(wwid, gen)

			#self.lineage[replacee] = self.lineage[i]

		# Crossover for unselected genes with 100 percent probability
		if len(unselects) % 2 != 0:  # Number of unselects left should be even
			unselects.append(unselects[random.randint(0, len(unselects)-1)])
		for i, j in zip(unselects[0::2], unselects[1::2]):
			off_i = random.choice(new_elitists);
			off_j = random.choice(offsprings)
			utils.hard_update(target=pop[i], source=pop[off_i])
			utils.hard_update(target=pop[j], source=pop[off_j])
			self.crossover_inplace(pop[i], pop[j])
			wwid1 = genealogy.crossover(int(pop[off_i].wwid.item()), int(pop[off_j].wwid.item()), gen)
			wwid2 = genealogy.crossover(int(pop[off_i].wwid.item()), int(pop[off_j].wwid.item()), gen)
			pop[i].wwid[0] = wwid1; pop[j].wwid[0] = wwid2

			#self.lineage[i] = (self.lineage[off_i]+self.lineage[off_j])/2
			#self.lineage[j] = (self.lineage[off_i] + self.lineage[off_j]) / 2

		# Crossover for selected offsprings
		for i, j in zip(offsprings[0::2], offsprings[1::2]):
			if random.random() < self.args.crossover_prob:
				self.crossover_inplace(pop[i], pop[j])
				wwid1 = genealogy.crossover(int(pop[i].wwid.item()), int(pop[j].wwid.item()), gen)
				wwid2 = genealogy.crossover(int(pop[i].wwid.item()), int(pop[j].wwid.item()), gen)
				pop[i].wwid[0] = wwid1; pop[j].wwid[0] = wwid2


		# Mutate all genes in the population except the new elitists
		for net_i in net_inds:
			if net_i not in new_elitists:  # Spare the new elitists
				if random.random() < self.args.mutation_prob:
					self.mutate_inplace(pop[net_i])
					genealogy.mutation(int(pop[net_i].wwid.item()), gen)


		self.all_offs[:] = offsprings[:]








