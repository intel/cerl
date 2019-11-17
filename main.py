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

import numpy as np, os, time, random, torch, sys
from core.neuroevolution import SSNE
from core.models import Actor
from core import mod_utils as utils
from core.mod_utils import str2bool
from core.ucb import ucb
from core.runner import rollout_worker
from core.portfolio import initialize_portfolio
from torch.multiprocessing import Process, Pipe, Manager
import threading
from core.buffer import Buffer
from core.genealogy import Genealogy
import gym
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-pop_size', type=int, help='#Policies in the population',  default=10)
parser.add_argument('-seed', type=int, help='Seed',  default=2018)
parser.add_argument('-rollout_size', type=int, help='#Policies in rolout size',  default=10)
parser.add_argument('-env', type=str, help='#Environment name',  default='Humanoid-v2')
parser.add_argument('-gradperstep', type=float, help='#Gradient step per env step',  default=1.0)
parser.add_argument('-savetag', type=str, help='#Tag to append to savefile',  default='')
parser.add_argument('-gpu_id', type=int, help='#GPU ID ',  default=0)
parser.add_argument('-buffer_gpu', type=str2bool, help='#Store buffer in GPU?',  default=0)
parser.add_argument('-portfolio', type=int, help='Portfolio ID',  default=10)
parser.add_argument('-total_steps', type=float, help='#Total steps in the env in millions ',  default=2)
parser.add_argument('-batchsize', type=int, help='Seed',  default=256)
parser.add_argument('-noise', type=float, help='Noise STD',  default=0.01)


POP_SIZE = vars(parser.parse_args())['pop_size']
BATCHSIZE = vars(parser.parse_args())['batchsize']
ROLLOUT_SIZE = vars(parser.parse_args())['rollout_size']
ENV_NAME = vars(parser.parse_args())['env']
GRADPERSTEP = vars(parser.parse_args())['gradperstep']
SAVETAG = vars(parser.parse_args())['savetag']
BUFFER_GPU = vars(parser.parse_args())['buffer_gpu']
SEED = vars(parser.parse_args())['seed']
GPU_DEVICE = vars(parser.parse_args())['gpu_id']
PORTFOLIO_ID = vars(parser.parse_args())['portfolio']
TOTAL_STEPS = int(vars(parser.parse_args())['total_steps'] * 1000000)
NOISE_STD = int(vars(parser.parse_args())['noise'])
os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_DEVICE)

#ICML EXPERIMENT
if PORTFOLIO_ID == 11 or PORTFOLIO_ID == 12 or PORTFOLIO_ID == 13 or PORTFOLIO_ID == 14 or PORTFOLIO_ID == 101 or PORTFOLIO_ID == 102: ISOLATE_PG = True
else:
    ISOLATE_PG = False
ALGO = "TD3"
SAVE = True
TEST_SIZE=10


class Parameters:
	def __init__(self):
		"""Parameter class stores all parameters for policy gradient

		Parameters:
			None

		Returns:
			None
		"""
		self.seed = SEED
		self.asynch_frac = 1.0 #Aynchronosity of NeuroEvolution
		self.algo = ALGO

		self.batch_size = BATCHSIZE #Batch size
		self.noise_std = NOISE_STD #Gaussian noise exploration std
		self.ucb_coefficient = 0.9 #Exploration coefficient in UCB
		self.gradperstep = GRADPERSTEP
		self.buffer_gpu = BUFFER_GPU
		self.rollout_size = ROLLOUT_SIZE #Size of learner rollouts

		#NeuroEvolution stuff
		self.pop_size = POP_SIZE
		self.elite_fraction = 0.2
		self.crossover_prob = 0.15
		self.mutation_prob = 0.90

		#######unused########
		self.extinction_prob = 0.005  # Probability of extinction event
		self.extinction_magnituide = 0.5  # Probabilty of extinction for each genome, given an extinction event
		self.weight_magnitude_limit = 10000000
		self.mut_distribution = 1  # 1-Gaussian, 2-Laplace, 3-Uniform


		#Save Results
		dummy_env = gym.make(ENV_NAME)
		self.state_dim = dummy_env.observation_space.shape[0]; self.action_dim = dummy_env.action_space.shape[0]
		self.action_low = float(dummy_env.action_space.low[0]); self.action_high = float(dummy_env.action_space.high[0])
		self.savefolder = 'Results/'
		if not os.path.exists('Results/'): os.makedirs('Results/')
		self.aux_folder = self.savefolder + 'Auxiliary/'
		if not os.path.exists(self.aux_folder): os.makedirs(self.aux_folder)


class CERL_Agent:
	"""Main CERL class containing all methods for CERL

		Parameters:
		args (int): Parameter class with all the parameters

	"""

	def __init__(self, args):
		self.args = args
		self.evolver = SSNE(self.args)

		#MP TOOLS
		self.manager = Manager()

		#Genealogy tool
		self.genealogy = Genealogy()

		#Initialize population
		self.pop = self.manager.list()
		for _ in range(args.pop_size):
			wwid = self.genealogy.new_id('evo')
			if ALGO == 'SAC': self.pop.append(GaussianPolicy(args.state_dim, args.action_dim, args.hidden_size, wwid))
			else: self.pop.append(Actor(args.state_dim, args.action_dim, wwid))

		if ALGO == "SAC": self.best_policy = GaussianPolicy(args.state_dim, args.action_dim, args.hidden_size, -1)
		else:
			self.best_policy = Actor(args.state_dim, args.action_dim, -1)


		#Turn off gradients and put in eval mod
		for actor in self.pop:
			actor = actor.cpu()
			actor.eval()

		#Init BUFFER
		self.replay_buffer = Buffer(1000000, self.args.buffer_gpu)

		#Intialize portfolio of learners
		self.portfolio = []
		self.portfolio = initialize_portfolio(self.portfolio, self.args, self.genealogy, PORTFOLIO_ID)
		self.rollout_bucket = self.manager.list()
		for _ in range(len(self.portfolio)):
			if ALGO == 'SAC': self.rollout_bucket.append(GaussianPolicy(args.state_dim, args.action_dim, args.hidden_size, -1))
			else: self.rollout_bucket.append(Actor(args.state_dim, args.action_dim, -1))



		# Initialize shared data bucket
		self.data_bucket = self.replay_buffer.tuples

		############## MULTIPROCESSING TOOLS ###################


		#Evolutionary population Rollout workers
		self.evo_task_pipes = [Pipe() for _ in range(args.pop_size)]
		self.evo_result_pipes = [Pipe() for _ in range(args.pop_size)]
		self.evo_workers = [Process(target=rollout_worker, args=(id, self.evo_task_pipes[id][1], self.evo_result_pipes[id][0], False, self.data_bucket, self.pop, ENV_NAME, None, ALGO)) for id in range(args.pop_size)]
		for worker in self.evo_workers: worker.start()
		self.evo_flag = [True for _ in range(args.pop_size)]

		#Learner rollout workers
		self.task_pipes = [Pipe() for _ in range(args.rollout_size)]
		self.result_pipes = [Pipe() for _ in range(args.rollout_size)]
		self.workers = [Process(target=rollout_worker, args=(id, self.task_pipes[id][1], self.result_pipes[id][0], True, self.data_bucket, self.rollout_bucket, ENV_NAME, args.noise_std, ALGO)) for id in range(args.rollout_size)]
		for worker in self.workers: worker.start()
		self.roll_flag = [True for _ in range(args.rollout_size)]

		#Test bucket
		self.test_bucket = self.manager.list()
		if ALGO == 'SAC':
			self.test_bucket.append(GaussianPolicy(args.state_dim, args.action_dim, args.hidden_size, -1))
		else:
			self.test_bucket.append(Actor(args.state_dim, args.action_dim, -1))

		#5 Test workers
		self.test_task_pipes = [Pipe() for _ in range(TEST_SIZE)]
		self.test_result_pipes = [Pipe() for _ in range(TEST_SIZE)]
		self.test_workers = [Process(target=rollout_worker, args=(id, self.test_task_pipes[id][1], self.test_result_pipes[id][0], False, None, self.test_bucket, ENV_NAME, args.noise_std, ALGO)) for id in range(TEST_SIZE)]
		for worker in self.test_workers: worker.start()
		self.test_flag = False

		#Meta-learning controller (Resource Distribution)
		self.allocation = [] #Allocation controls the resource allocation across learners
		for i in range(args.rollout_size): self.allocation.append(i % len(self.portfolio)) #Start uniformly (equal resources)
		#self.learner_stats = [{'fitnesses': [], 'ep_lens': [], 'value': 0.0, 'visit_count':0} for _ in range(len(self.portfolio))] #Track node statistsitic (each node is a learner), to compute UCB scores

		#Trackers
		self.best_score = 0.0; self.gen_frames = 0; self.total_frames = 0; self.best_shaped_score = None; self.test_score = None; self.test_std = None



	def train(self, gen, frame_tracker):
		"""Main training loop to do rollouts, neureoevolution, and policy gradients

			Parameters:
				gen (int): Current epoch of training

			Returns:
				None
		"""
		################ START ROLLOUTS ##############

		#Start Evolution rollouts
		if not ISOLATE_PG:
			for id, actor in enumerate(self.pop):
				if self.evo_flag[id]:
					self.evo_task_pipes[id][0].send(id)
					self.evo_flag[id] = False

		#Sync all learners actor to cpu (rollout) actor
		for i, learner in enumerate(self.portfolio):
			learner.algo.actor.cpu()
			utils.hard_update(self.rollout_bucket[i], learner.algo.actor)
			learner.algo.actor.cuda()

		# Start Learner rollouts
		for rollout_id, learner_id in enumerate(self.allocation):
			if self.roll_flag[rollout_id]:
				self.task_pipes[rollout_id][0].send(learner_id)
				self.roll_flag[rollout_id] = False

		#Start Test rollouts
		if gen % 5 == 0:
			self.test_flag = True
			for pipe in self.test_task_pipes: pipe[0].send(0)


		############# UPDATE PARAMS USING GRADIENT DESCENT ##########
		if self.replay_buffer.__len__() > self.args.batch_size * 10: ###BURN IN PERIOD
			self.replay_buffer.tensorify()  # Tensorify the buffer for fast sampling

			#Spin up threads for each learner
			threads = [threading.Thread(target=learner.update_parameters, args=(self.replay_buffer, self.args.buffer_gpu, self.args.batch_size, int(self.gen_frames * self.args.gradperstep))) for learner in
			           self.portfolio]

			# Start threads
			for thread in threads: thread.start()

			#Join threads
			for thread in threads: thread.join()
			self.gen_frames = 0


		########## SOFT -JOIN ROLLOUTS FOR EVO POPULATION ############
		if not ISOLATE_PG:
			all_fitness = []; all_net_ids = []; all_eplens = []
			while True:
				for i in range(self.args.pop_size):
					if self.evo_result_pipes[i][1].poll():
						entry = self.evo_result_pipes[i][1].recv()
						all_fitness.append(entry[1]); all_net_ids.append(entry[0]); all_eplens.append(entry[2]); self.gen_frames+= entry[2]; self.total_frames += entry[2]
						self.evo_flag[i] = True

				# Soft-join (50%)
				if len(all_fitness) / self.args.pop_size >= self.args.asynch_frac: break

		########## HARD -JOIN ROLLOUTS FOR LEARNER ROLLOUTS ############
		for i in range(self.args.rollout_size):
			entry = self.result_pipes[i][1].recv()
			learner_id = entry[0]; fitness = entry[1]; num_frames = entry[2]
			self.portfolio[learner_id].update_stats(fitness, num_frames)

			self.gen_frames += num_frames; self.total_frames += num_frames
			if fitness > self.best_score: self.best_score = fitness

			self.roll_flag[i] = True

		#Referesh buffer (housekeeping tasks - pruning to keep under capacity)
		self.replay_buffer.referesh()
		######################### END OF PARALLEL ROLLOUTS ################

		############ PROCESS MAX FITNESS #############
		if not ISOLATE_PG:
			champ_index = all_net_ids[all_fitness.index(max(all_fitness))]
			utils.hard_update(self.test_bucket[0], self.pop[champ_index])
			if max(all_fitness) > self.best_score:
				self.best_score = max(all_fitness)
				utils.hard_update(self.best_policy, self.pop[champ_index])
				if SAVE:
					torch.save(self.pop[champ_index].state_dict(), self.args.aux_folder + ENV_NAME+'_best'+SAVETAG)
					print("Best policy saved with score", '%.2f'%max(all_fitness))

		else: #Run PG in isolation
			utils.hard_update(self.test_bucket[0], self.rollout_bucket[0])

		###### TEST SCORE ######
		if self.test_flag:
			self.test_flag = False
			test_scores = []
			for pipe in self.test_result_pipes: #Collect all results
				entry = pipe[1].recv()
				test_scores.append(entry[1])
			test_scores = np.array(test_scores)
			test_mean = np.mean(test_scores); test_std = (np.std(test_scores))

			# Update score to trackers
			frame_tracker.update([test_mean], self.total_frames)
		else:
			test_mean, test_std = None, None


		#NeuroEvolution's probabilistic selection and recombination step
		if not ISOLATE_PG:
			if gen % 5 == 0:
				self.evolver.epoch(gen, self.genealogy, self.pop, all_net_ids, all_fitness, self.rollout_bucket)
			else:
				self.evolver.epoch(gen, self.genealogy, self.pop, all_net_ids, all_fitness, [])

		#META LEARNING - RESET ALLOCATION USING UCB
		if gen % 1 == 0:
			self.allocation = ucb(len(self.allocation), self.portfolio, self.args.ucb_coefficient)


		#Metrics
		if not ISOLATE_PG:
			champ_len = all_eplens[all_fitness.index(max(all_fitness))]
			champ_wwid = int(self.pop[champ_index].wwid.item())
			max_fit = max(all_fitness)
		else:
			champ_len = num_frames; champ_wwid = int(self.rollout_bucket[0].wwid.item())
			all_fitness = [fitness]; max_fit = fitness; all_eplens = [num_frames]

		return max_fit, champ_len, all_fitness, all_eplens, test_mean, test_std, champ_wwid

if __name__ == "__main__":
	args = Parameters()  # Create the Parameters class
	SAVETAG = SAVETAG + '_p' + str(PORTFOLIO_ID)
	SAVETAG = SAVETAG + '_s' + str(SEED)
	SAVETAg = SAVETAG + 'std' + str(NOISE_STD)

	frame_tracker = utils.Tracker(args.savefolder, ['score_'+ENV_NAME+SAVETAG], '.csv')  #Tracker class to log progress
	max_tracker = utils.Tracker(args.aux_folder, ['pop_max_score_'+ENV_NAME+SAVETAG], '.csv')  #Tracker class to log progress FOR MAX (NOT REPORTED)

	#Set seeds
	torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

	#INITIALIZE THE MAIN AGENT CLASS
	agent = CERL_Agent(args) #Initialize the agent
	print('Running CERL for', ENV_NAME, 'State_dim:', args.state_dim, ' Action_dim:', args.action_dim)

	time_start = time.time()
	for gen in range(1, 1000000000): #Infinite generations

		#Train one iteration
		best_score, test_len, all_fitness, all_eplen, test_mean, test_std, champ_wwid = agent.train(gen, frame_tracker)

		#PRINT PROGRESS
		print('Env', ENV_NAME, 'Gen', gen, 'Frames', agent.total_frames, ' Pop_max/max_ever:','%.2f'%best_score, '/','%.2f'%agent.best_score, ' Avg:','%.2f'%frame_tracker.all_tracker[0][1],
		      ' Frames/sec:','%.2f'%(agent.total_frames/(time.time()-time_start)),
			  ' Champ_len', '%.2f'%test_len, ' Test_score u/std', utils.pprint(test_mean), utils.pprint(test_std), 'savetag', SAVETAG, )

		# # PRINT MORE DETAILED STATS PERIODICALLY
		if gen % 5 == 0:
			print('Learner Fitness', [utils.pprint(learner.value) for learner in agent.portfolio], 'Sum_stats_resource_allocation', [learner.visit_count for learner in agent.portfolio])
			print('Pop/rollout size', args.pop_size,'/',args.rollout_size, 'gradperstep', args.gradperstep, 'Seed', SEED, 'Portfolio_id', PORTFOLIO_ID)
			try:
				print('Best Policy ever genealogy:', agent.genealogy.tree[int(agent.best_policy.wwid.item())].history)
				print('Champ genealogy:', agent.genealogy.tree[champ_wwid].history)
			except: None
			print()

		max_tracker.update([best_score], agent.total_frames)
		if agent.total_frames > TOTAL_STEPS:
			break

		#Save sum stats
		if PORTFOLIO_ID == 10 or PORTFOLIO_ID == 100:
			visit_counts = np.array([learner.visit_count for learner in agent.portfolio])
			np.savetxt(args.aux_folder + 'allocation_' + ENV_NAME + SAVETAG, visit_counts, fmt='%.3f', delimiter=',')

	###Kill all processes
	try:
		for p in agent.task_pipes: p[0].send('TERMINATE')
		for p in agent.test_task_pipes: p[0].send('TERMINATE')
		for p in agent.evo_task_pipes: p[0].send('TERMINATE')

	except: None


