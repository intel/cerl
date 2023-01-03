# DISCONTINUATION OF PROJECT #
This project will no longer be maintained by Intel.
Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.
Intel no longer accepts patches to this project.
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

Codebase for [Collaborative Evolutionary Reinforcement Learning](https://arxiv.org/pdf/1905.00976.pdf) accepted to be published in the Proceedings of the 36th International Conference on Machine Learning, Long Beach, California, PMLR 97, 2019. Copyright 2019 by the author(s).

## Guide to set up and run CERL Experiments


1. Setup Conda
    - Install Anaconda3
    - conda create -n $ENV_NAME$ python=3.6.1
    - source activate $ENV_NAME$

2. Install Pytorch version 1.0
    - Refer to https://pytorch.org/ for instructions
    - conda install pytorch torchvision -c pytorch [GPU-version]

3. Install Numpy, Cython and Scipy
    - pip install numpy==1.15.4
    - pip install cython==0.29.2
    - pip install scipy==1.1.0
    
4. Install Mujoco and OpenAI_Gym
    - Download mjpro150 from https://www.roboti.us/index.html
    - Unzip mjpro150 and place it + mjkey.txt (license file) in ~/.mujoco/ (create the .mujoco dir in you home folder)
    - pip install -U 'mujoco-py<1.50.2,>=1.50.1'
    - pip install 'gym[all]'
    
## Code labels

main.py: Main Script runs everything

core/runner.py: Rollout worker

core/ucb.py: Upper Confidence Bound implemented for learner selection by the resource-manager

core/portfolio.py: Portfolio of learners which can vary in their hyperparameters

core/learner.py: Learner agent encapsulating the algo and sum-statistics

core/buffer.py: Cyclic Replay buffer

core/env_wrapper.py: Wrapper around the Mujoco env

core/models.py: Actor/Critic model

core/neuroevolution.py: Implements Neuroevolution

core/off_policy_algo.py: Implements the off_policy_gradient learner TD3

core/mod_utils.py: Helper functions

## Reproduce Results

python main.py -env HalfCheetah-v2 -portfolio {10,14} -total_steps 2 -seed {2018,2022}

python main.py -env Hopper-v2 -portfolio {10,14} -total_steps 1.5 -seed {2018,2022}

python main.py -env Humanoid-v2 -portfolio {10,14} -total_steps 1 -seed {2018,2022}

python main.py -env Walker2d-v2 -portfolio {10,14} -total_steps 2 -seed {2018,2022}

python main.py -env Swimmer-v2 -portfolio {10,14} -total_steps 2 -seed {2018,2022}

python main.py -env Hopper-v2 -portfolio {100,102} -total_steps 5 -seed {2018,2022}

where {} represents an inclusive discrete range: {10, 14} --> {10, 11, 12, 13, 14}


## Note
All roll-outs (evaluation of actors in the evolutionary population and the explorative roll-outs 
conducted by the learners run in parallel). They are farmed out to different CPU cores, 
and write asynchronously to the collective replay buffer. Thus, slight variations in results 
are observed even with the same seed. 
