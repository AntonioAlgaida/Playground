# -*- coding: utf-8 -*-
"""
Created on Fri May  6 10:59:43 2022

@author: antonio.guillen@edu.upct.es
"""
"""
In this notebook, I will test Stable-Baselines3 to train moon lander agent
to **land correctly on the Moon with different configurations
"""
#%%
import gym

from huggingface_sb3 import load_from_hub, package_to_hub, push_to_hub
from huggingface_hub import notebook_login

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
#%%
env = make_vec_env('LunarLander-v2', n_envs=16)

model = PPO(policy = 'MlpPolicy',
            env = env,
            n_steps = 1024,
            batch_size = 64,
            n_epochs = 4,
            gamma = 0.999,
            gae_lambda = 0.98,
            ent_coef = 0.01,
            verbose=1)

#%%
model.learn(total_timesteps=int(2e6))

#%%
# Evaluate the agent
# Create a new environment for evaluation
eval_env = gym.make("LunarLander-v2")

# Evaluate the model with 10 evaluation episodes and deterministic=True
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

# Print the results
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
