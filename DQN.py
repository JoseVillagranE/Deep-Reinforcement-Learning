# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:20:56 2020

@author: joser
"""

import numpy as np
from itertools import count
import matplotlib.pyplot as plt
import time
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import mean_squared_error

import gym

from ExperienceReplayMemory import SequentialDequeMemory
from BehaviorPolicy import BehaviorPolicy



class DoubleDQN:
    
    def __init__(self, agent_name=None, env=gym.make('CartPole-v1'), n_episodes=500,
                 discounting_factor=0.9, learning_rate=0.001, behavior_policy="epsilon_decay", 
                 policy_parameters = {"epsilon":1.0, "min_epsilon":0.001, "epsilon_decay_rate":0.99},
                 deep_learning_model_hidden_layer_configuration = [32, 32]):
        
        self.agent_name = "ddqa_"+str(time.strftime("%Y%m%d-%H%M%S")) if agent_name is None else agent_name
        self.model_weights_dir = "model_weights"
        self.env = env
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.n_episodes = n_episodes
        self.episodes_completed = 0
        self.gamma = discounting_factor
        self.alpha = learning_rate
        self.policy = BehaviorPolicy(n_actions=self.n_actions, policy_type=behavior_policy, 
                                     policy_parameters=policy_parameters).getPolicy()
        self.policyParameter = policy_parameters
        self.model_hidden_layer_configuration = deep_learning_model_hidden_layer_configuration
        self.online_model = self._build_sequential_dnn_model()
        self.target_model = self._build_sequential_dnn_model()
        self.trainingStats_steps_in_each_episode = []
        self.trainingStats_rewards_in_each_episode = []
        self.trainingStats_discountedRewards_in_each_episode = []
        self.memory = SequentialDequeMemory(queue_capacity=20000)
        self.experience_replay_batch_size = 64
        
        
    def _build_sequential_dnn_model(self):
        
        model = Sequential()
        hidden_layers = self.model_hidden_layer_configuration
        model.add(Dense(hidden_layers[0], input_dim=self.n_states, activation="relu"))
        for layer_size in hidden_layers[1:]:
            model.add(Dense(layer_size, activation="relu"))
        model.add(Dense(self.n_actions, activation="linear"))
        model.compile(loss=mean_squared_error, optimizer=Adam(lr=self.alpha))
        return model
    
    def _sync_target_model_with_online_model(self):
        
        self.target_model.set_weights(self.online_model.get_weights())
        
    def _update_online_model(self, experience_tuple):
        
        # Update Qnetwork
        current_state, action, instantaneous_reward, next_state, done_flag = experience_tuple
        action_target_values = self.online_model.predict(current_state)
        action_values_for_state = action_target_values[0]
        if done_flag:
            action_values_for_state[action] = instantaneous_reward
        else:
            action_values_for_next_state = self.target_model.predict(next_state)[0]
            max_next_state_value = np.max(action_values_for_next_state)
            target_action_value = instantaneous_reward + self.gamma*max_next_state_value
            action_values_for_state[action] = target_action_value
            action_target_values[0] = action_values_for_state
            
            self.online_model.fit(current_state, action_target_values, epochs=1, verbose=0)
            
    
    def _reshape_state_for_model(self, state):
        
        return np.reshape(state, [1, self.n_states])
    
    def train_agent(self):
        
        
        self.load_model_weights()
        for episode in range(self.n_episodes):
            current_state = self._reshape_state_for_model(self.env.reset())
            cumulative_reward = 0
            discounted_cumulative_reward = 0
            for n_step in count():
                all_action_value_for_current_state = self.online_model.predict(current_state)[0]
                policy_defined_action = self.policy(all_action_value_for_current_state)
                next_state, instantaneous_reward, done, _ = self.env.step(policy_defined_action)
                next_state = self._reshape_state_for_model(next_state)
                experience_tuple = (current_state, policy_defined_action, instantaneous_reward, next_state, done)
                self.memory.add_to_memory(experience_tuple)
                cumulative_reward += instantaneous_reward
                discounted_cumulative_reward = instantaneous_reward + self.gamma*discounted_cumulative_reward
                if done:
                    self.trainingStats_steps_in_each_episode.append(n_step)
                    self.trainingStats_rewards_in_each_episode.append(cumulative_reward)
                    self.trainingStats_discountedRewards_in_each_episode.append(discounted_cumulative_reward)
                    self._sync_target_model_with_online_model()
                    
                    print("Episode: {}/{}, reward: {}, discounted_reward: {}".format(n_step, 
                          self.n_episodes, cumulative_reward, discounted_cumulative_reward))
                    break
                
                self.replay_experience_from_memory()
            #if episode % 2 == 0: self.plot_training_stats()
            if episode % 5 == 0: self.save_model_weights()
        
        return self.trainingStats_steps_in_each_episode, self.trainingStats_rewards_in_each_episode, \
    self.trainingStats_discountedRewards_in_each_episode
    
    
    def replay_experience_from_memory(self):
        
        if self.memory.get_memory_size() < self.experience_replay_batch_size:
            return False
        
        experience_mini_batch = self.memory.get_random_batch_for_replay(batch_size=self.experience_replay_batch_size)
        for experience_tuple in experience_mini_batch:
            self._update_online_model(experience_tuple)
        
        return True
    
    def save_model_weights(self, agent_name=None):
        
        if agent_name is None:
            agent_name = self.agent_name
            
        model_file = os.path.join(os.path.join(self.model_weights_dir, agent_name+".h5"))
        self.online_model.save_weights(model_file, overwrite=True)
        
    def load_model_weights(self, agent_name=None):
        
        if agent_name is None:
            agent_name = self.agent_name
        
        model_file = os.path.join(os.path.join(self.model_weights_dir, agent_name+".h5"))
        if os.path.exists(model_file):
            self.online_model.load_weights(model_file)
            self.target_model.load_weights(model_file)
            
    
    def plot_training_stats(self, training_stats=None):
        
        # training_stats(tuple) -> steps, list of rewards, list of cumulative rewards for each episode
        
        steps = self.trainingStats_steps_in_each_episode if training_stats is None else training_stats[0]
        rewards = self.trainingStats_rewards_in_each_episode if training_stats is None else training_stats[1]
        discount_rewards = self.trainingStats_discountedRewards_in_each_episode if training_stats is None else training_stats[2]
        episode = np.arange(len(self.trainingStats_steps_in_each_episode))
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Episodes (e)")
        ax1.set_ylabel("Steps to episode completion", color="red")
        
        ax1.plot(episode, steps, color="red")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Reward in each Episode", color="blue")
        ax2.plot(episode, rewards, color="blue")
        fig.tight_layout()
        plt.show()
        
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Episodes (e)")
        ax1.set_ylabel("Steps to episode completion", color="red")
        
        ax1.plot(episode, steps, color="red")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Discounted Reward in each Episode", color="green")
        ax2.plot(episode, discount_rewards, color="green")
        fig.tight_layout()
        plt.show()
        

if __name__=="__main__":
    agent = DoubleDQN()
    training_stats = agent.train_agent()
    agent.plot_training_stats(training_stats)
    
            
            
        
        