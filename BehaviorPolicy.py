# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:38:20 2020

@author: joser
"""

import numpy as np


class BehaviorPolicy:
    
    def __init__(self, n_actions, policy_type="epsilon_greedy", policy_parameters = {"epsilon":0.1}):
        
        self.policy = policy_type
        self.n_actions = n_actions
        self.policy_type = policy_type
        self.policy_parameters = policy_parameters
        
        self.epsilon = self.policy_parameters["epsilon"]
        self.min_epsilon = None
        self.epsilon_decay_rate = None
        
        
    def getPolicy(self):
        
        if self.policy_type=="epsilon_greedy":
            return self.return_epsilon_greedy_policy()
        elif self.policy_type=="epsilon_decay":
            self.epsilon = self.policy_parameters["epsilon"]
            self.min_epsilon = self.policy_parameters["min_epsilon"]
            self.epsilon_decay_rate = self.policy_parameters["epsilon_decay_rate"]
            return self.return_epsilon_decay_policy()
        
        
    
    def return_epsilon_decay_policy(self):
        
        def choose_action_by_epsilon_decay(values_of_all_possible_actions):
            prob_taking_best_action_only = 1 - self.epsilon
            prob_taking_any_random_action  =self.epsilon/self.n_actions
            action_prob_vector = [prob_taking_any_random_action]*self.n_actions
            exploitation_action_index = np.argmax(values_of_all_possible_actions)
            action_prob_vector[exploitation_action_index] += prob_taking_best_action_only
            chosen_action = np.random.choice(np.arange(self.n_actions), p=action_prob_vector)
            
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay_rate
            return chosen_action
        return choose_action_by_epsilon_decay
    
    def return_epsilon_greedy_policy(self):
        
        def choose_action_by_epsilon_greedy(values_of_all_possible_actions):
            prob_taking_best_action_only = 1 - self.epsilon
            prob_taking_any_random_action  =self.epsilon/self.n_actions
            action_prob_vector = [prob_taking_any_random_action]*self.n_actions
            exploitaition_action_index = np.argmax(values_of_all_possible_actions)
            action_prob_vector[exploitaition_action_index] += prob_taking_best_action_only
            chosen_action = np.random.choice(np.range(self.n_actions), p=action_prob_vector)
            return chosen_action
        return choose_action_by_epsilon_greedy
    

            