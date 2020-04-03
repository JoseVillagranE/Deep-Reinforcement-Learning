# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:54:50 2020

@author: joser
"""

import random
from collections import deque


class ExperienceReplayMemory:
    
    pass

class SequentialDequeMemory(ExperienceReplayMemory):
    
    def __init__(self, queue_capacity=2000):
        
        self.queue_capacity = queue_capacity
        self.memory = deque(maxlen=self.queue_capacity)
        
    
    def add_to_memory(self, experience_tuple):
        self.memory.append(experience_tuple)
        
    def get_random_batch_for_replay(self, batch_size=64):
        return random.sample(self.memory, batch_size)
    
    def get_memory_size(self):
        return len(self.memory)