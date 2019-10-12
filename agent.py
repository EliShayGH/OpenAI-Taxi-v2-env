import numpy as np
import random 
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.11
        self.gamma = 0.9
        self.epsilon = 0.01

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        if random.random() < self.epsilon:
            next_action = np.random.choice(self.nA)
        else:
            next_action = np.argmax(self.Q[state])   
                                            
        return next_action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        next_action = self.select_action(state)
        self.Q[state][action] += self.alpha*(reward + self.gamma*self.Q[next_state][next_action]- self.Q[state][action])
        
        if done:
           self.epsilon /= 2.0
            
