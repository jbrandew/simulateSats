import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random
from collections import deque

import pdb 

#this is the most basic as possible implementation of discretized pendulum :)
env = gym.make("Pendulum-v1", render_mode="rgb_array")

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
#action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

#first, create class for our QNetwork
#this approximates the value of each state-action pair 
class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, q_lr):
        super(QNetwork, self).__init__()
        
        #create 2 fully connected layers
        #the input space is equal to the dimensionality of the 
        #state vector. 
        self.fc_1 = nn.Linear(state_dim, 64)
        self.fc_2 = nn.Linear(64, 32)
        #create a final output layer. the dimensionality is equal to the 
        #number of possible actions. Please note, the "action dim" 
        #and "state dim" kind of represent different things 
        self.fc_out = nn.Linear(32, action_dim)

        #specify our learning rate, or how fast we learn from data
        self.lr = q_lr

        #create optimizer, or how we learn/descend 
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):

        #first, convert our input from numpy array to tensor 
        x = torch.tensor(x) 

        #F = "functional"
        #this essentially forward props inputs through the network
        #the leaky_relu is just the activation function, mapping input
        #to space 0 to 1...
        q = F.leaky_relu(self.fc_1(x))
        q = F.leaky_relu(self.fc_2(q))
        #then, we have a final propagation layer
        #note: we do not use an activation function for the final layer
        #for many reasons, like the fact that values/rewards can be negative
        q = self.fc_out(q)
        return q
        
class DQNAgent:
    def __init__(self):
        self.state_dim     = 3
        self.action_dim    = 9  # 9개 행동 : -2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0
        self.lr            = 0.01
        #how much we value future rewards 
        self.gamma         = 0.98
        #variables delineating our epsilon greedy exploration method 
        self.epsilon       = 1.0
        self.epsilon_decay = 0.98
        self.epsilon_min   = 0.001
        #not using batch or buffer, but will probably have to in the future

        #so then, create our network 
        self.Q = QNetwork(self.state_dim, self.action_dim, self.lr)

    #make function for choosing action given our current state 
    def choose_action(self, state):
        """
        Choose an action based on state.
        Returns: 
        action: output for max value of the neural net
        real action: what we actually end up doing 
        maxQ_action_count: if we explored or exploited 
        
        """
        #create random number for deciding if we are exploring or exploiting 
        random_number = np.random.rand()
        maxQ_action_count = 0
        #if we decide to exploit/use our policy 
        if self.epsilon < random_number:
            #dont compute gradients 
            #could use gradients later on for backprop 
            with torch.no_grad():
                #get the action index associated with this 
                action = float(torch.argmax(self.Q(state)).numpy())
                #convert discretized number to actual action 
                #im not sure wy this is divided by 2 here
                #changing it to be divided by 4  
                real_action = (action - 4) / 4
                maxQ_action_count = 1
        else:
            #seems like better way would be:  np.random.randint(low, high)
            action = np.random.choice([n for n in range(9)])
            real_action = (action - 4) / 4  # -2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0

        return action, real_action, maxQ_action_count

    #this computes the target for the QNetwork
    #essentially it gets the max rewards from subsequent stages 
    #and combines it with the received reward 
    #almost like giving a base truth for a partial correction :D 

    #not using target network currently, as trying to get simplest possible
    #value...
    def calc_target(self, mini_batch):
        s, a, r, s_prime, done = mini_batch
        with torch.no_grad():
            #then, get the max value for the next state transition-pair  
            q = self.Q(s_prime).max()
            #then, create the target from the received reward, value of 
            #future rewards ratio (gamma), and value 
            target = r + self.gamma * done * q
        return target

    #this function is for training the Q function/value evaluator :D 
    def train_agent(self):

        #just get a sample, the most recent one  
        sample = self.mostRecentSample
        #then get all necessary elements 
        s, action, r, s_prime, done = sample
        #our action after the little transfrom may not be correct, so
        #change it 
        #a = a.type(torch.int64)

        #then, get the target from this info 
        td_target = self.calc_target(sample)

        #### Q train ####

        #gather function gets the outptus of a set of sets of inputs
        #using the data from self.Q(s) (so the states propagated thru network)
        #then, along dimension 1, and along index a 

        #RMK: this was along axis = 1 when using batching
        #currently not using batching, so only one sample
        #convert action to tensor of the correct type 
        action = torch.tensor(action).to(torch.int64)
        
        Q_a = self.Q(s).gather(0, action)

        #then, for each of them, get the loss, with the given calculated target
        #"td_target" is the "temporal difference target"
        #so we get a specific type of loss that we aim to minimize 
        q_loss = F.smooth_l1_loss(Q_a, td_target)
        #zero out gradients/reset computation 
        self.Q.optimizer.zero_grad()
        #recompute gradients by going backward on loss/difference
        q_loss.mean().backward()
        #make the optimizer step with respect to gradients 
        self.Q.optimizer.step()
        #### Q train ####

#yay the hard stuff is done! now, on to the main execution loop

if __name__ == '__main__':

    #create our agent 
    agent = DQNAgent()

    #create the environment 
    env = gym.make('Pendulum-v1')

    #number of episodes to train over 
    EPISODE = 500

    #create storage for scores:
    score_list = []

    #within each episode 
    for EP in range(EPISODE):
        #first reset episode-specific variables
        #including current state, score, done, and how many 
        #action steps we can take before exiting 
        #just take the state information for env.reset()....discard the additional information
        state = env.reset()[0]
        score, done = 0.0, False
        maxQ_action_count = 0

        stepCount = 0 

        #while we arent done with this process 
        while not done:
            #choose an action based on the current state 
            #get the translation of it as well 
            action, real_action, count = agent.choose_action(torch.FloatTensor(state))

            #get env vars based on the action we took 
            #might not need trunc idk 
            state_prime, reward, done, trunc, _ = env.step([real_action])

            #only working with single component memory currently 
            #agent.memory.put((state, action, reward, state_prime, done))
            agent.mostRecentSample = [state, action, reward, state_prime, done]

            #then store the rewards 
            score += reward
            maxQ_action_count += count

            #go to next state based on the action we took 
            state = state_prime

            #then train sample by sample. Could make it better by 
            #batching in the future 
            agent.train_agent()

            #uh leave after a bit 
            stepCount+=1 
            if(stepCount > 1000):
                break
            

        #then, after an episode, store the score 
        print("EP:{}, Avg_Score:{:.1f}, MaxQ_Action_Count:{}, Epsilon:{:.5f}".format(EP, score, maxQ_action_count, agent.epsilon))
        score_list.append(score)

        #then, if our epsilon is more than the minimum threshold for 
        #exploration vs exploitation, then we decrease it 
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

    #then, plot the total reward/score over time 
    #this inherently plots against the index of the episode 
    plt.plot(score_list)
    plt.show()

    np.savetxt(log_save_dir + '/pendulum_score.txt', score_list)








#not using this implementation anymore :D 

# model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
# model.learn(total_timesteps=10000, log_interval=10)
# model.save("ddpg_pendulum")
# vec_env = model.get_env()

# del model # remove to demonstrate saving and loading

# model = DDPG.load("ddpg_pendulum")

# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     env.render("human")