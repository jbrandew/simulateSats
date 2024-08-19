import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import numpy as np
import argparse
from itertools import count

import pdb
#create actor
#actor is the thing that decides what action to take based on the state 
#so # inputs = state_dim 
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        #maybe this should be scaled by the min value as well... 
        #im not sure if we need like a specified layer to backprop on to 
        x = self.max_action * torch.tanh(self.l3(x))

        return x

#create critic for actions
#essentially, this evaluates if the action was good or not
#so has input shape = state_dim + action_dim  
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 1)

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        #weirdo type of final layer 
        x = self.l3(x)

        return x

#create buffer for data 
class Replay_Buffer(object):
    def __init__(self):
        self.buffer = []
        self.max_size = args.buffer_max_size
        self.index = 0

    def save(self, data):
        if len(self.buffer) == self.max_size:
            self.buffer[int(self.index)] = data
            self.index = (self.index + 1) % self.max_size
        else:
            self.buffer.append(data)

    #just get some sample of the data :) 
    def sample(self):
        random_index = np.random.choice(len(self.buffer), args.batch_size, replace=False)
        state = [self.buffer[i]['state'] for i in random_index]
        action = [self.buffer[i]['action'] for i in random_index]
        reward = [self.buffer[i]['reward'] for i in random_index]
        next_state = [self.buffer[i]['next_state'] for i in random_index]
        done = [1 - int(self.buffer[i]['done']) for i in random_index]

        return state, action, reward, next_state, done

#then, create our DDPG agent :D 
class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        #create actor and critics, with targets for each 
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        #match the weights and set the optimizer 
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        #create buffer for data 
        self.replay_buffer = Replay_Buffer()

    def select_action(self, state):
        #format state and get which action we want, using the model network  
        state = torch.FloatTensor(np.array(state)).unsqueeze(0)
        action = self.actor(state)

        return action.cpu().data.numpy().flatten()

    def update(self, ):
        #get our data 
        #pdb.set_trace() 
        state, action, reward, next_state, done = self.replay_buffer.sample()

        pdb.set_trace() 

        #create the state stuff  
        pdb.set_trace() 
        state = torch.FloatTensor(np.array(state))
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(np.array(next_state))

        done = torch.Tensor(done).unsqueeze(1)

        #create the target value 
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + done * args.gamma * target_Q.detach()
        #create the model value 
        Q = self.critic(state, action)

        #create the loss of the two 
        loss_critic = F.mse_loss(Q, target_Q)

        #go back on the model critic (using this...)
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        #so maximize the value of the chosen state-action pair
        #you are doing mean as you are computing across a batch 
        loss_actor = - self.critic(state, self.actor(state)).mean()

        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        #soft update to the target networks based on the model networks
        #i guess the update is as often as the model network, but its just a smaller update 
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data = args.tau * param.data + (1 - args.tau) * target_param.data
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data = args.tau * param.data + (1 - args.tau) * target_param.data
    

#now, for runtime code :D 
#first, enact parsing for hyperparameters: 
parser = argparse.ArgumentParser()

parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', type=int, default=9924)
parser.add_argument('--buffer-max-size', type=int, default=10000)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--total-episode', type=int, default=1000)
parser.add_argument('--exploration-noise', type=float, default=0.1)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--render', action='store_true')
parser.add_argument('--render_interval', type=int, default=20)

args = parser.parse_args()

#create our environment and get associated variables 
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]

#then enter the main runtime code 
if __name__ == '__main__':
    #create our agent 
    agent = DDPG(state_dim, action_dim, max_action)
    total_step = 0

    #iterate through episodes 
    for episode in range(args.total_episode):
        episode_reward = 0
        #reset the state, only taking the state info and not the debug info 
        state = env.reset()[0]
        #iterate indefinitely until we break
        #but still keep iterable number 
        for t in count():
            
            #get action using the model agent network 
            pdb.set_trace()
            action = agent.select_action(state)
            #add noise to whatever action we pick 
            action = (action + np.random.normal(0, args.exploration_noise, size=(action_dim))).clip(min_action,
                                                                                                    max_action)
            #get the info from stepping 
            
            next_state, reward, done, trunc, _ = env.step(action)

            #create dict and then save data
            data = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done}
            agent.replay_buffer.save(data)

            #update if we have enough data 
            if len(agent.replay_buffer.buffer) >= args.batch_size: agent.update()

            if args.render and episode % args.render_interval == 0: env.render()

            state = next_state
            total_step += 1
            episode_reward += reward
            if done:
                break

            if(t > 10000):
                break

            if(t%100 == 0 ):
                print(t)

        print('Total step: {}  Episode: {}  Episode reward: {:.2f}'.format(total_step, episode, episode_reward))