import gym 
import random 
import pdb 

#created environment 
env = gym.make('CartPole-v1', render_mode="rgb_array")

#created states from env () 
#there are 4 states, which im not entirely sure on
#i thought it would kind of be discretized 
states = env.observation_space.shape[0]
#got the number of possible actions in the space 
actions = env.action_space.n
#objective of this setup: get at least 200 points, where a point
#is awarded at each time step in which the cart pole has not fallen over 
print(actions) 

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    #step until we are done with game
    #i think in this case its if the pole has fallen over, or if
    #we have reached the 200 point mark 
    while not done:
        #allows us to see the environment and what the agent is doing :D 
        env.render()
        #randomize the step direction (so 0 or 1 for direction of step)
        action = random.choice([0,1])
        #get the info from the environment step based on action 
        #n_state: state of the environment 
        #reward: if we performed well :D 
        #done: env. is done changing. seems like if reward = 0 -> means env. is done
        #truncated: if the current time step got stepped early based
        #on some event like a spaceship hitting a wall :D. so this being true
        #implies done being true i believe. can also be caused by other externals,
        #like reaching the time limit for training 
        #info: uhhh....other info..not sure how this differs from state
        #i think its mainly that this is debugging and not used for training 
        n_state, reward, done, truncated, info = env.step(action)
        #after getting the proper reward from environment, add reward
        #to this score 
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))

#now, after just trying random stuff, work with actual deep learning model 
#import necessary packages and build the model 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

#use different activations for each layer 
def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

model = build_model(states, actions)


from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))

















