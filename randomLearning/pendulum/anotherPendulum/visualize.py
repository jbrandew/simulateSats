import tensorflow as tf
import numpy as np
import gym
import math
from PIL import Image
import pygame, sys
from pygame.locals import *
from tensorflow import keras

#pygame essentials
pygame.init()
DISPLAYSURF = pygame.display.set_mode((500,500),0,32)
clock = pygame.time.Clock()
pygame.display.flip()

#openai gym env
env = gym.make('Pendulum-v1')
state = env.reset()

done = False
count=0
done=False
steps = 0
#loading trained model
model = tf.keras.models.load_model('pendulum/actor/')
total_wins =0
episodes = 0


def print_summary(text,cood,size):
        font = pygame.font.Font(pygame.font.get_default_font(), size)
        text_surface = font.render(text, True, (0,0,0))
        DISPLAYSURF.blit(text_surface,cood)
     
while episodes<1000 :
    pygame.event.get()
    for event in pygame.event.get():
                if event.type==QUIT:
                                pygame.quit()
                                raise Exception('training ended')
    # Get the action probabilities from the policy network
    # Choose an action based on the action probabilities
    
    action = model.predict(np.array(state).reshape(-1,3))[0]
    
    next_state, reward, done, info = env.step(action) # take a step in the environment
    print('reward and done?',reward,done)
    image = env.render(mode='rgb_array') # render the environment to the screen
   
    #convert image to pygame surface object
    image = Image.fromarray(image,'RGB')
    mode,size,data = image.mode,image.size,image.tobytes()
    image = pygame.image.fromstring(data, size, mode)

    DISPLAYSURF.blit(image,(0,0))
    pygame.display.update()
    clock.tick(100)
    if done:
        state = env.reset()
        pygame.display.update()
        pygame.time.delay(100)
        episodes+=1
        
    pygame.time.delay(100)
    state = next_state

pygame.quit()