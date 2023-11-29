import gym
import __init__
import time

env = gym.make('pressureplate-linear-4p-v0')
env.reset()
env.reset()

for i in range(10000):
    #print(i)
    env.render()
    input("Enter something: ")
    
    if i == 0 or i == 1:
        env.step( (3,3,4,4) )
    elif i == 2 or i == 3:
        env.step( (4,1,4,4 ) )
    elif i == 4:
        env.step( (3,1,4,4) )
    elif i == 5:
        env.step( (3,1,4,4) )
    else:
        env.step( (4,4,4,4) )
    
    #env.step(env.action_space.sample())
    #time.sleep(0.25)
    