import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation
import pdb

class DataHolder(): 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y 

        self.currX = x[0]
        self.currY = y[0]

fig, ax = plt.subplots()
t = np.linspace(0, 3, 40)
g = -9.81
v0 = 12
z = g * t**2 / 2 + v0 * t

v02 = 5
z2 = g * t**2 / 2 + v02 * t

ax.scatter(1, 0, c="b", s=5, label=f'v0 = {v0} m/s')
line2 = ax.plot(1, 0, label=f'v0 = {v02} m/s')[0]
ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel='Time [s]', ylabel='Z [m]')
ax.legend()

dataHolder = DataHolder(t, z2)

#create function to have similar setup 
def updateData(frame):
    update(frame)
    ax.plot(dataHolder.currX, dataHolder.currY)

def update(frame):
    dataHolder.currX = t[0:frame]
    dataHolder.currY = z2[0:frame]  

ani = animation.FuncAnimation(fig=fig, func=updateData, frames=40, interval=30)
plt.show()

