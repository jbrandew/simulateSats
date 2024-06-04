import matplotlib.pyplot as plt
import numpy as np
import pdb

# Fixing random state for reproducibility
np.random.seed(19680801)

# Compute areas and colors
N = 150
r = 2 * np.random.rand(N)
theta = 2 * np.pi * np.random.rand(N)
area = 200 * r**2
colors = theta

#sameColor = np.linspace(0,1,N)
sameColor = np.ones(N)
#sameColor[0] = 2

sameColor[0:10] = 0

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
#pdb.set_trace()
c = ax.scatter(theta, r, c=sameColor, s=area, cmap='plasma', alpha=0.75)

plt.show()