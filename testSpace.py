# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # Create a figure and axis
# fig, ax = plt.subplots()
# x_data = np.linspace(0, 2 * np.pi, 100)
# line, = ax.plot(x_data, np.sin(x_data))

# # Animation update function
# def update(frame):
#     line.set_ydata(np.sin(x_data + frame * 0.1))  # Update the y-data for the line
#     return line,

# # Create animation and assign it to a variable
# animation = FuncAnimation(fig, update, frames=range(100), interval=50)

# # Keep a reference to the animation, so it's not deleted
# # Note: You can use any variable name, not necessarily "anim"
# animation.save()

# # Display the animation
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pdb 

#pdb.set_trace() 
import matplotlib
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt



# Generate data
x = np.linspace(0, 10, 100)  # Create an array of 100 values from 0 to 10
y = np.sin(x)  # Compute the sine of each element in x

# Create a line plot
plt.plot(x, y, label='Sin(x)')  # Plot the data, label for legend

# Customize the plot
plt.title('Simple Line Plot')  # Set the title
plt.xlabel('X-axis')  # Label for the X-axis
plt.ylabel('Y-axis')  # Label for the Y-axis
plt.legend()  # Display legend

# Show the plot
plt.show()
