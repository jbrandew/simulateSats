import numpy as np

def calculate_new_position(radius, normal_vector, current_position, angle_rad):
    # Calculate the rotation matrix using the normal vector
    rotation_matrix = np.array([[np.cos(angle_rad) + normal_vector[0]**2 * (1 - np.cos(angle_rad)),
                                 normal_vector[0] * normal_vector[1] * (1 - np.cos(angle_rad)) - normal_vector[2] * np.sin(angle_rad),
                                 normal_vector[0] * normal_vector[2] * (1 - np.cos(angle_rad)) + normal_vector[1] * np.sin(angle_rad)],
                                [normal_vector[1] * normal_vector[0] * (1 - np.cos(angle_rad)) + normal_vector[2] * np.sin(angle_rad),
                                 np.cos(angle_rad) + normal_vector[1]**2 * (1 - np.cos(angle_rad)),
                                 normal_vector[1] * normal_vector[2] * (1 - np.cos(angle_rad)) - normal_vector[0] * np.sin(angle_rad)],
                                [normal_vector[2] * normal_vector[0] * (1 - np.cos(angle_rad)) - normal_vector[1] * np.sin(angle_rad),
                                 normal_vector[2] * normal_vector[1] * (1 - np.cos(angle_rad)) + normal_vector[0] * np.sin(angle_rad),
                                 np.cos(angle_rad) + normal_vector[2]**2 * (1 - np.cos(angle_rad))]])

    # Calculate the new position using the rotation matrix
    new_position = np.dot(rotation_matrix, np.array(current_position))

    return new_position

# Example usage
radius = 5.0
normal_vector = [0.0, 0.0, 1.0]
current_position = [5.0, 0.0, 0.0]
angle_rad = np.pi / 2.0

new_position = calculate_new_position(radius, normal_vector, current_position, angle_rad)
print("New Position:", new_position)



# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from mpl_toolkits.mplot3d import Axes3D

# # Function to update the plot for each frame of the animation
# def update(frame):
#     # Clear the previous plot
#     #line[0].remove()
#     ax.cla() 
#     #plt.clf() 

#     # Update the data for the 3D parametric surface (here, a helix)
#     t = np.linspace(0, 4 * np.pi, 100)
#     x = np.sin(t + frame * 0.1)
#     y = np.cos(t + frame * 0.1)
#     z = t
    
#     # Plot the updated 3D parametric surface as a wireframe
#     ax.plot(x, y, z, color='b', label='Helix', linewidth=2)[0]
    
#     # Set plot labels and title
#     ax.set_xlabel('X-axis')
#     ax.set_ylabel('Y-axis')
#     ax.set_zlabel('Z-axis')
#     ax.set_title('3D Helix Animation')
    
#     # Display legend
#     ax.legend()

# # Set up the initial 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Create an empty line to be updated in the animation
# line = [ax.plot([], [], [], color='b', label='Helix', linewidth=2)[0]]

# # Set the axis limits
# ax.set_xlim([-1.5, 1.5])
# ax.set_ylim([-1.5, 1.5])
# ax.set_zlim([0, 4 * np.pi])

# # Create the animation with 100 frames, each lasting 50 milliseconds
# animation = FuncAnimation(fig, update, frames=100, interval=50, fargs=(line,))

# # Display the animation
# plt.show()





# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from mpl_toolkits.mplot3d import Axes3D

# # Function to update the plot for each frame of the animation
# def update(frame, line):
#     # Clear the previous plot
#     line[0].remove()
    
#     # Update the data for the 3D parametric surface (here, a helix)
#     t = np.linspace(0, 4 * np.pi, 100)
#     x = np.sin(t + frame * 0.1)
#     y = np.cos(t + frame * 0.1)
#     z = t
    
#     # Plot the updated 3D parametric surface as a wireframe
#     line[0] = ax.plot(x, y, z, color='b', label='Helix', linewidth=2)[0]
    
#     # Set plot labels and title
#     ax.set_xlabel('X-axis')
#     ax.set_ylabel('Y-axis')
#     ax.set_zlabel('Z-axis')
#     ax.set_title('3D Helix Animation')
    
#     # Display legend
#     ax.legend()

# # Set up the initial 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Create an empty line to be updated in the animation
# line = [ax.plot([], [], [], color='b', label='Helix', linewidth=2)[0]]

# # Set the axis limits
# ax.set_xlim([-1.5, 1.5])
# ax.set_ylim([-1.5, 1.5])
# ax.set_zlim([0, 4 * np.pi])

# # Create the animation with 100 frames, each lasting 50 milliseconds
# animation = FuncAnimation(fig, update, frames=100, interval=50, fargs=(line,))

# # Display the animation
# plt.show()



# # import numpy as np
# # import matplotlib.pyplot as plt
# # from matplotlib.animation import FuncAnimation

# # # Create a figure and axis
# # fig, ax = plt.subplots()
# # x_data = np.linspace(0, 2 * np.pi, 100)
# # line, = ax.plot(x_data, np.sin(x_data))

# # # Animation update function
# # def update(frame):
# #     line.set_ydata(np.sin(x_data + frame * 0.1))  # Update the y-data for the line
# #     return line,

# # # Create animation and assign it to a variable
# # animation = FuncAnimation(fig, update, frames=range(100), interval=50)

# # # Keep a reference to the animation, so it's not deleted
# # # Note: You can use any variable name, not necessarily "anim"
# # animation.save()

# # # Display the animation
# # plt.show()
# import pdb
# #pdb.set_trace()
# import matplotlib.pyplot as plt
# import numpy as np  
# import pdb 

# #pdb.set_trace() 
# import matplotlib
# #matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt



# # Generate data
# x = np.linspace(0, 10, 100)  # Create an array of 100 values from 0 to 10
# y = np.sin(x)  # Compute the sine of each element in x

# # Create a line plot
# plt.plot(x, y, label='Sin(x)')  # Plot the data, label for legend

# # Customize the plot
# plt.title('Simple Line Plot')  # Set the title
# plt.xlabel('X-axis')  # Label for the X-axis
# plt.ylabel('Y-axis')  # Label for the Y-axis
# plt.legend()  # Display legend

# # Show the plot
# plt.show()
