import matplotlib.pyplot as plt
import numpy as np

#hello! this package works with graphics that we need 
#please note, this usually opts for plotting the object we create,
#instead of passing the figure object as a return of the function 
    
def plot3d(pointsToPlot):
    """
    This function takes in plots of the form [3, num points]
    in order to plot it using matplotlib 
    yay :D 

    pointsToPlot: pointsToPlot 

    """
    #setUp Figure 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.plot(pointsToPlot[0,:], pointsToPlot[1,:], pointsToPlot[2,:] )
    plt.show()
        
def plotWalkerStar(allPlanes, ax = []): 
    """
    This function works with just plotting all the points in a walker star constellation.
    Inputs are planes with points in the xyz space 

    allPlanes: in the format (numPlanes, numPoints, 3<-x or y or z) 
    """

    #if we arent given the axes, set them up 
    if(ax == []): 
        #setUp Figure 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    #start plotting the points in the planes 
    for planeInd in range(np.shape(allPlanes)[0]): 
        #get the set of points for one plane 
        onePlanePointSet = allPlanes[planeInd]
        #plot the set of points  
        ax.scatter(onePlanePointSet[:,0], onePlanePointSet[:,1], onePlanePointSet[:,2], c = "black", s = 10, zorder = 2)

    # #show the plot 
    #plt.show()

def plot_sphere(radius=1, ax = []):

    #if we arent given the axes, set them up 
    if(ax == []): 
        #setUp Figure 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
        # Set equal aspect ratio for all axes
        ax.set_box_aspect([1, 1, 1])

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    # Create a meshgrid of spherical coordinates
    theta, phi = np.linspace(0, 2*np.pi, 50), np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    # Plot the sphere
    ax.plot_surface(x, y, z, color='g', alpha=1, zorder = 1)

    # # # Show the plot
    # plt.show()    

def plotPoints(points, ax = []): 
    #if we arent given the axes, set them up 
    if(ax == []): 
        #setUp Figure 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    for point in points: 
        ax.scatter(point[0], point[1], point[2], c = "red", s = 50, zorder = 2)
   

def plot_line_segments(pair_of_points_list, ax = []):

    if(ax == []): 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Set labels and title (optional)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('3D Line Segments Plot')

    for pair in pair_of_points_list:
        # Extract coordinates of the two points in each pair
        x1, y1, z1 = pair[0]
        x2, y2, z2 = pair[1]

        # Plot the line segment for each pair
        ax.plot([x1, x2], [y1, y2], [z1, z2], marker='o', c = "black", markersize = 0)


#trying for pipeline of sphere, walker star, base stations 
def multiPlot(radius, satPoints, baseStationPoints, links, axisLimit = 8000): 
    #get figure and axes to use for all this 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis limits
    ax.set_xlim([-axisLimit, axisLimit])  # Adjust the limits for the X-axis
    ax.set_ylim([-axisLimit, axisLimit])  # Adjust the limits for the Y-axis
    ax.set_zlim([-axisLimit, axisLimit])  # Adjust the limits for the Z-axis

    #plot satellites   
    plotWalkerStar(satPoints, ax)

    #plot base stations 
    plotPoints(baseStationPoints, ax)

    #plot sphere 
    plot_sphere(radius, ax)
    
    #plot links 
    plot_line_segments(links, ax)
     
    #show entire plot 
    plt.show()