import matplotlib.pyplot as plt
import numpy as np
import pdb 
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
        
def plotWalkerStar(allPlanes, ax = [], pointSizes = []): 
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

    #reshape points across planes
    points = np.reshape(allPlanes, [360,3])

    #then scatter based on how many sizes we have 
    if(len(pointSizes) == 0): 
        ax.scatter(points[:,0], points[:,1], points[:,2], c = "black", s = 10, zorder = 2)
    else: 
        ax.scatter(points[:,0], points[:,1], points[:,2], c = "black", s = pointSizes, zorder = 2)

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
   

def plot_line_segments(pair_of_points_list, 
                       numLinks, 
                       ax = []):

    if(ax == []): 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Set labels and title (optional)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('3D Line Segments Plot')

    for ind, pair in enumerate(pair_of_points_list):
        if(ind > numLinks): 
            break 
        # Extract coordinates of the two points in each pair
        x1, y1, z1 = pair[0]
        x2, y2, z2 = pair[1]

        # Plot the line segment for each pair
        ax.plot([x1, x2], [y1, y2], [z1, z2], marker='o', c = "black", markersize = 0)
     

class GraphicsView:
    #really shouldnt have access to manager
    def __init__(self, manager, fig, ax):
        self.manager = manager
        self.fig = fig 
        self.ax = ax 
        
    #trying for pipeline of sphere, walker star, base stations 
    def multiplot(self, 
                  radius, 
                  satPoints, 
                  baseStationPoints, 
                  links,
                  numLinks,  
                  pointSizes = [],
                  axisLimit = 8000,
                  showFigure = True,
                  ): 
        """
        Function to plot all players and links. Plotting with the globe doesnt work very well, 
        as it tends to absorb links/players graphically...so for now just make the radius very small 

        Inputs: 
        radius: radius of the globe 
        satPoints: xyz of all satellites 
        baseStationPoints: xyz of all base stations 
        links: connections to plot using line segments 
        numLinks: how many links we use 
        pointSizes: how large each point should be 
        showFigure: do we output the figure at the end? 

        """

        #first, clear axes
        self.ax.cla()

        #get figure and axes to use for all this 
        if(self.fig==[]): 
           self.fig = plt.figure()
           self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set axis limits
        self.ax.set_xlim([-axisLimit, axisLimit])  # Adjust the limits for the X-axis
        self.ax.set_ylim([-axisLimit, axisLimit])  # Adjust the limits for the Y-axis
        self.ax.set_zlim([-axisLimit, axisLimit])  # Adjust the limits for the Z-axis

        #plot satellites   
        plotWalkerStar(satPoints, self.ax, pointSizes)

        #plot base stations 
        plotPoints(baseStationPoints, self.ax)

        #plot sphere 
        plot_sphere(radius, self.ax)
        
        #plot links 
        #pdb.set_trace() 
        plot_line_segments(links, min(len(links),numLinks), self.ax)

        if(showFigure):
            plt.show() 

    # def multiplot(self): 
    #    self.update_graphics()
    #    plt.show() 













