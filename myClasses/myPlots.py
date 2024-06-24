import matplotlib.pyplot as plt
import numpy as np
import pdb 
import myPackages.myMath as myMath 

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
        
def plotWalkerStar(allPlanes, ax = [], sphereColors = []): 
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
    if(len(sphereColors) == 0): 
        ax.scatter(points[:,0], points[:,1], points[:,2], c = "black", s = 10, zorder = 2, alpha=1)
    else:
        #use log scale with the sphereColors. Sphere colors initially are from 0 to 1, 1 being brightest 
        
        sphereColors = sphereColors > 0 #np.log(sphereColors + 1)

        #make size and color vary based on queue size 
        #s can be array or single # 
        s = 20
        if(max(sphereColors) != min(sphereColors)): 
            s = (sphereColors)*80 + 20
        ax.scatter(points[:,0], points[:,1], points[:,2], c=sphereColors, cmap = "magma", zorder = 2, alpha=1, s = s)

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
        ax.scatter(point[0], point[1], point[2], c = "red", s = 25, zorder = 2)
   

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

def plotXY(x, y, xlabel = "", ylabel = "", title = ""): 
    """
    Generic plotting
    """
    
    fig,ax = plt.subplots(1)
    ax.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

class GraphicsView:
    """
    This class is just for graphing to a given axes. Uses generic util functions as above ^
    
    """

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
                  sphereColors = [],
                  axisLimit = 8000e3,
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
        sphereColors: how large each point should be 
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
        plotWalkerStar(satPoints, self.ax, sphereColors)

        #plot base stations 
        #plotPoints(baseStationPoints, self.ax)

        #plot sphere 
        plot_sphere(radius, self.ax)
        
        #plot links 
        plot_line_segments(links, min(len(links),numLinks), self.ax)

        #if(showFigure):
        #    plt.show() 

    # def multiplot(self): 
    #    self.update_graphics()
    #    plt.show() 













