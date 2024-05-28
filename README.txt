Hi Jun :D 
Here are a couple basic use cases you can have fun with: 

Plotting Topologies: 
A topology is essentially a structure of how a set of satellites are connected. 

You can connect different topologies in the following function format: 
simmer.manager.<function related to topology you want to connect>() 

This function could be: 
"connectSpiralTopologySimple()" 
"connect2ISL()" 
"connectZigZagTopology()" 

And then, you can make a pretty plot of the connections using the "multiplot" function. 

Simulating Transmissions: 

You can simulate a set # of transmissions with different properties. These functions could be: 

"executeChainSimulationWithCollisions()"
-this simulates a bunch of transmits from users across the globe, and uses the exponential distributions to 
model the process rate of packets at satellites or base stations
"simulateTransmits(100)" 
-this simulates n transmits without collisions/process wait time at servers 

Uh yea, good luck and have fun looking around. 


Command to set python path on start: 
export PYTHONPATH=`pwd` or /home/jbrandew/projects/simulateSats

Needed packages/install method: 
pip install pyyaml pdb math heapq numpy matplotlib

Random notes: 
based on walker delta 360 (20,18) constellation. 
most math is done in terms of km (i.e., divide by 3e5 for prop delay)
base stations are not connected to other base stations 

Used the following guide to be able to use graphics: 
https://medium.com/@shaoyenyu/make-matplotlib-works-correctly-with-x-server-in-wsl2-9d9928b4e36a
Summary: 
to be able to use graphics like matplot lib use the guide above, and remember
to have xlaunch running each time you execute your code 

By the way, the architecture of this project is loosely based on the model, 
view, controller architecture. There isnt a contant GUI running per say, 
but its more so around the structure of: 
view/GraphicsView: contains plotting functionality 
model/Manager: contains computational components 
controller/Simulator: contains simulation processes, functioning as the 
interface between model/manager and the view/GraphicsView

General code structure: 
main: has necessary config and runs general execution requests 
Manager: has functions for initialization and runtime updates of topology, including satellites and base stations. Also contains all the useful actual objects 
Player: interface/general implementations for stuff like base stations and satellites. also has necessary data structures, like PQueue that are used within player implementations
Simulator: contains execution protocols. Assumes already initialized framework, then operates on that framework for simulations 
Event: will contain all types of events that could occur in the environment 

