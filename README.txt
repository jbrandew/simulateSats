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