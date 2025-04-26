import sys
import os
import time
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add it to the system path
if script_dir not in sys.path:
    sys.path.append(script_dir)

from graphplotter3d import GraphPlotter
from graph3d import Graph3D
from blenderutils import delete_all_objects_and_collections

def sq(x,y):
    return x**2+y**2
bg =Graph3D()
cart=bg.create_cartesian_system()
point=bg.create_point((1,4,2))
#bg.highlight_point(point)
#bg.unhighlight_point(point)