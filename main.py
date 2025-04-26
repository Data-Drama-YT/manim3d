import sys
import os

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

# Example Usage Module
def create_examples():
    """Create example graphs to demonstrate the library capabilities"""
    # Initialize the library
    bg = Graph3D()
    plotter = GraphPlotter(bg)
    
    # Create a coordinate system
    coord_system = plotter.setup_cartesian_system(dimension=3, size=10)
    plotter.animate_3d_overfitting()
    return {
        "coordinate_system": coord_system,
    }
    #bg.plot_function_3d(sq)
    # Example 1: 2D Function
    #sine = plotter.sine_wave(amplitude=2, frequency=0.5, phase=0)
    
    # Example 2: 3D Surface
    #paraboloid = plotter.paraboloid(a=0.2, b=0.2, x_range=(-5, 5), y_range=(-5, 5))
    
    # Example 3: Parametric curve
    #helix = plotter.helix(radius=3, pitch=0.8, num_turns=5)
    
    # Example 4: Animated function
    #wave_anim = plotter.animate_3d_wave(frames=120, start_frame=1)
    # Define our parabolic function
    #def parabola_func(x, y, t):
    #    """Demo function: An expanding parabolic surface"""
        # Scale the parabola with time (t goes from 0 to 1)
       # amplitude = 2 * t  # Height grows over time
      #  spread = 0.5 + 1.5 * t  # Width expands over time
        
        # Create a circular parabola
     #   r_squared = x**2 + y**2
     #   return amplitude * (1 - r_squared/(spread**2))
    
    #parabola_animation=bg.animate_function_3d(parabola_func,x_range=(-2,2),y_range=(-2,2))
    #point = bg.create_points_from_csv('points_list.csv',animate=True,show_from=5,animation_interval=10)
    # Example 5: Implicit surface
    #sphere = plotter.sphere(radius=3, center=(0, 0, 6))
    
create_examples()