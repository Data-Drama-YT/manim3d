import bpy
from mathutils import Vector
import sympy as sp
import numpy as np
import traceback

import bpy
import numpy as np
import sympy as sp
from mathutils import Vector

class Graph2D:
    def __init__(self, points=[], grid_size=10, spacing=1,
                 name="Graph2D", material=None,
                 animation_start_frame=0,
                 animation_point_interval=1):
        """
        points: list of (x,y) tuples
        material: Blender material to apply (optional)
        """
        # Ensure all points are in 3D format (x, y, 0)
        self.points = [Vector((x, y, 0)) for x, y in points] if points else []
        self.animation_start_frame = animation_start_frame
        self.animation_point_interval = animation_point_interval
        self.name = name
        self.grid_size = grid_size
        self.spacing = spacing
        self.material = material
        self.grid_collection = None
        self.obj = None

    def draw_line(self, start, end, name="MyLine", collection=None, animate=False, start_frame=1, duration=20):
        """
        Draws a 3D line between two points (start, end).
        """
        # Ensure points are 3D
        start = (start[0], start[1], 0) if len(start) == 2 else start
        end = (end[0], end[1], 0) if len(end) == 2 else end

        # Create curve data
        curve_data = bpy.data.curves.new(name=f"{name}_Curve", type='CURVE')
        curve_data.dimensions = '3D'

        # Add polyline with two points
        polyline = curve_data.splines.new(type='POLY')
        polyline.points.add(count=1)
        polyline.points[0].co = (*start, 1)
        polyline.points[1].co = (*end, 1)

        # Create object from curve
        curve_obj = bpy.data.objects.new(name, curve_data)

        # Link to collection
        if collection:
            collection.objects.link(curve_obj)
        else:
            bpy.context.collection.objects.link(curve_obj)

        # Add thickness to make it visible
        curve_data.bevel_depth = 0.01
        curve_data.bevel_resolution = 3

        if animate:
            # Set initial bevel factor
            curve_data.bevel_factor_start = 0.0
            curve_data.bevel_factor_end = 0.0
            curve_data.keyframe_insert(data_path="bevel_factor_end", frame=start_frame)

            # Animate to full draw
            curve_data.bevel_factor_end = 1.0
            curve_data.keyframe_insert(data_path="bevel_factor_end", frame=start_frame + duration)

        return curve_obj

    def draw_cartesian_grid(self, collection_name='Grid_Lines'):
        """
        The goal is to draw a grid of lines using draw_line
        """
        self.grid_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(self.grid_collection)
        half = self.grid_size // 2  # Useful for centering the grid

        for x in range(-half, half + 1):
            self.draw_line(
                (x * self.spacing, -half * self.spacing),
                (x * self.spacing, half * self.spacing),
                name=f"VLine{x}",
                collection=self.grid_collection
            )

        for y in range(-half, half + 1):
            self.draw_line(
                (-half * self.spacing, y * self.spacing),
                (half * self.spacing, y * self.spacing),
                name=f"HLine_{y}",
                collection=self.grid_collection
            )

    def erect_grid(self):
        """
        Rotate the grid by 90 degrees on the x axis.
        """
        if self.grid_collection:
            for obj in self.grid_collection.objects:
                obj.rotation_euler = (1.5708, 0, 0)  # 90 degrees on the x axis.

    def create_icosphere(self, location):
        """
        This is a utility to plot points on the 2D graph
        """
        bpy.ops.mesh.primitive_ico_sphere_add(
            radius=0.1,
            location=(location.x, location.y, location.z)
        )
        sphere = bpy.context.object
        return sphere

    def plot_points(self, points):
        frame_counter = self.animation_start_frame
        point_counter = 0
        for point in points:
            # Ensure point is in 3D format
            if len(point) == 2:
                point = (point[0], point[1], 0)

            sphere = self.create_icosphere(Vector(point))
            self.animate_point(sphere, frame_counter)
            frame_counter += self.animation_point_interval
            point_counter += 1

    def animate_point(self, sphere, frame):
        # Make the point invisible initially
        sphere.hide_viewport = True
        sphere.hide_render = True

        # Insert a keyframe for visibility at the given frame
        sphere.keyframe_insert(data_path="hide_viewport", frame=frame)
        sphere.keyframe_insert(data_path="hide_render", frame=frame)

        # Make the point visible after the time interval (next frame)
        sphere.hide_viewport = False
        sphere.hide_render = False

        # Insert a keyframe to make it visible after the time interval
        sphere.keyframe_insert(data_path="hide_viewport", frame=frame + self.animation_point_interval)
        sphere.keyframe_insert(data_path="hide_render", frame=frame + self.animation_point_interval)

    def plot_linear_equation(self, m, c, x_start=-5, x_end=5, step=1):
        """
        Plots y = mx + c by sampling x from x_start to x_end with the given step.
        Places icospheres on each calculated (x, y).
        Also draws a line from the first point to the last.
        """
        points = []

        x = x_start
        while x <= x_end:
            y = m * x + c
            points.append((x, y))
            x += step

        self.plot_points(points)

        # Draw a line from the first to the last point
        if len(points) >= 2:
            start = points[0] + (0,)
            end = points[-1] + (0,)
            self.draw_line(
                start, end,
                collection=self.grid_collection,
                animate=True, start_frame=10, duration=40)

    def draw_curve(self, points, name="Curve", collection=None, animate=True, start_frame=1, duration=1):
        """
        Draw a curve by connecting each consecutive pair of points.
        points: list of (x, y, z) tuples or (x, y) tuples, which will be treated as (x, y, 0).
        """
        # Ensure that the points are in the correct format (x, y, z)
        # If point is (x, y), convert it to (x, y, 0)
        points = [(point[0], point[1], 0) if len(point) == 2 else point for point in points]

        # Iterate through consecutive pairs of points
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            self.draw_line(start, end, name=f"{name}_Segment_{i}", collection=collection, animate=animate, 
                           start_frame=start_frame + i * duration, duration=duration)

    def plot_function(self, expr_str, x_range=(-10, 10), step=0.5):
        """
        Plots a math function like 'x**2' or 'sin(x)' by generating points.
        
        - expr_str: string, e.g. "x**2" or "sin(x)"
        - x_range: tuple, the range of x values
        - step: float, the spacing between x points
        """
        x = sp.symbols('x')
        expr = sp.sympify(expr_str)

        # Evaluate y for x in given range
        x_vals = np.arange(x_range[0], x_range[1], step)
        y_vals = [float(expr.subs(x, val)) for val in x_vals]

        # Create 3D points (x, y, 0) for the XY grid
        points = [(float(x), float(y), 0) for x, y in zip(x_vals, y_vals)]

        # Plot points
        self.plot_points(points)

        # Draw line connecting first and last points
        if len(points) >= 2:
            self.draw_curve(points)




my_graph=Graph2D(grid_size=50,animation_point_interval=2)
my_graph.draw_cartesian_grid()
#my_graph.erect_grid()
#my_graph.plot_points([(1,2),(3,4)])
my_graph.plot_linear_equation(m=1,c=3)
my_graph.plot_function("x**2 + 2*x + 1", x_range=(-5, 5), step=0.2)
# Print the error type
#print(f"Error Type: {type(e).__name__}")
# Optionally print the error message
#print(f"Error Message: {str(e)}")
#print("Stack Trace:")
#traceback.print_exc()
# Delete all objects
#bpy.ops.object.select_all(action='SELECT')
#bpy.ops.object.delete()
# Delete all collections
#for collection in bpy.data.collections:
#   bpy.data.collections.remove(collection)