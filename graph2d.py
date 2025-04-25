import bpy
from mathutils import Vector

class Graph2D:
    def __init__(self,points=[],grid_size=10,spacing=1,
                 name="Graph2D",material=None,
                 animation_start_frame=0,
                 animation_point_interval=1):
        """
        points: list of (x,y) tuples
        material: Blender material to apply (optional)
        """
        self.points=[Vector((x,y,0)) for x,y in points] or []
        self.animation_start_frame=animation_start_frame
        self.animation_point_interval=animation_point_interval
        self.name=name
        self.grid_size=grid_size
        self.spacing=spacing
        self.material=material
        self.grid_collection=None
        self.obj=None

    def draw_line(self, start, end, name="MyLine", collection=None, animate=False, start_frame=1, duration=20):

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

    
    def draw_cartesian_grid(self,collection_name='Grid_Lines'):
        """
        The goal is to draw a grid of lines using draw_line
        """
        self.grid_collection=bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(self.grid_collection)
        half=self.grid_size//2 #useful for centering the grid

        for x in range(-half,half+1):
            self.draw_line(
                (x*self.spacing,-half*self.spacing,0),
                (x*self.spacing,half*self.spacing,0),
                name=f"VLine{x}",
                collection=self.grid_collection
            )
        
        for y in range(-half,half+1):
            self.draw_line(
                (-half * self.spacing, y * self.spacing, 0),
                (half * self.spacing, y * self.spacing, 0),
                name=f"HLine_{y}",
                collection=self.grid_collection
            )
    
    def erect_grid(self):
        """
        Rotate the grid by 90 degrees on the x axis.
        """
        if self.grid_collection:
            for obj in self.grid_collection.objects:
                obj.rotation_euler=(1.5708,0,0) # This is 90 degrees on the x axis.

    def create_icosphere(self,location):
        """
        This is a utility to plot points on the 2D graph
        """
        bpy.ops.mesh.primitive_ico_sphere_add(
            radius=0.1,
            location=(location.x, location.y, location.z)
        )
        sphere=bpy.context.object
        return sphere
    
    def plot_points(self,points):
        frame_counter=self.animation_start_frame
        point_counter=0
        for point in points:
            x,y=point
            sphere=self.create_icosphere(
                Vector((x * self.spacing, y * self.spacing, 0))
            )
            #print(f"For {point_counter}, frame is {frame_counter}")
            self.animate_point(sphere,frame_counter)
            frame_counter+=self.animation_point_interval
            point_counter+=1
    
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
            start = points[0]+(0,)
            end = points[-1]+(0,)
            self.draw_line(
                start, end, 
                collection=self.grid_collection,
                animate=True, start_frame=10, duration=40)


my_graph=Graph2D(grid_size=50,animation_point_interval=2)
my_graph.draw_cartesian_grid()
#my_graph.erect_grid()
my_graph.plot_points([(1,2),(3,4)])
my_graph.plot_linear_equation(m=1,c=3)