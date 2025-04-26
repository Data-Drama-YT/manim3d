"""
Graph3D - A Blender Python library for graph plotting and animation

This library provides tools to easily create, customize, and animate
mathematical graphs in Blender.
"""

import bpy
import bmesh
import numpy as np
import math
from mathutils import Vector, Matrix

class Graph3D:
    """Main class for the Graph3D library"""
    
    def __init__(self, scene=None):
        """Initialize Graph3D with optional scene"""
        self.scene = scene if scene else bpy.context.scene
        self.collections = {}
        self._setup_collections()
        
    def _setup_collections(self):
        """Create collections for organizing the graph elements"""
        main_collection = bpy.data.collections.new("Graph3D")
        bpy.context.scene.collection.children.link(main_collection)
        
        subcollections = ["Axes", "Grids", "Graphs", "Labels", "Text"]
        for name in subcollections:
            collection = bpy.data.collections.new(name)
            main_collection.children.link(collection)
            self.collections[name.lower()] = collection
    
    def create_cartesian_system(self, origin=(0, 0, 0), size=5, dimension=3, 
                                axis_material=None, grid=True, grid_material=None, 
                                labels=True, label_material=None):
        """
        Create a Cartesian coordinate system
        
        Parameters:
        -----------
        origin : tuple
            The origin point (x, y, z) of the coordinate system
        size : float
            The size of each axis
        dimension : int
            2 for 2D system, 3 for 3D system
        axis_material : Blender material
            Material for the axes
        grid : bool
            Whether to create grid lines
        grid_material : Blender material
            Material for the grid lines
        labels : bool
            Whether to create labels for axes
        label_material : Blender material
            Material for the labels
            
        Returns:
        --------
        dict : Dictionary containing references to created objects
        """
        # Create default materials if none provided
        if not axis_material:
            axis_material = self._create_default_material("AxisMaterial", (0.8, 0.8, 0.8, 1.0))
        
        if not grid_material and grid:
            grid_material = self._create_default_material("GridMaterial", (0.5, 0.5, 0.5, 0.3))
            
        if not label_material and labels:
            label_material = self._create_default_material("LabelMaterial", (1.0, 1.0, 1.0, 1.0))
        
        # Create axes
        axes = {}
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # RGB for XYZ
        for i, axis in enumerate(['x', 'y', 'z']):
            if dimension == 2 and axis == 'z':
                continue
                
            # Create axis line
            bm = bmesh.new()
            bmesh.ops.create_cone(bm, 
                                 segments=16, 
                                 radius1=0.02, 
                                 radius2=0.02, 
                                 depth=size)
            
            # Create arrow head
            arrow_head = bmesh.ops.create_cone(bm, 
                                              segments=16, 
                                              radius1=0.05, 
                                              radius2=0, 
                                              depth=0.2)
            
            # Translate arrow head to end of axis
            for v in arrow_head["verts"]:
                v.co.z += size/2
                
            # Rotate to correct orientation
            if axis == 'x':
                matrix_rot = Matrix.Rotation(math.radians(90), 3, 'Y')
            elif axis == 'y':
                matrix_rot = Matrix.Rotation(math.radians(-90), 3, 'X')
            else:  # z-axis
                matrix_rot = Matrix.Identity(4)
                
            bmesh.ops.transform(bm, matrix=matrix_rot, verts=bm.verts)
            
            # Create mesh and object
            mesh = bpy.data.meshes.new(f"{axis}_axis")
            bm.to_mesh(mesh)
            bm.free()
            
            obj = bpy.data.objects.new(f"{axis}_axis", mesh)
            obj.location = origin
            
            # Apply material with axis color
            axis_mat = axis_material.copy()
            axis_mat.name = f"{axis}_axis_material"
            axis_mat.diffuse_color = (*colors[i], 1.0)
            obj.data.materials.append(axis_mat)
            
            # Link to collection
            self.collections["axes"].objects.link(obj)
            axes[axis] = obj
        
        # Create grid if requested
        grid_objects = {}
        if grid:
            grid_size = size
            grid_subdivs = 10
            grid_spacing = grid_size / grid_subdivs
            
            planes = [('xy', 'z'), ('xz', 'y'), ('yz', 'x')]
            if dimension == 2:
                planes = [planes[0]]  # Only XY plane for 2D
                
            for plane_name, normal_axis in planes:
                grid_obj = self._create_grid(
                    plane_name, grid_size, grid_subdivs, 
                    origin, normal_axis, grid_material
                )
                grid_objects[plane_name] = grid_obj
        
        # Create labels if requested
        label_objects = {}
        if labels:
            # Implement label creation here
            for axis in ['x', 'y', 'z'][:dimension]:
                # Create tick marks and labels
                for i in range(-int(size), int(size) + 1):
                    if i == 0:  # Skip origin
                        continue
                    
                    # Create text label for tick
                    label_obj = self._create_text_label(
                        str(i), self._get_position_on_axis(origin, axis, i),
                        size=0.2, material=label_material
                    )
                    label_objects[f"{axis}_{i}"] = label_obj
        
        # Return all created objects
        return {
            "axes": axes,
            "grids": grid_objects,
            "labels": label_objects
        }
    
    def plot_function_2d(self, func, x_range=(-5, 5), samples=100, 
                        color=(0, 0.8, 0.2, 1.0), thickness=0.05, 
                        equation_text=None, material=None):
        """
        Plot a 2D function y = f(x)
        
        Parameters:
        -----------
        func : callable
            The function to plot, taking x as input and returning y
        x_range : tuple
            The range of x values (min, max)
        samples : int
            Number of sample points
        color : tuple
            RGBA color for the plot
        thickness : float
            Thickness of the plot line
        equation_text : str
            Optional text representing the equation to display
        material : Blender material
            Optional material for the plot
            
        Returns:
        --------
        dict : Dictionary containing references to created objects
        """
        if not material:
            material = self._create_default_material(f"Plot2DMaterial", color)
        
        # Generate points
        x_values = np.linspace(x_range[0], x_range[1], samples)
        points = [(x, func(x), 0) for x in x_values]
        
        # Create curve object
        curve_data = bpy.data.curves.new('plot_curve', type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 2
        
        # Create the path for the curve
        polyline = curve_data.splines.new('POLY')
        polyline.points.add(len(points) - 1)
        
        for i, point in enumerate(points):
            polyline.points[i].co = (*point, 1)
        
        # Create the curve object and link it
        curve_obj = bpy.data.objects.new("FunctionPlot", curve_data)
        curve_data.bevel_depth = thickness
        curve_data.bevel_resolution = 4
        curve_data.materials.append(material)
        
        self.collections["graphs"].objects.link(curve_obj)
        
        # Create equation text if provided
        text_obj = None
        if equation_text:
            text_obj = self._create_text_label(
                equation_text, 
                position=(0, 5, 5), 
                size=0.5, 
                material=material
            )
        
        return {
            "curve": curve_obj,
            "equation_text": text_obj
        }
    
    def plot_function_3d(self, func, x_range=(-5, 5), y_range=(-5, 5), 
                         samples=50, color=(0, 0.8, 0.2, 1.0), 
                         equation_text=None, material=None):
        """
        Plot a 3D function z = f(x, y)
        
        Parameters:
        -----------
        func : callable
            The function to plot, taking (x, y) as input and returning z
        x_range, y_range : tuple
            The ranges of x and y values (min, max)
        samples : int
            Number of sample points in each dimension
        color : tuple
            RGBA color for the plot
        equation_text : str
            Optional text representing the equation to display
        material : Blender material
            Optional material for the plot
            
        Returns:
        --------
        dict : Dictionary containing references to created objects
        """
        if not material:
            material = self._create_default_material(f"Plot3DMaterial", color)
        
        # Generate grid points
        x_values = np.linspace(x_range[0], x_range[1], samples)
        y_values = np.linspace(y_range[0], y_range[1], samples)
        
        # Create a mesh
        mesh = bpy.data.meshes.new("surface_mesh")
        bm = bmesh.new()
        
        # Create vertices
        vertices = []
        for y in y_values:
            for x in x_values:
                z = func(x, y)
                bm.verts.new((x, y, z))
                
        bm.verts.ensure_lookup_table()
        
        # Create faces
        for j in range(samples - 1):
            for i in range(samples - 1):
                v1 = bm.verts[j * samples + i]
                v2 = bm.verts[j * samples + i + 1]
                v3 = bm.verts[(j + 1) * samples + i + 1]
                v4 = bm.verts[(j + 1) * samples + i]
                bm.faces.new([v1, v2, v3, v4])
        
        # Create the mesh object
        bm.to_mesh(mesh)
        bm.free()
        
        mesh_obj = bpy.data.objects.new("SurfacePlot", mesh)
        mesh.materials.append(material)
        
        self.collections["graphs"].objects.link(mesh_obj)
        
        # Create equation text if provided
        text_obj = None
        if equation_text:
            text_obj = self._create_text_label(
                equation_text, 
                position=(0, 0, max([func(x, y) for x in x_values for y in y_values]) + 1), 
                size=0.5, 
                material=material
            )
        
        return {
            "surface": mesh_obj,
            "equation_text": text_obj
        }
    
    def plot_parametric_3d(self, x_func, y_func, z_func, t_range=(0, 2*np.pi), 
                          samples=100, color=(0.8, 0.2, 0.8, 1.0), 
                          thickness=0.05, equation_text=None, material=None):
        """
        Plot a 3D parametric curve
        
        Parameters:
        -----------
        x_func, y_func, z_func : callable
            Functions of parameter t returning x, y, z coordinates
        t_range : tuple
            The range of t values (min, max)
        samples : int
            Number of sample points
        color : tuple
            RGBA color for the plot
        thickness : float
            Thickness of the curve
        equation_text : str
            Optional text representing the equations
        material : Blender material
            Optional material for the plot
            
        Returns:
        --------
        dict : Dictionary containing references to created objects
        """
        if not material:
            material = self._create_default_material(f"ParametricMaterial", color)
        
        # Generate points
        t_values = np.linspace(t_range[0], t_range[1], samples)
        points = [(x_func(t), y_func(t), z_func(t)) for t in t_values]
        
        # Create curve object
        curve_data = bpy.data.curves.new('parametric_curve', type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 12
        
        # Create the path for the curve
        polyline = curve_data.splines.new('POLY')
        polyline.points.add(len(points) - 1)
        
        for i, point in enumerate(points):
            polyline.points[i].co = (*point, 1)
        
        # Create the curve object and link it
        curve_obj = bpy.data.objects.new("ParametricPlot", curve_data)
        curve_data.bevel_depth = thickness
        curve_data.bevel_resolution = 6
        curve_data.materials.append(material)
        
        self.collections["graphs"].objects.link(curve_obj)
        
        # Create equation text if provided
        text_obj = None
        if equation_text:
            text_obj = self._create_text_label(
                equation_text, 
                position=(0, 0, max([p[2] for p in points]) + 1), 
                size=0.5, 
                material=material
            )
        
        return {
            "curve": curve_obj,
            "equation_text": text_obj
        }
    
    def animate_function_2d(self, func, frames=100, start_frame=1, 
                           x_range=(-5, 5), samples=100, color=(0, 0.8, 0.2, 1.0),
                           thickness=0.05, equation_text=None, material=None, 
                           animation_type="grow", progress_func=None):
        """
        Animate a 2D function plot
        
        Parameters:
        -----------
        func : callable
            The function to plot, taking x as input and returning y
        frames : int
            Number of frames for the animation
        start_frame : int
            Starting frame number
        x_range : tuple
            The range of x values (min, max)
        samples : int
            Number of sample points
        color : tuple
            RGBA color for the plot
        thickness : float
            Thickness of the plot line
        equation_text : str
            Optional text representing the equation to display
        material : Blender material
            Optional material for the plot
        animation_type : str
            Type of animation ("grow", "evolve", "draw")
        progress_func : callable
            Optional function that modifies the animation progress
            
        Returns:
        --------
        dict : Dictionary containing references to created objects
        """
        if not material:
            material = self._create_default_material(f"AnimatedPlotMaterial", color)
        
        # Create empty curve for initial frame
        curve_data = bpy.data.curves.new('animated_curve', type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 2
        curve_data.bevel_depth = thickness
        curve_data.bevel_resolution = 4
        curve_data.materials.append(material)
        
        curve_obj = bpy.data.objects.new("AnimatedFunctionPlot", curve_data)
        self.collections["graphs"].objects.link(curve_obj)
        
        # Create text object if provided
        text_obj = None
        if equation_text:
            text_obj = self._create_text_label(
                equation_text, 
                position=(0, 5, 5), 
                size=0.5, 
                material=material
            )
        
        # Handle different animation types
        if animation_type == "grow":
            self._animate_growing_curve(
                curve_obj, func, x_range, samples, frames, start_frame, progress_func
            )
        elif animation_type == "evolve":
            self._animate_evolving_function(
                curve_obj, func, x_range, samples, frames, start_frame, progress_func
            )
        elif animation_type == "draw":
            self._animate_drawing_curve(
                curve_obj, func, x_range, samples, frames, start_frame, progress_func
            )
        
        # If there's equation text, animate its appearance
        if text_obj:
            self._animate_text_appearance(text_obj, start_frame, frames)
        
        return {
            "curve": curve_obj,
            "equation_text": text_obj
        }
    
    def animate_function_3d(self, func, frames=100, start_frame=1, 
                       x_range=(-5, 5), y_range=(-5, 5), samples=30, 
                       color=(0, 0.8, 0.2, 1.0), equation_text=None, 
                       material=None):
        """
        Animate a 3D function plot with the domain expanding uniformly over time
        """
        if not material:
            material = self._create_default_material(f"Animated3DMaterial", color)
        
        # Generate coordinates for mesh
        x_min, x_max = x_range
        y_min, y_max = y_range
        x_values = np.linspace(x_min, x_max, samples)
        y_values = np.linspace(y_min, y_max, samples)
        
        # Create initial flat mesh (all z=0)
        verts = []
        faces = []
        
        # Create vertices
        for j, y in enumerate(y_values):
            for i, x in enumerate(x_values):
                verts.append((x, y, 0))
        
        # Create faces
        for j in range(samples - 1):
            for i in range(samples - 1):
                v1 = j * samples + i
                v2 = j * samples + i + 1
                v3 = (j + 1) * samples + i + 1
                v4 = (j + 1) * samples + i
                faces.append([v1, v2, v3, v4])
        
        # Create mesh with initial vertices
        mesh = bpy.data.meshes.new("animated_surface")
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        mesh_obj = bpy.data.objects.new("AnimatedSurfacePlot", mesh)
        mesh.materials.append(material)
        self.collections["graphs"].objects.link(mesh_obj)
        
        # Create text object if provided
        text_obj = None
        if equation_text:
            text_obj = self._create_text_label(
                equation_text, 
                position=(0, 0, 10), 
                size=0.5, 
                material=material
            )
        
        # Create basis shape key (reference shape - flat plane)
        mesh_obj.shape_key_add(name="Basis")
        
        # Create all shape keys
        shape_keys = []
        for frame_idx in range(frames):
            t = frame_idx / (frames - 1)
            sk = mesh_obj.shape_key_add(name=f"Frame_{frame_idx}")
            shape_keys.append(sk)
            
            # Calculate progress for this frame (0 to 1)
            progress = t
            
            # Update vertex positions for this shape key
            for j, y in enumerate(y_values):
                for i, x in enumerate(x_values):
                    idx = j * samples + i
                    # Calculate normalized position in grid (0 to 1)
                    x_norm = (x - x_min) / (x_max - x_min)
                    y_norm = (y - y_min) / (y_max - y_min)
                    
                    # Only calculate z if this point should be "revealed" yet
                    if (x_norm + y_norm) / 2 <= progress:
                        z = func(x, y, t)
                    else:
                        z = 0
                    
                    sk.data[idx].co = (x, y, z)
        
        # Animate shape keys
        for frame_idx in range(frames):
            frame = start_frame + frame_idx
            bpy.context.scene.frame_set(frame)
            
            # Set all shape keys to 0 except the current one
            for i, sk in enumerate(shape_keys):
                sk.value = 1.0 if i == frame_idx else 0.0
                sk.keyframe_insert("value", frame=frame)
        
        # If there's equation text, animate its appearance
        if text_obj:
            self._animate_text_appearance(text_obj, start_frame, frames)
        
        return {
            "surface": mesh_obj,
            "equation_text": text_obj
        }
    
    def plot_implicit_surface(self, func, bounds=(-5, 5, -5, 5, -5, 5), 
                             resolution=50, color=(0.2, 0.8, 0.8, 1.0),
                             equation_text=None, material=None):
        """
        Plot an implicit surface defined by f(x,y,z) = 0
        
        Parameters:
        -----------
        func : callable
            Function taking (x,y,z) that defines the implicit surface
        bounds : tuple
            The x, y, z bounds as (xmin, xmax, ymin, ymax, zmin, zmax)
        resolution : int
            Resolution of the marching cubes algorithm
        color : tuple
            RGBA color for the surface
        equation_text : str
            Optional text representing the equation
        material : Blender material
            Optional material for the surface
            
        Returns:
        --------
        dict : Dictionary with references to created objects
        """
        if not material:
            material = self._create_default_material(f"ImplicitMaterial", color)
        
        # Create voxel grid
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        z = np.linspace(z_min, z_max, resolution)
        
        # Create a mesh using marching cubes
        # (This is a simplified implementation; a full marching cubes
        # algorithm would be more complex but produce better results)
        mesh = bpy.data.meshes.new("implicit_surface")
        bm = bmesh.new()
        
        # Sample some points on the surface
        # Note: This is an oversimplified approach for demonstration
        # A proper implementation would use marching cubes or similar
        for i in range(resolution):
            u = i / (resolution - 1)
            for j in range(resolution):
                v = j / (resolution - 1)
                
                # Parameterize the surface (simplified)
                x_val = x_min + u * (x_max - x_min)
                y_val = y_min + v * (y_max - y_min)
                
                # Find approximate z where func(x,y,z) = 0
                # Using binary search (simplified approach)
                z_lower, z_upper = z_min, z_max
                z_val = (z_lower + z_upper) / 2
                
                # Simple binary search for zero crossing
                for _ in range(10):  # 10 iterations for approximation
                    f_val = func(x_val, y_val, z_val)
                    if abs(f_val) < 0.01:
                        break
                    if f_val > 0:
                        z_upper = z_val
                    else:
                        z_lower = z_val
                    z_val = (z_lower + z_upper) / 2
                
                bm.verts.new((x_val, y_val, z_val))
        
        bm.verts.ensure_lookup_table()
        
        # Create a simple triangulation (not ideal but demonstrates the concept)
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                v1 = bm.verts[i * resolution + j]
                v2 = bm.verts[i * resolution + j + 1]
                v3 = bm.verts[(i+1) * resolution + j]
                v4 = bm.verts[(i+1) * resolution + j + 1]
                
                if (i + j) % 2 == 0:
                    bm.faces.new([v1, v2, v4])
                    bm.faces.new([v1, v4, v3])
                else:
                    bm.faces.new([v1, v2, v3])
                    bm.faces.new([v2, v4, v3])
        
        bm.to_mesh(mesh)
        bm.free()
        
        # Smooth the mesh
        for p in mesh.polygons:
            p.use_smooth = True
        
        mesh_obj = bpy.data.objects.new("ImplicitSurface", mesh)
        mesh.materials.append(material)
        self.collections["graphs"].objects.link(mesh_obj)
        
        # Create equation text if provided
        text_obj = None
        if equation_text:
            text_obj = self._create_text_label(
                equation_text, 
                position=(0, 0, z_max + 1), 
                size=0.5, 
                material=material
            )
        
        return {
            "surface": mesh_obj,
            "equation_text": text_obj
        }
    
    # Helper methods
    def _create_default_material(self, name, color):
        """Create a default material with given color"""
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        
        # Get the principled BSDF
        principled = mat.node_tree.nodes.get('Principled BSDF')
        if principled:
            principled.inputs['Base Color'].default_value = color
            principled.inputs['Roughness'].default_value = 0.4
            if 'Specular IOR Level' in principled.inputs:
                principled.inputs['Specular IOR Level'].default_value = 0.5
            elif 'Specular' in principled.inputs:
                principled.inputs['Specular'].default_value = 0.5
            
            # Handle transparency if alpha < 1
            if color[3] < 1.0:
                mat.blend_method = 'BLEND'
                principled.inputs['Alpha'].default_value = color[3]
        
        return mat
    
    def _create_grid(self, plane_name, size, subdivs, origin, normal_axis, material):
        """Create a grid on a specified plane"""
        # Create a mesh grid
        bm = bmesh.new()
        
        # Create a grid of vertices
        spacing = size / subdivs
        half_size = size / 2
        
        vertices = {}
        for i in range(-subdivs, subdivs + 1):
            for j in range(-subdivs, subdivs + 1):
                x, y, z = origin
                
                if plane_name == 'xy':
                    x += i * spacing
                    y += j * spacing
                elif plane_name == 'xz':
                    x += i * spacing
                    z += j * spacing
                elif plane_name == 'yz':
                    y += i * spacing
                    z += j * spacing
                    
                vertices[(i, j)] = bm.verts.new((x, y, z))
        
        # Create edges
        for i in range(-subdivs, subdivs + 1):
            for j in range(-subdivs, subdivs):
                if plane_name == 'xy':
                    # Horizontal lines (constant y)
                    bm.edges.new([vertices[(i, j)], vertices[(i, j+1)]])
                    # Vertical lines (constant x)
                    bm.edges.new([vertices[(j, i)], vertices[(j+1, i)]])
                elif plane_name == 'xz':
                    # Lines along x (constant z)
                    bm.edges.new([vertices[(i, j)], vertices[(i, j+1)]])
                    # Lines along z (constant x)
                    bm.edges.new([vertices[(j, i)], vertices[(j+1, i)]])
                elif plane_name == 'yz':
                    # Lines along y (constant z)
                    bm.edges.new([vertices[(i, j)], vertices[(i, j+1)]])
                    # Lines along z (constant y)
                    bm.edges.new([vertices[(j, i)], vertices[(j+1, i)]])
        
        # Create mesh
        mesh = bpy.data.meshes.new(f"{plane_name}_grid")
        bm.to_mesh(mesh)
        bm.free()
        
        obj = bpy.data.objects.new(f"{plane_name}_grid", mesh)
        obj.data.materials.append(material)
        
        self.collections["grids"].objects.link(obj)
        return obj
    
    def _create_text_label(self, text, position, size=0.5, material=None):
        """Create a 3D text object"""
        if not material:
            material = self._create_default_material("TextMaterial", (1, 1, 1, 1))
        
        # Create text curve data
        curve = bpy.data.curves.new(name=f"Text_{text}", type='FONT')
        curve.body = text
        curve.size = size
        curve.align_x = 'CENTER'
        curve.align_y = 'CENTER'
        
        # Create object
        text_obj = bpy.data.objects.new(name=f"Text_{text}", object_data=curve)
        text_obj.location = position
        text_obj.data.materials.append(material)
        
        self.collections["text"].objects.link(text_obj)
        return text_obj
    
    def _get_position_on_axis(self, origin, axis, value):
        """Get position along specified axis"""
        x, y, z = origin
        if axis == 'x':
            return (x + value, y, z)
        elif axis == 'y':
            return (x, y + value, z)
        else:  # z-axis
            return (x, y, z + value)
    
    def _animate_growing_curve(self, curve_obj, func, x_range, samples, frames, start_frame, progress_func=None):
        """Animate a curve by progressively increasing its domain"""
        x_min, x_max = x_range
        x_values = np.linspace(x_min, x_max, samples)
        
        # Create and keyframe the curve
        for frame in range(frames):
            bpy.context.scene.frame_set(start_frame + frame)
            
            # Calculate progress (0 to 1)
            progress = frame / (frames - 1)
            if progress_func:
                progress = progress_func(progress)
            
            # Calculate current domain endpoint
            current_max_idx = int(progress * samples)
            if current_max_idx < 2:
                current_max_idx = 2  # Need at least 2 points
            
            current_points = [(x, func(x), 0) for x in x_values[:current_max_idx]]
            
            # Update the curve
            curve_data = curve_obj.data
            
            # Clear existing splines
            curve_data.splines.clear()
            
            # Create new spline with current points
            polyline = curve_data.splines.new('POLY')
            polyline.points.add(len(current_points) - 1)
            
            for i, point in enumerate(current_points):
                polyline.points[i].co = (*point, 1)
            
            # Insert keyframe
            curve_obj.keyframe_insert("location")
    
    def _animate_evolving_function(self, curve_obj, func, x_range, samples, frames, start_frame, progress_func=None):
        """Animate a curve by smoothly transitioning between function states"""
        x_min, x_max = x_range
        x_values = np.linspace(x_min, x_max, samples)
        
        # Create and keyframe the curve
        for frame in range(frames):
            bpy.context.scene.frame_set(start_frame + frame)
            
            # Calculate progress (0 to 1)
            progress = frame / (frames - 1)
            if progress_func:
                progress = progress_func(progress)
            
            # Wrapper function to evolve over time
            def time_variant_func(x):
                return func(x, progress)
            
            # Calculate points for current frame
            current_points = [(x, time_variant_func(x), 0) for x in x_values]
            
            # Update the curve
            curve_data = curve_obj.data
            
            # Clear existing splines
            curve_data.splines.clear()
            
            # Create new spline with current points
            polyline = curve_data.splines.new('POLY')
            polyline.points.add(len(current_points) - 1)
            
            for i, point in enumerate(current_points):
                polyline.points[i].co = (*point, 1)
            
            # Insert keyframe
            curve_obj.keyframe_insert("location")

    def _animate_drawing_curve(self, curve_obj, func, x_range, samples, frames, start_frame, progress_func=None):
        """Animate a curve as if it's being drawn, with a trailing effect"""
        x_min, x_max = x_range
        x_values = np.linspace(x_min, x_max, samples)
        y_values = [func(x) for x in x_values]
        
        window_size = int(samples / 5)  # Size of the visible window
        
        # Create and keyframe the curve
        for frame in range(frames):
            bpy.context.scene.frame_set(start_frame + frame)
            
            # Calculate progress (0 to 1)
            progress = frame / (frames - 1)
            if progress_func:
                progress = progress_func(progress)
            
            # Calculate current center point index
            center_idx = int(progress * samples)
            
            # Calculate window start and end
            start_idx = max(0, center_idx - window_size // 2)
            end_idx = min(samples, center_idx + window_size // 2)
            
            # Get current points in the window
            current_points = [(x_values[i], y_values[i], 0) for i in range(start_idx, end_idx)]
            
            # Update the curve
            curve_data = curve_obj.data
            
            # Clear existing splines
            curve_data.splines.clear()
            
            # Create new spline with current points
            polyline = curve_data.splines.new('POLY')
            polyline.points.add(len(current_points) - 1)
            
            for i, point in enumerate(current_points):
                polyline.points[i].co = (*point, 1)
            
            # Insert keyframe
            curve_obj.keyframe_insert("location")
    
    
    def _animate_growing_surface(self, mesh_obj, func, x_range, y_range, samples, frames, start_frame):
        """Animate a surface by gradually revealing it from the center outward"""
        """Animate a surface by gradually revealing it from the center outward"""
        """Animate a surface by gradually revealing it from the center outward"""
        x_min, x_max = x_range
        y_min, y_max = y_range
        x_values = np.linspace(x_min, x_max, samples)
        y_values = np.linspace(y_min, y_max, samples)
        
        # --- 1. Create the base mesh (flat)
        # First check if mesh_obj is valid and has mesh data
        if mesh_obj is None or not hasattr(mesh_obj, 'data') or mesh_obj.data is None:
            # Create a new mesh data block
            mesh_data = bpy.data.meshes.new(name="GrowingSurface")
            # Assign it to the mesh_obj if it exists
            if mesh_obj is not None:
                mesh_obj.data = mesh_data
            else:
                # Create a new object with the mesh data
                mesh_obj = bpy.data.objects.new("GrowingSurface", mesh_data)
                # Link it to the current scene
                bpy.context.collection.objects.link(mesh_obj)
        
        # Clear existing mesh data
        if len(mesh_obj.data.vertices) > 0:
            bpy.ops.object.select_all(action='DESELECT')
            mesh_obj.select_set(True)
            bpy.context.view_layer.objects.active = mesh_obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.delete(type='VERT')
            bpy.ops.object.mode_set(mode='OBJECT')
        
        # Now create the mesh using bmesh
        bm = bmesh.new()
        vert_grid = []
        for j, y in enumerate(y_values):
            row = []
            for i, x in enumerate(x_values):
                v = bm.verts.new((x, y, 0))
                row.append(v)
            vert_grid.append(row)
        
        for j in range(samples - 1):
            for i in range(samples - 1):
                v1 = vert_grid[j][i]
                v2 = vert_grid[j][i+1]
                v3 = vert_grid[j+1][i+1]
                v4 = vert_grid[j+1][i]
                bm.faces.new([v1, v2, v3, v4])
        
        # Make sure we have a valid mesh data object
        bm.to_mesh(mesh_obj.data)
        bm.free()
        
        # Update the mesh
        mesh_obj.data.update()
        
        # --- 2. Create animation
        # Calculate center and max distance
        center_i = samples // 2
        center_j = samples // 2
        max_dist = math.sqrt((samples-1)**2 + (samples-1)**2)
        
        # Create animation data if it doesn't exist
        if not mesh_obj.animation_data:
            mesh_obj.animation_data_create()
        
        # Create a new action or use existing one
        action_name = f"{mesh_obj.name}_grow_action"
        if action_name in bpy.data.actions:
            action = bpy.data.actions[action_name]
        else:
            action = bpy.data.actions.new(action_name)
        
        mesh_obj.animation_data.action = action
        
        # --- 3. Add a custom property to control the growth
        mesh_obj["grow_factor"] = 0.0
        
        # Set up animation for the custom property
        for frame_num in range(frames):
            current_frame = start_frame + frame_num
            bpy.context.scene.frame_set(current_frame)
            
            # Calculate growth factor (0 to 1)
            mesh_obj["grow_factor"] = frame_num / (frames - 1)
            
            # Keyframe the custom property
            mesh_obj.keyframe_insert(data_path='["grow_factor"]', frame=current_frame)
        
        # --- 4. Add drivers to vertices
        # Store final heights for each vertex
        vertex_heights = {}
        
        # Calculate final heights for each vertex
        for j in range(samples):
            for i in range(samples):
                index = j * samples + i
                x = x_values[i]
                y = y_values[j]
                final_z = func(x, y)
                dist = math.sqrt((i - center_i)**2 + (j - center_j)**2)
                normalized_dist = dist / (max_dist * 1.2)  # 1.2 to ensure full coverage
                
                # Save the data for this vertex
                vertex_heights[index] = (final_z, normalized_dist)
        
        # Make sure we're in object mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Add driver to each vertex
        for index, (final_z, normalized_dist) in vertex_heights.items():
            if index < len(mesh_obj.data.vertices):
                # Create a new driver for this vertex's z position
                fcurve = mesh_obj.data.vertices[index].driver_add("co", 2)
                driver = fcurve.driver
                driver.type = 'SCRIPTED'
                
                # Add variable to the driver
                var = driver.variables.new()
                var.name = "grow"
                var.type = 'SINGLE_PROP'
                var.targets[0].id = mesh_obj
                var.targets[0].data_path = '["grow_factor"]'
                
                # Set up the driver expression
                driver.expression = f"grow > {normalized_dist} ? {final_z} : 0"
        
        # Return the mesh object for further use
        return mesh_obj

    
    def _animate_evolving_surface(self, mesh_obj, func, x_range, y_range, samples, frames, start_frame):
        """Animate a surface by smoothly transitioning between function states"""
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        # Generate base points
        x_values = np.linspace(x_min, x_max, samples)
        y_values = np.linspace(y_min, y_max, samples)
        
        for frame in range(frames):
            bpy.context.scene.frame_set(start_frame + frame)
            
            # Calculate progress (0 to 1)
            progress = frame / (frames - 1)
            
            # Create new mesh for this frame
            bm = bmesh.new()
            
            # Create vertices with time-variant function
            vertices = {}
            for j, y in enumerate(y_values):
                for i, x in enumerate(x_values):
                    z = func(x, y, progress)  # Time-variant function
                    vertices[(i, j)] = bm.verts.new((x, y, z))
            
            # Create faces
            for j in range(samples - 1):
                for i in range(samples - 1):
                    v1 = vertices[(i, j)]
                    v2 = vertices[(i+1, j)]
                    v3 = vertices[(i+1, j+1)]
                    v4 = vertices[(i, j+1)]
                    bm.faces.new([v1, v2, v3, v4])
            
            # Update the mesh
            bm.to_mesh(mesh_obj.data)
            bm.free()
            
            # Insert keyframe
            mesh_obj.keyframe_insert("location")
    
    def _animate_extruding_surface(self, mesh_obj, func, x_range, y_range, samples, frames, start_frame):
        """Animate a surface by extruding it from the xy-plane"""
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        # Generate base points
        x_values = np.linspace(x_min, x_max, samples)
        y_values = np.linspace(y_min, y_max, samples)
        
        # Calculate z values for final surface
        z_values = np.array([[func(x, y) for x in x_values] for y in y_values])
        
        for frame in range(frames):
            bpy.context.scene.frame_set(start_frame + frame)
            
            # Calculate progress (0 to 1)
            progress = frame / (frames - 1)
            
            # Create new mesh for this frame
            bm = bmesh.new()
            
            # Create vertices with partial extrusion
            vertices = {}
            for j, y in enumerate(y_values):
                for i, x in enumerate(x_values):
                    # Scale z by progress
                    z = progress * z_values[j][i]
                    vertices[(i, j)] = bm.verts.new((x, y, z))
            
            # Create faces
            for j in range(samples - 1):
                for i in range(samples - 1):
                    v1 = vertices[(i, j)]
                    v2 = vertices[(i+1, j)]
                    v3 = vertices[(i+1, j+1)]
                    v4 = vertices[(i, j+1)]
                    bm.faces.new([v1, v2, v3, v4])
            
            # Update the mesh
            bm.to_mesh(mesh_obj.data)
            bm.free()
            
            # Insert keyframe
            mesh_obj.keyframe_insert("location")
    
    def _animate_text_appearance(self, text_obj, start_frame, frames):
        """Animate the appearance of a text object"""
        # Hide text initially
        text_obj.hide_render = True
        text_obj.hide_viewport = True
        text_obj.keyframe_insert("hide_render", frame=start_frame)
        text_obj.keyframe_insert("hide_viewport", frame=start_frame)
        
        # Show text after 25% of the animation
        reveal_frame = start_frame + int(frames * 0.25)
        text_obj.hide_render = False
        text_obj.hide_viewport = False
        text_obj.keyframe_insert("hide_render", frame=reveal_frame)
        text_obj.keyframe_insert("hide_viewport", frame=reveal_frame)
        
        # Animate scale
        text_obj.scale = (0.01, 0.01, 0.01)
        text_obj.keyframe_insert("scale", frame=reveal_frame)
        
        # Grow to full size
        grow_frame = reveal_frame + int(frames * 0.15)
        text_obj.scale = (1.0, 1.0, 1.0)
        text_obj.keyframe_insert("scale", frame=grow_frame)
        
        # Add some subtle rotation
        text_obj.rotation_euler = (0, 0, 0)
        text_obj.keyframe_insert("rotation_euler", frame=reveal_frame)
        
        text_obj.rotation_euler = (0, 0, 0.05)
        text_obj.keyframe_insert("rotation_euler", frame=grow_frame)
        
        text_obj.rotation_euler = (0, 0, 0)
        text_obj.keyframe_insert("rotation_euler", frame=grow_frame + int(frames * 0.1))

# High-level convenience API for common graph functions
class GraphPlotter:
    """High-level API for common mathematical functions"""
    
    def __init__(self, blender_graphs=None):
        """Initialize with Graph3D instance"""
        self.bg = blender_graphs if blender_graphs else Graph3D()
    
    def setup_cartesian_system(self, dimension=3, size=5, grid=True):
        """Setup a standard Cartesian coordinate system"""
        return self.bg.create_cartesian_system(
            dimension=dimension,
            size=size,
            grid=grid
        )
    
    def sine_wave(self, amplitude=1, frequency=1, phase=0, 
                 x_range=(-5, 5), color=(0, 0.8, 0.2, 1.0), with_text=True):
        """Plot a sine wave: y = A * sin(ω * x + φ)"""
        def func(x):
            return amplitude * math.sin(frequency * x + phase)
        
        equation = f"y = {amplitude} sin({frequency}x + {phase})"
        
        return self.bg.plot_function_2d(
            func=func,
            x_range=x_range,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def cosine_wave(self, amplitude=1, frequency=1, phase=0, 
                   x_range=(-5, 5), color=(0, 0.2, 0.8, 1.0), with_text=True):
        """Plot a cosine wave: y = A * cos(ω * x + φ)"""
        def func(x):
            return amplitude * math.cos(frequency * x + phase)
        
        equation = f"y = {amplitude} cos({frequency}x + {phase})"
        
        return self.bg.plot_function_2d(
            func=func,
            x_range=x_range,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def parabola(self, a=1, b=0, c=0, 
                x_range=(-5, 5), color=(0.8, 0.2, 0.0, 1.0), with_text=True):
        """Plot a parabola: y = a*x² + b*x + c"""
        def func(x):
            return a * x**2 + b * x + c
        
        equation = f"y = {a}x² + {b}x + {c}"
        
        return self.bg.plot_function_2d(
            func=func,
            x_range=x_range,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def line(self, slope=1, intercept=0, 
            x_range=(-5, 5), color=(0.5, 0.5, 0.5, 1.0), with_text=True):
        """Plot a straight line: y = mx + b"""
        def func(x):
            return slope * x + intercept
        
        equation = f"y = {slope}x + {intercept}"
        
        return self.bg.plot_function_2d(
            func=func,
            x_range=x_range,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def exponential(self, a=1, b=2, 
                   x_range=(-5, 5), color=(0.8, 0.5, 0.0, 1.0), with_text=True):
        """Plot an exponential function: y = a * b^x"""
        def func(x):
            return a * b**x
        
        equation = f"y = {a} · {b}^x"
        
        return self.bg.plot_function_2d(
            func=func,
            x_range=x_range,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def logarithm(self, base=10, scale=1, 
                 x_range=(0.1, 5), color=(0.0, 0.5, 0.8, 1.0), with_text=True):
        """Plot a logarithmic function: y = scale * log_base(x)"""
        def func(x):
            # Handle domain issues
            if x <= 0:
                return float('nan')
            return scale * math.log(x, base)
        
        equation = f"y = {scale} · log_{base}(x)"
        
        return self.bg.plot_function_2d(
            func=func,
            x_range=x_range,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def circle(self, radius=3, center=(0, 0, 0), 
              color=(0.8, 0.2, 0.8, 1.0), with_text=True):
        """Plot a circle using parametric equations"""
        def x_func(t):
            return center[0] + radius * math.cos(t)
        
        def y_func(t):
            return center[1] + radius * math.sin(t)
        
        def z_func(t):
            return center[2]
        
        equation = f"(x - {center[0]})² + (y - {center[1]})² = {radius}²"
        
        return self.bg.plot_parametric_3d(
            x_func=x_func,
            y_func=y_func,
            z_func=z_func,
            t_range=(0, 2*math.pi),
            color=color,
            equation_text=equation if with_text else None
        )
    
    def helix(self, radius=3, pitch=0.5, num_turns=3, 
             color=(0.2, 0.8, 0.6, 1.0), with_text=True):
        """Plot a 3D helix"""
        def x_func(t):
            return radius * math.cos(t)
        
        def y_func(t):
            return radius * math.sin(t)
        
        def z_func(t):
            return pitch * t / (2*math.pi)
        
        equation = "x = r·cos(t), y = r·sin(t), z = p·t/(2π)"
        
        return self.bg.plot_parametric_3d(
            x_func=x_func,
            y_func=y_func,
            z_func=z_func,
            t_range=(0, 2*math.pi*num_turns),
            samples=100*num_turns,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def paraboloid(self, a=0.25, b=0.25, 
                  x_range=(-5, 5), y_range=(-5, 5), 
                  color=(0.2, 0.7, 0.3, 1.0), with_text=True):
        """Plot a paraboloid: z = a*x² + b*y²"""
        def func(x, y):
            return a * x**2 + b * y**2
        
        equation = f"z = {a}x² + {b}y²"
        
        return self.bg.plot_function_3d(
            func=func,
            x_range=x_range,
            y_range=y_range,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def sinc_function(self, x_range=(-10, 10), y_range=(-10, 10), 
                     color=(0.1, 0.6, 0.9, 1.0), with_text=True):
        """Plot the 2D sinc function: z = sin(√(x² + y²)) / √(x² + y²)"""
        def func(x, y):
            r = math.sqrt(x**2 + y**2)
            if r < 0.001:
                return 1.0  # Limit as r approaches 0
            return math.sin(r) / r
        
        equation = "z = sin(√(x² + y²)) / √(x² + y²)"
        
        return self.bg.plot_function_3d(
            func=func,
            x_range=x_range,
            y_range=y_range,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def wave_interference(self, x_range=(-10, 10), y_range=(-10, 10), 
                         sources=2, frequency=1, 
                         color=(0.3, 0.3, 0.9, 1.0), with_text=True):
        """Plot wave interference pattern from point sources"""
        def func(x, y):
            result = 0
            # Create several point sources arranged in a circle
            for i in range(sources):
                angle = 2 * math.pi * i / sources
                source_x = 5 * math.cos(angle)
                source_y = 5 * math.sin(angle)
                
                # Distance from point to source
                r = math.sqrt((x - source_x)**2 + (y - source_y)**2)
                
                # Add wave contribution (decaying with distance)
                result += math.sin(frequency * r) / max(1, r**0.5)
            
            return result
        
        equation = f"{sources} point sources wave interference"
        
        return self.bg.plot_function_3d(
            func=func,
            x_range=x_range,
            y_range=y_range,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def sphere(self, radius=3, center=(0, 0, 0), 
              resolution=50, color=(0.1, 0.5, 0.9, 1.0), with_text=True):
        """Plot a sphere using implicit surface"""
        def func(x, y, z):
            return (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 - radius**2
        
        equation = f"(x - {center[0]})² + (y - {center[1]})² + (z - {center[2]})² = {radius}²"
        
        bounds = (
            center[0] - radius*1.2, center[0] + radius*1.2,
            center[1] - radius*1.2, center[1] + radius*1.2,
            center[2] - radius*1.2, center[2] + radius*1.2
        )
        
        return self.bg.plot_implicit_surface(
            func=func,
            bounds=bounds,
            resolution=resolution,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def animate_sine_evolve(self, frames=100, start_frame=1, frequency_range=(0.5, 3),
                           x_range=(-5, 5), color=(0, 0.8, 0.2, 1.0), with_text=True):
        """Animate an evolving sine wave with changing frequency"""
        def func(x, progress):
            # Frequency evolves from min to max
            current_freq = frequency_range[0] + progress * (frequency_range[1] - frequency_range[0])
            return math.sin(current_freq * x)
        
        equation = "y = sin(ωx), ω evolving"
        
        return self.bg.animate_function_2d(
            func=func,
            frames=frames,
            start_frame=start_frame,
            x_range=x_range,
            color=color,
            equation_text=equation if with_text else None,
            animation_type="evolve"
        )
    
    def animate_3d_wave(self, frames=100, start_frame=1, 
                       x_range=(-5, 5), y_range=(-5, 5), 
                       color=(0.2, 0.6, 0, 1.0), with_text=True):
        """Animate a 3D wave propagation"""
        def func(x, y, progress=1):
            time = progress * 5  # Scale time for animation effect
            r = math.sqrt(x**2 + y**2)
            return 2 * math.sin(r - time) / max(1, r**0.7)
        
        equation = "z = sin(√(x² + y²) - t)"
        
        return self.bg.animate_function_3d(func)


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
    def parabola_func(x, y, t):
        """Demo function: An expanding parabolic surface"""
        # Scale the parabola with time (t goes from 0 to 1)
        amplitude = 2 * t  # Height grows over time
        spread = 0.5 + 1.5 * t  # Width expands over time
        
        # Create a circular parabola
        r_squared = x**2 + y**2
        return amplitude * (1 - r_squared/(spread**2))
    
    parabola_animation=bg.animate_function_3d(parabola_func,x_range=(-2,2),y_range=(-2,2))
    # Example 5: Implicit surface
    #sphere = plotter.sphere(radius=3, center=(0, 0, 6))
    
    return {
        "coordinate_system": coord_system,
    }

create_examples()