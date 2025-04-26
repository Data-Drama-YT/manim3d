import bpy
import numpy as np
import sympy as sp
import bmesh
from mathutils import Vector, Matrix
import colorsys
import traceback
from math import *
from skimage.measure import marching_cubes


class Graph3D:
    def __init__(self,planes,grid_size,name="myGraph",grid_spacing=1,grid_thickness=0.01,point_size=0.05):
        """
        Args:
            planes: list of planes to render grids on, options are ['xy','xz','yz'].
            grid_size: Size of the grid in terms of sqaures
            grid_spacing: Space between each line. Defaults to 1.
            grid_thickness: Thickness of the lines in the grid. Defaults to 0.01
            point_size: Size of the points plotted on the graph. Defaults to 0.05
        """
        self.planes=planes
        self.grid_size=grid_size
        self.grid_spacing=grid_spacing
        self.name=name
        self.grid_thickness = grid_thickness
        self.grid_objects = {'xy': [], 'xz': [], 'yz': []}
        self.point_size=point_size

        # Main collection for this graph
        self.main_collection = self._create_collection(f"{name}_Collection")

        #Collection for all the points plotted on the graph
        self.points_collection = None

        #Collection for all 3d functions plotted
        self.functions_collection_3d = self._create_collection(f"{name}_3D_Functions_Collection")

    def draw_line(self, start, end, name="Line", collection=None, 
                  material=None, thickness=None, animate=False, 
                  start_frame=1, duration=20):
        """
        Draw a 3D line between two points
        
        Args:
            start: starting point (x,y,z)
            end: ending point (x,y,z)
            name: name for the line object
            collection: collection to add the line to
            material: material to apply to the line
            thickness: line thickness (overrides default)
            animate: whether to animate the line drawing
            start_frame: frame to start animation
            duration: duration of animation in frames
            
        Returns:
            The created curve object
        """
        # Ensure points are 3D
        start = Vector(start)
        end = Vector(end)
        
        if thickness is None:
            thickness = self.grid_thickness

        # Create curve data
        curve_data = bpy.data.curves.new(name=f"{name}_Curve", type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 2

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
            self.main_collection.objects.link(curve_obj)

        # Add thickness
        curve_data.bevel_depth = thickness
        curve_data.bevel_resolution = 3

        # Apply material if provided
        if material:
            if curve_obj.data.materials:
                curve_obj.data.materials[0] = material
            else:
                curve_obj.data.materials.append(material)

        if animate:
            # Set initial bevel factor
            curve_data.bevel_factor_start = 0.0
            curve_data.bevel_factor_end = 0.0
            curve_data.use_fill_caps = True
            curve_data.keyframe_insert(data_path="bevel_factor_end", frame=start_frame)

            # Animate to full draw
            curve_data.bevel_factor_end = 1.0
            curve_data.keyframe_insert(data_path="bevel_factor_end", frame=start_frame + duration)
            # Add easing

            if curve_obj.animation_data is None:
                curve_obj.animation_data_create()

            if curve_obj.animation_data.action is None:
                curve_obj.animation_data.action = bpy.data.actions.new(name=f"{curve_obj.name}_Action")

            for fc in curve_obj.animation_data.action.fcurves:
                for kfp in fc.keyframe_points:
                    kfp.interpolation = 'ELASTIC'

        return curve_obj
    
    def _create_collection(self, name, parent=None):
        """Create a new collection and link it appropriately"""
        if name in bpy.data.collections:
            collection = bpy.data.collections[name]
        else:
            collection = bpy.data.collections.new(name)
            
        if parent:
            # Check if collection is already linked to parent
            if collection.name not in [c.name for c in parent.children]:
                parent.children.link(collection)
        else:
            # Check if collection is already linked to scene collection
            if collection.name not in [c.name for c in bpy.context.scene.collection.children]:
                bpy.context.scene.collection.children.link(collection)
            
        return collection
    
    def _create_default_material(self, name, color=(0.8, 0.2, 0.2, 1.0)):
        """Create a default material with the given color"""
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
        
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        
        # Clear existing nodes
        for node in nodes:
            nodes.remove(node)
            
        # Create principled BSDF
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.inputs['Base Color'].default_value = color
        
        # In Blender 4.2.3, 'Specular' parameter is now 'Specular IOR Level'
        if 'Specular IOR Level' in bsdf.inputs:
            bsdf.inputs['Specular IOR Level'].default_value = 0.2
        elif 'Specular' in bsdf.inputs:
            bsdf.inputs['Specular'].default_value = 0.2
        
        # Set other parameters
        bsdf.inputs['Metallic'].default_value = 0.0
        bsdf.inputs['Roughness'].default_value = 0.3
        
        # Create output node
        output = nodes.new(type='ShaderNodeOutputMaterial')
        
        # Link nodes
        links = mat.node_tree.links
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        return mat

    def draw_grids(self, tick_labels=False,animate_lines=False):
        """
        Draw grid lines on specified planes
        
        Args:
            planes: list of planes to draw grids on. Can include 'xy', 'xz', 'yz'
            tick_labels: whether to add numeric labels at grid ticks
        """
        self.grid_collection = self._create_collection(f"{self.name}_Grid", self.main_collection)
        
        # Create grid materials
        xy_mat = self._create_default_material("XY_Grid_Material", color=(0.8, 0.8, 0.8, 0.3))
        xy_mat.blend_method = 'BLEND'
        
        xz_mat = self._create_default_material("XZ_Grid_Material", color=(0.8, 0.8, 0.8, 0.3))
        xz_mat.blend_method = 'BLEND'
        
        yz_mat = self._create_default_material("YZ_Grid_Material", color=(0.8, 0.8, 0.8, 0.3))
        yz_mat.blend_method = 'BLEND'
        
        half = self.grid_size // 2
        
        # Draw XY plane grid (parallel to ground)
        if 'xy' in self.planes:
            for x in range(-half, half + 1):
                x_pos = x * self.grid_spacing
                line = self.draw_line(
                    (x_pos, -half * self.grid_spacing, 0),
                    (x_pos, half * self.grid_spacing, 0),
                    name=f"XY_Grid_X{x}",
                    collection=self.grid_collection,
                    material=xy_mat,
                    animate=animate_lines
                )
                self.grid_objects['xy'].append(line)
                
                # Add tick label if requested
                if tick_labels and x % 2 == 0:
                    self._add_tick_label(str(x), (x_pos, -half * self.grid_spacing * 1.05, 0), f"X{x}_Label", xy_mat)

            for y in range(-half, half + 1):
                y_pos = y * self.grid_spacing
                line = self.draw_line(
                    (-half * self.grid_spacing, y_pos, 0),
                    (half * self.grid_spacing, y_pos, 0),
                    name=f"XY_Grid_Y{y}",
                    collection=self.grid_collection,
                    material=xy_mat,
                    animate=animate_lines
                )
                self.grid_objects['xy'].append(line)
                
                # Add tick label if requested
                if tick_labels and y % 2 == 0:
                    self._add_tick_label(str(y), (-half * self.grid_spacing * 1.05, y_pos, 0), f"Y{y}_Label", xy_mat)
        
        # Draw XZ plane grid (vertical, frontal)
        if 'xz' in self.planes:
            for x in range(-half, half + 1):
                x_pos = x * self.grid_spacing
                line = self.draw_line(
                    (x_pos, 0, -half * self.grid_spacing),
                    (x_pos, 0, half * self.grid_spacing),
                    name=f"XZ_Grid_X{x}",
                    collection=self.grid_collection,
                    material=xz_mat,
                    animate=animate_lines
                )
                self.grid_objects['xz'].append(line)

            for z in range(-half, half + 1):
                z_pos = z * self.grid_spacing
                line = self.draw_line(
                    (-half * self.grid_spacing, 0, z_pos),
                    (half * self.grid_spacing, 0, z_pos),
                    name=f"XZ_Grid_Z{z}",
                    collection=self.grid_collection,
                    material=xz_mat,
                    animate=animate_lines
                )
                self.grid_objects['xz'].append(line)
                
                # Add tick label if requested
                if tick_labels and z % 2 == 0:
                    self._add_tick_label(str(z), (-half * self.grid_spacing * 1.05, 0, z_pos), f"Z{z}_Label", xz_mat)
        
        # Draw YZ plane grid (vertical, side)
        if 'yz' in self.planes:
            for y in range(-half, half + 1):
                y_pos = y * self.grid_spacing
                line = self.draw_line(
                    (0, y_pos, -half * self.grid_spacing),
                    (0, y_pos, half * self.grid_spacing),
                    name=f"YZ_Grid_Y{y}",
                    collection=self.grid_collection,
                    material=yz_mat,
                    animate=animate_lines
                )
                self.grid_objects['yz'].append(line)

            for z in range(-half, half + 1):
                z_pos = z * self.grid_spacing
                line = self.draw_line(
                    (0, -half * self.grid_spacing, z_pos),
                    (0, half * self.grid_spacing, z_pos),
                    name=f"YZ_Grid_Z{z}",
                    collection=self.grid_collection,
                    material=yz_mat,
                    animate=animate_lines
                )
                self.grid_objects['yz'].append(line)
                
    def _add_tick_label(self, text, location, name, material):
        """Add a small numeric label at grid tick marks"""
        text_curve = bpy.data.curves.new(type="FONT", name=f"{name}_Curve")
        text_curve.body = text
        text_curve.size = self.grid_thickness * 8
        text_curve.align_x = 'CENTER'
        text_curve.align_y = 'CENTER'
        
        text_obj = bpy.data.objects.new(name, text_curve)
        text_obj.location = location
        
        # Apply material
        if text_obj.data.materials:
            text_obj.data.materials[0] = material
        else:
            text_obj.data.materials.append(material)
            
        # Link to collection
        self.grid_collection.objects.link(text_obj)
        
        # Always face the camera
        constraint = text_obj.constraints.new(type='COPY_ROTATION')
        constraint.target = bpy.context.scene.camera
        constraint.use_offset = True
        
        return text_obj
    
    def create_point(self, location, name="Point", color=None, size=None):
        """
        Create a point at the specified 3D location
        
        Args:
            location: (x, y, z) coordinates
            name: name for the point object
            color: color for the point (optional)
            size: size of the point (overrides default)
            
        Returns:
            The created point object
        """
        if size is None:
            size = self.point_size
            
            
        # Create points collection if it doesn't exist
        if self.points_collection is None:
            self.points_collection = self._create_collection(f"{self.name}_Points", self.main_collection)
            
        # Create material for this point
        point_mat = self._create_default_material(f"{name}_Material", color=(0.8, 0.8, 0.8, 0.3))
        
        # Create icosphere
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=2,
            radius=size,
            location=location
        )
        
        point = bpy.context.active_object
        point.name = name
        
        # Apply material
        if point.data.materials:
            point.data.materials[0] = point_mat
        else:
            point.data.materials.append(point_mat)
            
        # Link to points collection
        bpy.context.collection.objects.unlink(point)
        self.points_collection.objects.link(point)
        
        return point
    
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
        
        self.functions_collection_3d.objects.link(mesh_obj)
        
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

    

my_graph=Graph3D(planes=['xy','xz'],grid_size=10)
my_graph.draw_grids(animate_lines=True)
my_graph.draw_line((0,0,0),(5,5,0),"First Line",animate=True)
my_graph.create_point((0,0,5))

def upper_hemisphere(x, y, r=5):
    return np.sqrt(r**2 - x**2 - y**2)


my_graph.plot_function_3d(upper_hemisphere)