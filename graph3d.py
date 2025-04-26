import bpy
import numpy as np
import sympy as sp
from mathutils import Vector, Matrix
import colorsys
import traceback
from math import *
from skimage.measure import marching_cubes


class Graph3D:
    def __init__(self,planes,grid_size,name="myGraph",grid_spacing=1,grid_thickness=0.01):
        """
        Args:
            planes: list of planes to render grids on, options are ['xy','xz','yz'].
            grid_size: Size of the grid in terms of sqaures
            grid_spacing: Space between each line. Defaults to 1.
        """
        self.planes=planes
        self.grid_size=grid_size
        self.grid_spacing=grid_spacing
        self.name=name
        self.grid_thickness = grid_thickness
        self.grid_objects = {'xy': [], 'xz': [], 'yz': []}

        # Main collection for this graph
        self.main_collection = self._create_collection(f"{name}_Collection")

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

    def draw_grids(self, tick_labels=False):
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
                if x == 0:  # Skip the axis line, it's drawn separately
                    continue
                x_pos = x * self.grid_spacing
                line = self.draw_line(
                    (x_pos, -half * self.grid_spacing, 0),
                    (x_pos, half * self.grid_spacing, 0),
                    name=f"XY_Grid_X{x}",
                    collection=self.grid_collection,
                    material=xy_mat
                )
                self.grid_objects['xy'].append(line)
                
                # Add tick label if requested
                if tick_labels and x % 2 == 0:
                    self._add_tick_label(str(x), (x_pos, -half * self.grid_spacing * 1.05, 0), f"X{x}_Label", xy_mat)

            for y in range(-half, half + 1):
                if y == 0:  # Skip the axis line
                    continue
                y_pos = y * self.grid_spacing
                line = self.draw_line(
                    (-half * self.grid_spacing, y_pos, 0),
                    (half * self.grid_spacing, y_pos, 0),
                    name=f"XY_Grid_Y{y}",
                    collection=self.grid_collection,
                    material=xy_mat
                )
                self.grid_objects['xy'].append(line)
                
                # Add tick label if requested
                if tick_labels and y % 2 == 0:
                    self._add_tick_label(str(y), (-half * self.grid_spacing * 1.05, y_pos, 0), f"Y{y}_Label", xy_mat)
        
        # Draw XZ plane grid (vertical, frontal)
        if 'xz' in self.planes:
            for x in range(-half, half + 1):
                if x == 0:  # Skip the axis line
                    continue
                x_pos = x * self.grid_spacing
                line = self.draw_line(
                    (x_pos, 0, -half * self.grid_spacing),
                    (x_pos, 0, half * self.grid_spacing),
                    name=f"XZ_Grid_X{x}",
                    collection=self.grid_collection,
                    material=xz_mat
                )
                self.grid_objects['xz'].append(line)

            for z in range(-half, half + 1):
                if z == 0:  # Skip the axis line
                    continue
                z_pos = z * self.grid_spacing
                line = self.draw_line(
                    (-half * self.grid_spacing, 0, z_pos),
                    (half * self.grid_spacing, 0, z_pos),
                    name=f"XZ_Grid_Z{z}",
                    collection=self.grid_collection,
                    material=xz_mat
                )
                self.grid_objects['xz'].append(line)
                
                # Add tick label if requested
                if tick_labels and z % 2 == 0:
                    self._add_tick_label(str(z), (-half * self.grid_spacing * 1.05, 0, z_pos), f"Z{z}_Label", xz_mat)
        
        # Draw YZ plane grid (vertical, side)
        if 'yz' in self.planes:
            for y in range(-half, half + 1):
                if y == 0:  # Skip the axis line
                    continue
                y_pos = y * self.grid_spacing
                line = self.draw_line(
                    (0, y_pos, -half * self.grid_spacing),
                    (0, y_pos, half * self.grid_spacing),
                    name=f"YZ_Grid_Y{y}",
                    collection=self.grid_collection,
                    material=yz_mat
                )
                self.grid_objects['yz'].append(line)

            for z in range(-half, half + 1):
                if z == 0:  # Skip the axis line
                    continue
                z_pos = z * self.grid_spacing
                line = self.draw_line(
                    (0, -half * self.grid_spacing, z_pos),
                    (0, half * self.grid_spacing, z_pos),
                    name=f"YZ_Grid_Z{z}",
                    collection=self.grid_collection,
                    material=yz_mat
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

    

my_graph=Graph3D(planes=['xy','xz'],grid_size=10)
my_graph.draw_grids()
my_graph.draw_line((0,0,0),(5,5,0),"First Line",animate=True)