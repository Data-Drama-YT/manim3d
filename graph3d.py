import bpy
import numpy as np
import sympy as sp
from mathutils import Vector, Matrix
import colorsys
import traceback
from math import *
from skimage.measure import marching_cubes

class Graph3D:
    def __init__(self, points=[], grid_size=10, spacing=1,
                 name="Graph3D", material=None,
                 animation_start_frame=0,
                 animation_point_interval=1,
                 axis_thickness=0.02,
                 grid_thickness=0.01,
                 point_size=0.1,
                 curve_thickness=0.03):
        """
        Initialize a 3D graphing environment
        
        Args:
            points: list of (x,y,z) tuples for initial points
            grid_size: size of the grid (determines range)
            spacing: spacing between grid lines
            name: base name for the graph objects
            material: Blender material to apply (optional)
            animation_start_frame: frame to start animations
            animation_point_interval: frames between point animations
            axis_thickness: thickness of the coordinate axes
            grid_thickness: thickness of the grid lines
            point_size: size of point markers
            curve_thickness: thickness of function curves
        """
        # Store all points in 3D format
        self.points = [Vector(p) if len(p) == 3 else Vector((p[0], p[1], 0)) for p in points]
        self.animation_start_frame = animation_start_frame
        self.animation_point_interval = animation_point_interval
        self.name = name
        self.grid_size = grid_size
        self.spacing = spacing
        self.material = material
        self.axis_thickness = axis_thickness
        self.grid_thickness = grid_thickness
        self.point_size = point_size
        self.curve_thickness = curve_thickness
        
        # Create main collection for all graph objects
        self.main_collection = self._create_collection(f"{name}_Collection")
        self.grid_collection = None
        self.axis_collection = None
        self.points_collection = None
        self.curves_collection = None
        self.surfaces_collection=None
        
        # Store references to objects we may want to access later
        self.axis_objects = {'x': None, 'y': None, 'z': None}
        self.grid_objects = {'xy': [], 'xz': [], 'yz': []}
        
        # Create a colormap for functions
        self.color_index = 0
        self.color_list = self._generate_colors(10)  # Generate 10 distinct colors
        
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
    
    def _generate_colors(self, num_colors):
        """Generate visually distinct colors"""
        colors = []
        for i in range(num_colors):
            # Use HSV for more visually distinct colors, convert to RGB
            hue = i / num_colors
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append((rgb[0], rgb[1], rgb[2], 1.0))
        return colors
    
    def _get_next_color(self):
        """Get the next color in rotation"""
        color = self.color_list[self.color_index % len(self.color_list)]
        self.color_index += 1
        return color
    
    def _create_material(self, name, color=(0.8, 0.2, 0.2, 1.0)):
        """Create a new material with the given color"""
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
            for fc in curve_obj.animation_data.action.fcurves:
                for kfp in fc.keyframe_points:
                    kfp.interpolation = 'ELASTIC'

        return curve_obj
    
    def draw_axes(self, length=None, labels=True, arrows=True, thickness=None):
        """
        Draw the coordinate axes (x, y, z)
        
        Args:
            length: length of each axis (defaults to grid_size * spacing)
            labels: whether to add text labels to axes
            arrows: whether to add arrow tips to axes
            thickness: thickness of axis lines (overrides default)
        """
        if length is None:
            length = self.grid_size * self.spacing
            
        if thickness is None:
            thickness = self.axis_thickness
            
        # Create a collection for axes
        self.axis_collection = self._create_collection(f"{self.name}_Axes", self.main_collection)
        
        # Create materials for each axis
        x_mat = self._create_material("X_Axis_Material", color=(1, 0, 0, 1))
        y_mat = self._create_material("Y_Axis_Material", color=(0, 1, 0, 1))
        z_mat = self._create_material("Z_Axis_Material", color=(0, 0, 1, 1))
        
        # Draw the axes
        origin = Vector((0, 0, 0))
        x_end = Vector((length, 0, 0))
        y_end = Vector((0, length, 0))
        z_end = Vector((0, 0, length))
        
        self.axis_objects['x'] = self.draw_line(
            origin, x_end, name="X_Axis", 
            collection=self.axis_collection, 
            material=x_mat, thickness=thickness
        )
        
        self.axis_objects['y'] = self.draw_line(
            origin, y_end, name="Y_Axis", 
            collection=self.axis_collection, 
            material=y_mat, thickness=thickness
        )
        
        self.axis_objects['z'] = self.draw_line(
            origin, z_end, name="Z_Axis", 
            collection=self.axis_collection, 
            material=z_mat, thickness=thickness
        )
        
        # Add arrow tips if requested
        if arrows:
            arrow_size = thickness * 15
            self._add_arrow_tip(x_end, Vector((1, 0, 0)), arrow_size, "X_Arrow", x_mat)
            self._add_arrow_tip(y_end, Vector((0, 1, 0)), arrow_size, "Y_Arrow", y_mat)
            self._add_arrow_tip(z_end, Vector((0, 0, 1)), arrow_size, "Z_Arrow", z_mat)
            
        # Add text labels if requested
        if labels:
            label_offset = length * 0.05
            self._add_text_label("X", x_end + Vector((label_offset, 0, 0)), "X_Label", x_mat)
            self._add_text_label("Y", y_end + Vector((0, label_offset, 0)), "Y_Label", y_mat)
            self._add_text_label("Z", z_end + Vector((0, 0, label_offset)), "Z_Label", z_mat)
            
    def _add_arrow_tip(self, location, direction, size, name, material):
        """Add an arrow tip at the specified location"""
        bpy.ops.mesh.primitive_cone_add(
            radius1=size, radius2=0,
            depth=size*2,
            location=location,
            scale=(1, 1, 1),
        )
        
        cone = bpy.context.active_object
        cone.name = name
        
        # Point in the correct direction
        #direction.normalize()
        # Convert direction to the rotation
        # This is a simplification - we're aligning the cone's Z axis with our direction
        up_vector = Vector((0, 0, 1))
        axis = up_vector.cross(direction)
        angle = up_vector.angle(direction)
        
        if axis.length > 0.001:  # Avoid division by zero
            axis.normalize()
            cone.rotation_mode = 'AXIS_ANGLE'
            cone.rotation_axis_angle = [angle, axis.x, axis.y, axis.z]
        elif angle > 3.14:  # If vectors are opposite, rotate 180° around X
            cone.rotation_euler = (3.14159, 0, 0)
        
        # Apply material
        if cone.data.materials:
            cone.data.materials[0] = material
        else:
            cone.data.materials.append(material)
            
        # Link to collection
        bpy.context.collection.objects.unlink(cone)
        self.axis_collection.objects.link(cone)
        
    def _add_text_label(self, text, location, name, material):
        """Add a text label at the specified location"""
        text_curve = bpy.data.curves.new(type="FONT", name=f"{name}_Curve")
        text_curve.body = text
        text_curve.size = self.axis_thickness * 20
        text_curve.align_x = 'CENTER'
        text_curve.align_y = 'CENTER'
        
        text_obj = bpy.data.objects.new(name, text_curve)
        text_obj.location = location
        
        # Apply material
        if text_obj.data.materials:
            text_obj.data.materials[0] = material
        else:
            text_obj.data.materials.append(material)
            
        # Add extrusion for better visibility
        text_curve.extrude = self.axis_thickness * 2
        
        # Link to collection
        self.axis_collection.objects.link(text_obj)
        
        # Always face the camera
        text_obj.rotation_euler = (1.5708, 0, 0)  # 90 degrees in X
        constraint = text_obj.constraints.new(type='COPY_ROTATION')
        constraint.target = bpy.context.scene.camera
        constraint.use_x = False
        constraint.use_y = False
        constraint.use_z = True
        constraint.invert_z = True
        
        return text_obj
    
    def draw_grid(self, planes=['xy'], tick_labels=False):
        """
        Draw grid lines on specified planes
        
        Args:
            planes: list of planes to draw grids on. Can include 'xy', 'xz', 'yz'
            tick_labels: whether to add numeric labels at grid ticks
        """
        self.grid_collection = self._create_collection(f"{self.name}_Grid", self.main_collection)
        
        # Create grid materials
        xy_mat = self._create_material("XY_Grid_Material", color=(0.8, 0.8, 0.8, 0.3))
        xy_mat.blend_method = 'BLEND'
        
        xz_mat = self._create_material("XZ_Grid_Material", color=(0.8, 0.8, 0.8, 0.3))
        xz_mat.blend_method = 'BLEND'
        
        yz_mat = self._create_material("YZ_Grid_Material", color=(0.8, 0.8, 0.8, 0.3))
        yz_mat.blend_method = 'BLEND'
        
        half = self.grid_size // 2
        
        # Draw XY plane grid (parallel to ground)
        if 'xy' in planes:
            for x in range(-half, half + 1):
                if x == 0:  # Skip the axis line, it's drawn separately
                    continue
                x_pos = x * self.spacing
                line = self.draw_line(
                    (x_pos, -half * self.spacing, 0),
                    (x_pos, half * self.spacing, 0),
                    name=f"XY_Grid_X{x}",
                    collection=self.grid_collection,
                    material=xy_mat
                )
                self.grid_objects['xy'].append(line)
                
                # Add tick label if requested
                if tick_labels and x % 2 == 0:
                    self._add_tick_label(str(x), (x_pos, -half * self.spacing * 1.05, 0), f"X{x}_Label", xy_mat)

            for y in range(-half, half + 1):
                if y == 0:  # Skip the axis line
                    continue
                y_pos = y * self.spacing
                line = self.draw_line(
                    (-half * self.spacing, y_pos, 0),
                    (half * self.spacing, y_pos, 0),
                    name=f"XY_Grid_Y{y}",
                    collection=self.grid_collection,
                    material=xy_mat
                )
                self.grid_objects['xy'].append(line)
                
                # Add tick label if requested
                if tick_labels and y % 2 == 0:
                    self._add_tick_label(str(y), (-half * self.spacing * 1.05, y_pos, 0), f"Y{y}_Label", xy_mat)
        
        # Draw XZ plane grid (vertical, frontal)
        if 'xz' in planes:
            for x in range(-half, half + 1):
                if x == 0:  # Skip the axis line
                    continue
                x_pos = x * self.spacing
                line = self.draw_line(
                    (x_pos, 0, -half * self.spacing),
                    (x_pos, 0, half * self.spacing),
                    name=f"XZ_Grid_X{x}",
                    collection=self.grid_collection,
                    material=xz_mat
                )
                self.grid_objects['xz'].append(line)

            for z in range(-half, half + 1):
                if z == 0:  # Skip the axis line
                    continue
                z_pos = z * self.spacing
                line = self.draw_line(
                    (-half * self.spacing, 0, z_pos),
                    (half * self.spacing, 0, z_pos),
                    name=f"XZ_Grid_Z{z}",
                    collection=self.grid_collection,
                    material=xz_mat
                )
                self.grid_objects['xz'].append(line)
                
                # Add tick label if requested
                if tick_labels and z % 2 == 0:
                    self._add_tick_label(str(z), (-half * self.spacing * 1.05, 0, z_pos), f"Z{z}_Label", xz_mat)
        
        # Draw YZ plane grid (vertical, side)
        if 'yz' in planes:
            for y in range(-half, half + 1):
                if y == 0:  # Skip the axis line
                    continue
                y_pos = y * self.spacing
                line = self.draw_line(
                    (0, y_pos, -half * self.spacing),
                    (0, y_pos, half * self.spacing),
                    name=f"YZ_Grid_Y{y}",
                    collection=self.grid_collection,
                    material=yz_mat
                )
                self.grid_objects['yz'].append(line)

            for z in range(-half, half + 1):
                if z == 0:  # Skip the axis line
                    continue
                z_pos = z * self.spacing
                line = self.draw_line(
                    (0, -half * self.spacing, z_pos),
                    (0, half * self.spacing, z_pos),
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
            
        if color is None:
            color = self._get_next_color()
            
        # Create points collection if it doesn't exist
        if self.points_collection is None:
            self.points_collection = self._create_collection(f"{self.name}_Points", self.main_collection)
            
        # Create material for this point
        point_mat = self._create_material(f"{name}_Material", color=color)
        
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
        
    def plot_points(self, points, animated=True, trail=False, color=None):
        """
        Plot a series of 3D points
        
        Args:
            points: list of (x,y,z) coordinates
            animated: whether to animate the appearance of points
            trail: whether to draw a line connecting the points
            color: color for the points (optional)
        """
        if color is None:
            color = self._get_next_color()
            
        frame_counter = self.animation_start_frame
        point_objects = []
        
        for i, point in enumerate(points):
            # Ensure point is in 3D format
            if len(point) == 2:
                point = (point[0], point[1], 0)
                
            # Create the point
            point_obj = self.create_point(
                point, 
                name=f"Point_{i}", 
                color=color,
                size=self.point_size
            )
            point_objects.append(point_obj)
            
            # Animate point appearance if requested
            if animated:
                self.animate_object_appearance(
                    point_obj, 
                    frame_counter
                )
                frame_counter += self.animation_point_interval
                
        # Draw a trail connecting the points if requested
        if trail and len(points) > 1:
            if self.curves_collection is None:
                self.curves_collection = self._create_collection(f"{self.name}_Curves", self.main_collection)
                
            trail_mat = self._create_material("Trail_Material", color=color)
            
            self.draw_curve(
                points, 
                name="Points_Trail",
                collection=self.curves_collection,
                material=trail_mat,
                animate=animated,
                start_frame=self.animation_start_frame,
                duration=len(points) * self.animation_point_interval
            )
            
        return point_objects
    
    def animate_object_appearance(self, obj, frame):
        """
        Animate an object's appearance by scaling it from zero
        
        Args:
            obj: Blender object to animate
            frame: frame to start the animation
        """
        # Store original scale
        original_scale = obj.scale.copy()
        
        # Set scale to zero at start
        obj.scale = (0, 0, 0)
        obj.keyframe_insert(data_path="scale", frame=frame)
        
        # Restore original scale with animation
        obj.scale = original_scale
        obj.keyframe_insert(data_path="scale", frame=frame + self.animation_point_interval)
        
        # Add some elastic bounce to the animation
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kfp in fc.keyframe_points:
                    kfp.interpolation = 'ELASTIC'
                    
    def draw_curve(self, points, name="Curve", collection=None, material=None, 
                   thickness=None, animate=False, start_frame=1, duration=20):
        """
        Draw a curve by connecting a series of points
        
        Args:
            points: list of 3D points to connect
            name: name for the curve object
            collection: collection to add to (optional)
            material: material to apply (optional)
            thickness: curve thickness (overrides default)
            animate: whether to animate the curve drawing
            start_frame: frame to start animation
            duration: duration of animation in frames
            
        Returns:
            The created curve object
        """
        if thickness is None:
            thickness = self.curve_thickness
            
        if material is None:
            material = self._create_material(f"{name}_Material", self._get_next_color())
            
        if collection is None:
            if self.curves_collection is None:
                self.curves_collection = self._create_collection(f"{self.name}_Curves", self.main_collection)
            collection = self.curves_collection
            
        # Create curve data
        curve_data = bpy.data.curves.new(name=f"{name}_Curve", type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 12
        
        # Add polyline with all points
        polyline = curve_data.splines.new(type='BEZIER')
        polyline.bezier_points.add(len(points) - 1)
        
        for i, point in enumerate(points):
            polyline.bezier_points[i].co = Vector(point)
            polyline.bezier_points[i].handle_left_type = 'AUTO'
            polyline.bezier_points[i].handle_right_type = 'AUTO'
            
        # Create object from curve
        curve_obj = bpy.data.objects.new(name, curve_data)
        
        # Link to collection
        collection.objects.link(curve_obj)
        
        # Add thickness
        curve_data.bevel_depth = thickness
        curve_data.bevel_resolution = 4
        
        # Apply material if provided
        if curve_obj.data.materials:
            curve_obj.data.materials[0] = material
        else:
            curve_obj.data.materials.append(material)
            
        # Animate if requested
        if animate:
            curve_data.bevel_factor_start = 0.0
            curve_data.bevel_factor_end = 0.0
            curve_data.keyframe_insert(data_path="bevel_factor_end", frame=start_frame)
            
            curve_data.bevel_factor_end = 1.0
            curve_data.keyframe_insert(data_path="bevel_factor_end", frame=start_frame + duration)
            
            # Add easing
            if curve_obj.animation_data and curve_obj.animation_data.action:
                for fc in curve_obj.animation_data.action.fcurves:
                    for kfp in fc.keyframe_points:
                        kfp.interpolation = 'SINE'
                        
        return curve_obj
        
    def plot_function_z(self, expr_str, x_range=(-5, 5), y_range=(-5, 5), 
                       x_samples=20, y_samples=20, colored=True, wireframe=True,
                       animate=True):
        """
        Plot a 3D surface where z = f(x,y)
        
        Args:
            expr_str: string expression of z in terms of x and y
            x_range: tuple with (min_x, max_x)
            y_range: tuple with (min_y, max_y)
            x_samples: number of x samples
            y_samples: number of y samples
            colored: whether to color the surface by height
            wireframe: whether to show wireframe lines
            animate: whether to animate surface appearance
            
        Returns:
            The created surface object
        """
        # Create symbols and expression
        x, y = sp.symbols('x y')
        expr = sp.sympify(expr_str)
        
        # Create meshgrid for x, y coordinates
        x_vals = np.linspace(x_range[0], x_range[1], x_samples)
        y_vals = np.linspace(y_range[0], y_range[1], y_samples)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Calculate z values
        Z = np.zeros((y_samples, x_samples))
        z_min, z_max = float('inf'), float('-inf')
        
        for i in range(y_samples):
            for j in range(x_samples):
                try:
                    Z[i, j] = float(expr.subs({x: X[i, j], y: Y[i, j]}))
                    z_min = min(z_min, Z[i, j])
                    z_max = max(z_max, Z[i, j])
                except:
                    Z[i, j] = 0  # In case of evaluation errors
                    
        # Create vertices and faces
        vertices = []
        for i in range(y_samples):
            for j in range(x_samples):
                vertices.append((X[i, j], Y[i, j], Z[i, j]))
                
        faces = []
        for i in range(y_samples - 1):
            for j in range(x_samples - 1):
                # Calculate indices of 4 corners of each grid cell
                idx1 = i * x_samples + j
                idx2 = i * x_samples + j + 1
                idx3 = (i + 1) * x_samples + j + 1
                idx4 = (i + 1) * x_samples + j
                # Add two triangular faces per grid cell
                faces.append((idx1, idx2, idx3))
                faces.append((idx1, idx3, idx4))
                
        # Create mesh and object
        mesh = bpy.data.meshes.new(f"Surface_{expr_str}_Mesh")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        
        # Create object
        obj = bpy.data.objects.new(f"Surface_{expr_str}", mesh)
        
        # Add to collection
        if self.surfaces_collection is None:
            self.surfaces_collection = self._create_collection(f"{self.name}_Surfaces", self.main_collection)
        self.surfaces_collection.objects.link(obj)
        
        # Apply material based on height if colored
        if colored:
            # Create vertex color layer for height-based coloring
            if not mesh.vertex_colors:
                mesh.vertex_colors.new()
            color_layer = mesh.vertex_colors.active
            
            # Create a gradient of colors for the height
            for i, face in enumerate(mesh.polygons):
                for j, vert_idx in enumerate(face.vertices):
                    # Normalize z height to 0-1 range
                    if z_max > z_min:
                        norm_z = (vertices[vert_idx][2] - z_min) / (z_max - z_min)
                    else:
                        norm_z = 0.5
                        
                    # Create a color using HSV (blue to red gradient)
                    # Hue: 0.7 (blue) to 0.0 (red)
                    hue = 0.7 - 0.7 * norm_z
                    rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
                    color_layer.data[i * 3 + j].color = (*rgb, 1.0)
                    
            # Create material that uses vertex colors
            mat = bpy.data.materials.new(f"Surface_{expr_str}_Material")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            
            # Clear existing nodes
            for node in nodes:
                nodes.remove(node)
                
            # Add vertex color input
            vert_color = nodes.new(type='ShaderNodeVertexColor')
            vert_color.layer_name = color_layer.name
            
            # Add principled BSDF shader
            bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf.inputs['Specular'].default_value = 0.2
            bsdf.inputs['Roughness'].default_value = 0.3
            
            # Add output node
            output = nodes.new(type='ShaderNodeOutputMaterial')
            
            # Connect nodes
            links.new(vert_color.outputs['Color'], bsdf.inputs['Base Color'])
            links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
            
            obj.data.materials.append(mat)
        else:
            # Use a single material
            mat = self._create_material(f"Surface_{expr_str}_Material", self._get_next_color())
            obj.data.materials.append(mat)
            
        # Add wireframe if requested
        if wireframe:
            self._add_wireframe_to_surface(obj, expr_str, x_samples, y_samples)
            
        # Animate if requested
        if animate:
            self._animate_surface_growth(obj, self.animation_start_frame, 
                                         duration=30, growth_type='spiral')
            
        return obj
    
    def _add_wireframe_to_surface(self, surface_obj, expr_str, x_samples, y_samples):
        """Add wireframe lines to a surface object"""
        # Create wireframe material
        wire_mat = self._create_material(f"Wireframe_{expr_str}_Material", color=(0.1, 0.1, 0.1, 1.0))
        
        # Create wireframe modifier
        wireframe = surface_obj.modifiers.new(name="Wireframe", type='WIREFRAME')
        wireframe.thickness = self.grid_thickness * 3
        wireframe.use_relative_offset = True
        wireframe.material_offset = len(surface_obj.data.materials)
        
        # Add wireframe material to object
        surface_obj.data.materials.append(wire_mat)
        
    def _animate_surface_growth(self, obj, start_frame, duration=30, growth_type='uniform'):
        """Animate the growth of a surface
        
        Args:
            obj: The surface object to animate
            start_frame: Frame to start animation
            duration: Duration of animation in frames
            growth_type: Type of growth animation ('uniform', 'spiral', 'wave')
        """
        # Make the object invisible at start
        obj.hide_viewport = True
        obj.hide_render = True
        
        obj.keyframe_insert(data_path="hide_viewport", frame=start_frame)
        obj.keyframe_insert(data_path="hide_render", frame=start_frame)
        
        # Make it visible at animation start
        obj.hide_viewport = False
        obj.hide_render = False
        
        obj.keyframe_insert(data_path="hide_viewport", frame=start_frame + 1)
        obj.keyframe_insert(data_path="hide_render", frame=start_frame + 1)
        
        # Scale from zero animation
        obj.scale = (0, 0, 0)
        obj.keyframe_insert(data_path="scale", frame=start_frame + 1)
        
        if growth_type == 'uniform':
            # Simple uniform scale animation
            obj.scale = (1, 1, 1)
            obj.keyframe_insert(data_path="scale", frame=start_frame + duration)
            
        elif growth_type == 'spiral':
            # More interesting spiral growth
            # Add several keyframes with partial scales
            quarter = duration // 4
            
            # X axis grows first
            obj.scale = (0.5, 0, 0)  
            obj.keyframe_insert(data_path="scale", frame=start_frame + quarter)
            
            # Then Y axis
            obj.scale = (1, 0.5, 0)
            obj.keyframe_insert(data_path="scale", frame=start_frame + quarter*2)
            
            # Then Z axis
            obj.scale = (1, 1, 0.5)
            obj.keyframe_insert(data_path="scale", frame=start_frame + quarter*3)
            
            # Final full scale
            obj.scale = (1, 1, 1)
            obj.keyframe_insert(data_path="scale", frame=start_frame + duration)
            
        elif growth_type == 'wave':
            # Wave-like animation that overshoots and settles
            half = duration // 2
            
            # Overshoot
            obj.scale = (1.2, 1.2, 1.2)
            obj.keyframe_insert(data_path="scale", frame=start_frame + half)
            
            # Settle back to normal
            obj.scale = (1, 1, 1)
            obj.keyframe_insert(data_path="scale", frame=start_frame + duration)
        
        # Add easing to animations
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kfp in fc.keyframe_points:
                    kfp.interpolation = 'ELASTIC'
    

    
my_graph=Graph3D(grid_size=20)
my_graph.draw_grid(planes=['xy','yz'])
my_graph.draw_axes()
my_graph.plot_function_z("x**2 + y**2")
