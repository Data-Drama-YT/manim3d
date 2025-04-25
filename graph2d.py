import bpy
from mathutils import Vector

class Graph2D:
    def __init__(self,points=[],name="Graph2D",material=None):
        """
        points: list of (x,y) tuples
        material: Blender material to apply (optional)
        """
        self.points=[Vector((x,y,0)) for x,y in points] or []
        self.name=name
        self.material=material
        self.grid_collection=None
        self.obj=None

    def draw_line(self,start,end,name="MyLine",collection=None):
        """
        The goal is to draw a line between 2 points.
        This is to be used by a cartesian grid function
        later on to draw a whole 2d or 3d grid.
        
        Inputs:
        start: A tuple in the form (x,y,z)
        end: A tuple in the form (x,y,z)
        name: Optional, name for the line.
        """
        mesh=bpy.data.meshes.new(name)
        obj=bpy.data.objects.new(name,mesh)
        bpy.context.collection.objects.link(obj)

        if collection:
            collection.objects.link(obj)

        vertices=[Vector(start),Vector(end)]
        edges=[(0,1)]
        mesh.from_pydata(vertices,edges,[])
        mesh.update()

        return obj
    
    def draw_cartesian_grid(self,grid_size=10,spacing=1,collection_name='Grid_Lines'):
        """
        The goal is to draw a grid of lines using draw_line
        """
        self.grid_collection=bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(self.grid_collection)
        half=grid_size//2 #useful for centering the grid

        for x in range(-half,half+1):
            self.draw_line(
                (x*spacing,-half*spacing,0),
                (x*spacing,half*spacing,0),
                name=f"VLine{x}",
                collection=self.grid_collection
            )
        
        for y in range(-half,half+1):
            self.draw_line(
                (-half * spacing, y * spacing, 0),
                (half * spacing, y * spacing, 0),
                name=f"HLine_{y}",
                collection=self.grid_collection
            )
    
    def erect_grid(self):
        if self.grid_collection:
            for obj in self.grid_collection.objects:
                obj.rotation_euler=(1.5708,0,0) # This is 90 degrees on the x axis.

my_graph=Graph2D()
my_graph.draw_cartesian_grid()
my_graph.erect_grid()