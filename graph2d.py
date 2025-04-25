import bpy
from mathutils import Vector

class Graph2D:
    def __init__(self,points,name="Graph2D",material=None):
        """
        points: list of (x,y) tuples
        material: Blender material to apply (optional)
        """
        self.points=[Vector((x,y,0)) for x,y in points]
        self.name=name
        self.material=material
        self.obj=None

    def draw_line(start,end,name="MyLine"):
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

        vertices=[Vector(start),Vector(end)]
        edges=[(0,1)]
        mesh.from_pydata(vertices,edges,[])
        mesh.update()

        return obj

Graph2D.draw_line((0,0,0),(0,0,5),"Starter Line")