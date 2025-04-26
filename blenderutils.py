import bpy

def delete_all_objects_and_collections():
    # Delete all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Delete all collections
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)
