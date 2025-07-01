import bpy
import os
import json
import numpy as np
from mathutils import Vector

# -------------------------------
# Configuration Variables
# -------------------------------
RENDER_STILL = True       # Render final still image.
RENDER_VIDEO = False       # Render simulation as a video.
DROP_OFFSET = 100          # Rocks are dropped from this many units above their given z.
SIMULATION_FRAME_END = 150 # End frame for the simulation.

# Global caches for FBX templates and materials.
fbx_cache = {}
material_cache = {}

render_path = os.getenv("RENDERPATH", "render.png")

# -------------------------------
# FBX Import and Material Functions
# -------------------------------

def import_and_center_fbx(filepath):
    before_objs = set(bpy.context.scene.objects)
    bpy.ops.import_scene.fbx(filepath=filepath)
    imported_objs = [obj for obj in bpy.context.scene.objects if obj not in before_objs]
    
    if not imported_objs:
        print(f"Failed to import {filepath}")
        return None

    if len(imported_objs) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in imported_objs:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = imported_objs[0]
        bpy.ops.object.join()
        rock_obj = bpy.context.object
    else:
        rock_obj = imported_objs[0]
    
    bpy.context.view_layer.objects.active = rock_obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return rock_obj

def get_rock_model(fbx_file):
    global fbx_cache
    if fbx_file in fbx_cache:
        dup = fbx_cache[fbx_file].copy()
        dup.data = fbx_cache[fbx_file].data.copy()
        bpy.context.collection.objects.link(dup)
        return dup
    else:
        rock = import_and_center_fbx(fbx_file)
        if rock:
            fbx_cache[fbx_file] = rock
            bpy.context.collection.objects.unlink(rock)
            dup = rock.copy()
            dup.data = rock.data.copy()
            bpy.context.collection.objects.link(dup)
            return dup
    return None

def get_rock_material(index):
    global material_cache
    if index in material_cache:
        return material_cache[index]
    
    mat = bpy.data.materials.new(name=f"RockMaterial_{index:02d}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in nodes:
        nodes.remove(node)
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (400, 0)
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (0, 0)
    # Set the shader to be non-metallic and high roughness for a matte look.
    principled.inputs['Metallic'].default_value = 0
    principled.inputs['Roughness'].default_value = 0.9
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    basecolor_node = nodes.new(type='ShaderNodeTexImage')
    basecolor_node.location = (-400, 200)
    basecolor_path = os.path.join("..", "rock_models", "orange_natural_rock_texture", f"orange natural_rock {index:02d}_BaseColor.png")
    if os.path.exists(basecolor_path):
        try:
            basecolor_node.image = bpy.data.images.load(basecolor_path)
        except Exception as e:
            print(f"Failed to load BaseColor for index {index}: {e}")
    links.new(basecolor_node.outputs['Color'], principled.inputs['Base Color'])
    
    metallic_node = nodes.new(type='ShaderNodeTexImage')
    metallic_node.location = (-400, 0)
    metallic_path = os.path.join("..", "rock_models", "orange_natural_rock_texture", f"orange natural_rock {index:02d}_Metallic.png")
    if os.path.exists(metallic_path):
        try:
            metallic_node.image = bpy.data.images.load(metallic_path)
        except Exception as e:
            print(f"Failed to load Metallic for index {index}: {e}")
    links.new(metallic_node.outputs['Color'], principled.inputs['Metallic'])
    
    roughness_node = nodes.new(type='ShaderNodeTexImage')
    roughness_node.location = (-400, -200)
    roughness_path = os.path.join("..", "rock_models", "orange_natural_rock_texture", f"orange natural_rock {index:02d}_Roughness.png")
    if os.path.exists(roughness_path):
        try:
            roughness_node.image = bpy.data.images.load(roughness_path)
        except Exception as e:
            print(f"Failed to load Roughness for index {index}: {e}")
    links.new(roughness_node.outputs['Color'], principled.inputs['Roughness'])
    
    normal_tex_node = nodes.new(type='ShaderNodeTexImage')
    normal_tex_node.location = (-800, -200)
    normal_path = os.path.join("..", "rock_models", "orange_natural_rock_texture", f"orange natural_rock {index:02d}_Normal.png")
    if os.path.exists(normal_path):
        try:
            normal_tex_node.image = bpy.data.images.load(normal_path)
            normal_tex_node.image.colorspace_settings.name = 'Non-Color'
        except Exception as e:
            print(f"Failed to load Normal for index {index}: {e}")
    normal_map_node = nodes.new(type='ShaderNodeNormalMap')
    normal_map_node.location = (-400, -400)
    links.new(normal_tex_node.outputs['Color'], normal_map_node.inputs['Color'])
    links.new(normal_map_node.outputs['Normal'], principled.inputs['Normal'])
    
    material_cache[index] = mat
    return mat

# -------------------------------
# Scene Setup Functions
# -------------------------------

def setup_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def setup_world():
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bpy.context.scene.world = world
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.9, 0.9, 0.9, 1)
        bg_node.inputs["Strength"].default_value = 1.0
    # Increase gravity to settle objects faster (units in mm/s^2)
    bpy.context.scene.gravity = Vector((0, 0, -200))

def setup_table():
    bpy.ops.mesh.primitive_plane_add(size=320, location=(0, 0, -1.5))
    table = bpy.context.object
    table.name = "Table"
    
    # Add a Solidify modifier to give the table thickness.
    solidify_mod = table.modifiers.new(name="Solidify", type='SOLIDIFY')
    solidify_mod.thickness = 5  # Adjust thickness as needed.
    # Optionally, you can apply the modifier:
    bpy.context.view_layer.objects.active = table
    bpy.ops.object.modifier_apply(modifier=solidify_mod.name)
    
    # Create a simple white material.
    mat = bpy.data.materials.new(name="TableMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in nodes:
        nodes.remove(node)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (1, 1, 1, 1)
    bsdf.inputs['Roughness'].default_value = 0.9
    output = nodes.new('ShaderNodeOutputMaterial')
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    table.data.materials.append(mat)
    
    # Add rigid body physics to the table.
    bpy.context.view_layer.objects.active = table
    bpy.ops.rigidbody.object_add()
    table.rigid_body.type = 'PASSIVE'
    table.rigid_body.friction = 0.5


def create_container():
    container_size = 320
    wall_thickness = 10
    wall_height = 200
    table_z = -1.5
    half = container_size / 2
    wall_size = 350
    
    walls = []
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, half, table_z + wall_height/2 - 20))
    front = bpy.context.object
    front.name = "FrontWall"
    front.scale = (wall_size, wall_thickness, wall_height)
    walls.append(front)
    
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -half, table_z + wall_height/2 - 20))
    back = bpy.context.object
    back.name = "BackWall"
    back.scale = (wall_size, wall_thickness, wall_height)
    walls.append(back)
    
    bpy.ops.mesh.primitive_cube_add(size=1, location=(half, 0, table_z + wall_height/2 - 20))
    right = bpy.context.object
    right.name = "RightWall"
    right.scale = (wall_thickness, wall_size, wall_height)
    walls.append(right)
    
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-half, 0, table_z + wall_height/2 - 20))
    left = bpy.context.object
    left.name = "LeftWall"
    left.scale = (wall_thickness, wall_size, wall_height)
    walls.append(left)
    
    top_z = table_z + wall_height + wall_thickness/2 - 40
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, top_z))
    top = bpy.context.object
    top.name = "TopWall"
    top.scale = (wall_size, wall_size, wall_thickness/2)
    walls.append(top)
    
    for wall in walls:
        wall.display_type = 'WIRE'
        bpy.context.view_layer.objects.active = wall
        bpy.ops.rigidbody.object_add()
        wall.rigid_body.type = 'PASSIVE'
        wall.rigid_body.friction = 0.5
        wall.rigid_body.collision_margin = 0.01
        wall.hide_render = True
    return walls

def setup_lighting():
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()
    bpy.ops.object.light_add(type='AREA', location=(30, -30, 50))
    main_light = bpy.context.object
    main_light.data.energy = 1200
    main_light.data.size = 15
    main_light.rotation_euler = (np.radians(60), 0, np.radians(45))
    bpy.ops.object.light_add(type='AREA', location=(-30, 30, 50))
    fill_light = bpy.context.object
    fill_light.data.energy = 600
    fill_light.data.size = 10
    fill_light.rotation_euler = (np.radians(60), 0, np.radians(-135))

def setup_camera():
    bpy.ops.object.select_by_type(type='CAMERA')
    bpy.ops.object.delete()
    cam_loc = Vector((0, 0, 400))
    cam_rot = (np.radians(0), 0, 0)
    bpy.ops.object.camera_add(location=cam_loc, rotation=cam_rot)
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 320

def setup_render():
    scene = bpy.context.scene
    scene.render.resolution_x = 1000
    scene.render.resolution_y = 1000
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 128
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = False
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 0.001
    bpy.context.scene.render.engine = 'CYCLES'
    # Get Cycles preferences
    prefs = bpy.context.preferences.addons['cycles'].preferences

    # Choose between 'CUDA' or 'OPTIX' based on your preference and Blender build
    prefs.compute_device_type = 'CUDA'  # Change to 'CUDA' if you prefer

    gpu_available = False

    # Print devices for debugging
    prefs.get_devices()

    for dev in prefs.devices:
        print(dev)
        # Enable devices of type 'CUDA' or 'OPTIX'
        if dev.type in {'CUDA', 'OPTIX'}:
            dev.use = True
            gpu_available = True

    # Set rendering device
    if gpu_available:
        bpy.context.scene.cycles.device = 'GPU'
        print("✅ GPU is selected for rendering!")
    else:
        bpy.context.scene.cycles.device = 'CPU'
        print("⚠️ No GPU found, falling back to CPU.")

    # Confirm active rendering device
    print("\n=== Active Rendering Device ===")
    for dev in prefs.devices:
        print(f"  Device: {dev.name} (Type: {dev.type}), Use: {dev.use}")
    print("===============================\n")

def load_particle_data():
    try:
        with open(os.getenv("PARTICLESPATH", "particles.json"), 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading particles: {e}")
        return None

# -------------------------------
# Particle Creation & Physics
# -------------------------------

def create_particles(data):
    if not data:
        return

    fbx_files = [os.path.join("..", "rock_models", f"rock_{i:02d}.FBX") for i in range(1, 13)]
    
    for i, p in enumerate(data['particles']):
        model_path = np.random.choice(fbx_files)
        rock = get_rock_model(model_path)
        if rock is None:
            continue

        rock.scale = (1, 1, 1)
        bpy.context.view_layer.update()
        dims = rock.dimensions

        scale = p['size'] / max(dims.x, dims.y, dims.z) if dims.x != 0 else 1
        rock.scale = (scale, scale, scale)

        x = p['x']
        y = p['y']
        z = DROP_OFFSET
        rock.location = (x, y, z)
        
        try:
            base_name = os.path.basename(model_path)
            index_str = base_name.split('_')[-1].split('.')[0]
            index = int(index_str)
        except Exception as e:
            index = 1
        
        if rock.data.materials:
            rock.data.materials.clear()
        rock.data.materials.append(get_rock_material(index))
        
        bpy.context.view_layer.objects.active = rock
        bpy.ops.rigidbody.object_add()
        rock.rigid_body.mass = 5
        rock.rigid_body.friction = 0.5
        # Add damping to help them settle quickly.
        rock.rigid_body.linear_damping = 0.8
        rock.rigid_body.angular_damping = 0.8

        
        if (i + 1) % 50 == 0:
            print(f"Created {i+1} rocks...")

# -------------------------------
# Simulation Baking & Rendering
# -------------------------------

def bake_simulation(frame_start, frame_end):

    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.frame_set(frame_start)
    bpy.ops.ptcache.bake_all(bake=True)



def main():
    setup_scene()
    setup_world()
    setup_table()
    create_container()
    setup_lighting()
    setup_camera()
    setup_render()
    
    data = load_particle_data()
    create_particles(data)
    
    bake_simulation(1, SIMULATION_FRAME_END)
    
    bpy.context.scene.frame_set(SIMULATION_FRAME_END)
    
    if RENDER_VIDEO:
        bpy.context.scene.render.filepath = os.getenv("RENDERPATH", "simulation.mp4")
        bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
        bpy.ops.render.render(animation=True)
    
    if RENDER_STILL:
        bpy.context.scene.render.filepath = os.getenv("STILLPATH", render_path)
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.ops.render.render(write_still=True)
    
    print("Render complete!")

if __name__ == '__main__':
    main()
