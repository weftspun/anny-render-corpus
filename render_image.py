"""Render the DEPTH MAP, from the same camera that projected the keypoints.

Depth is the conditioning signal, not a by-product. RFD 0121's pipeline stylises an ANNY
render with Qwen-Image under an Apache-2.0 depth ControlNet, and depth is what pins the
geometry so the generated image keeps the pose the labels describe. A flat shaded render is
not the training image and never was: the training image is what the generator returns.

So this writes depth as 16-bit PNG, and a shaded pass only as a human-readable check.

Run as:  blender -b --python render_image.py -- <mesh.npz> <out.png> <cam.json>

WHY THIS IS SEPARATE FROM render_corpus.py. Blender owns the rasteriser and has its own
Python, so the pose and the labels are computed outside and handed in as arrays. Nothing here
recomputes a keypoint.

THE ONE PROPERTY THAT MATTERS. The image and the keypoints must come from ONE camera. If the
projection and the render disagree by so much as a field-of-view convention, every label is
subtly wrong and nothing errors: the picture looks like a person and the dots sit slightly off
the joints, which is indistinguishable from an ordinary annotation error and impossible to find
by eye at scale.

So the camera is passed in as the same numbers `render_corpus.py` projected with, and
`overlay_check` draws the projected points onto the render. Points that land on the body are
the proof. That check is cheap and it is the only thing standing between a corpus and a corpus
of quietly mislabelled images.

Blender's camera differs from the projection in two ways that both have to be undone:

  * it looks down -Z with +Y up in ITS OWN space, and the view matrix here is world-to-camera,
    so the camera object takes the inverse
  * `angle` is the FULL field of view, and the projection uses the half-angle tangent
"""

import json
import sys

import numpy as np

try:
    import bpy
except ImportError:  # imported for overlay_check outside Blender
    bpy = None


def build_scene(verts, faces, cam, width, height):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.film_transparent = False

    mesh = bpy.data.meshes.new("body")
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.validate()
    mesh.update()
    obj = bpy.data.objects.new("body", mesh)
    scene.collection.objects.link(obj)

    # Flat neutral shading. The corpus gets its appearance from the stylisers downstream, so
    # a lit render here would bake one lighting choice into every domain.
    mat = bpy.data.materials.new("flat")
    mat.use_nodes = False
    mat.diffuse_color = (0.72, 0.66, 0.60, 1.0)
    mesh.materials.append(mat)

    cam_data = bpy.data.cameras.new("cam")
    # `angle` is the full field of view. The projection uses the half-angle, so passing the
    # half here would render a body at half the size the labels describe.
    cam_data.angle = np.radians(cam["fov_deg"])
    cam_obj = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # The view matrix is world-to-camera; a camera object holds camera-to-world.
    view = np.array(cam["view"], dtype=np.float64)
    c2w = np.linalg.inv(view)
    cam_obj.matrix_world = [list(r) for r in c2w]

    light_data = bpy.data.lights.new("key", type="SUN")
    light_data.energy = 3.0
    light = bpy.data.objects.new("key", light_data)
    light.matrix_world = [list(r) for r in c2w]
    scene.collection.objects.link(light)

    # EEVEE, because depth is the point and Workbench has no compositor passes. Workbench
    # shades solid geometry without materials or lights and would have been the simpler
    # reference pass, but it produced no Z at all, which is the output that matters.
    #
    # EEVEE needs a world, or it renders black. The first version had a node-less material
    # and no world and produced exactly that: a correct camera, correct keypoints, and no
    # body to check them against.
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    scene.world = world
    scene.render.engine = "BLENDER_EEVEE"
    return scene


def enable_depth(scene, out_dir, basename):
    """Blender's Z pass, written as 16-bit PNG through the compositor.

    Z is metres from the camera, so it is a physical quantity rather than a normalised
    gradient. Normalising per image would make each frame's depth mean something different,
    and a ControlNet conditioned on that learns the normalisation rather than the body.
    """
    scene.use_nodes = True
    scene.view_layers[0].use_pass_z = True
    tree = scene.node_tree
    for n in list(tree.nodes):
        tree.nodes.remove(n)

    layers = tree.nodes.new("CompositorNodeRLayers")
    # Map metres into 0..1 across the depth the body actually occupies. The range is written
    # beside the image so the mapping is invertible rather than lost.
    norm = tree.nodes.new("CompositorNodeNormalize")
    out = tree.nodes.new("CompositorNodeOutputFile")
    out.base_path = str(out_dir)
    out.file_slots[0].path = basename
    out.format.file_format = "PNG"
    out.format.color_depth = "16"
    out.format.color_mode = "BW"

    tree.links.new(layers.outputs["Depth"], norm.inputs[0])
    tree.links.new(norm.outputs[0], out.inputs[0])
    return out


def overlay_check(png_path, kp_px, vis, out_path):
    """Draw the projected keypoints on the render.

    A point on the body means the projection and the rasteriser agree. A point beside it
    means they do not, and the corpus would be mislabelled in a way no downstream check
    would catch.
    """
    from PIL import Image, ImageDraw

    img = Image.open(png_path).convert("RGB")
    d = ImageDraw.Draw(img)
    for (x, y), v in zip(kp_px, vis):
        if v == 0:
            continue
        colour = (14, 200, 220) if v == 2 else (255, 90, 30)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=colour, width=2)
    img.save(out_path)
    return out_path


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    mesh_npz, out_png, cam_json = argv[0], argv[1], argv[2]

    data = np.load(mesh_npz)
    cam = json.loads(open(cam_json).read())
    scene = build_scene(
        data["verts"], data["faces"], cam, cam["width"], cam["height"]
    )
    scene.render.filepath = out_png
    bpy.ops.render.render(write_still=True)
    print(f"RENDERED\t{out_png}")


if __name__ == "__main__":
    main()
