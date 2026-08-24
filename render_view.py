"""Render one view with Pixal3D's own camera distribution, parameterised on numbers.

WHY NOT AN ORBIT. The first pass here put four cameras at 0, 90, 180 and 270 degrees around
the body at a radius of three times its extent. That is a made-up rig. Pixal3D was trained
on cameras drawn from a specific distribution, and an input from outside it is a
distribution shift nobody measured. The generator is upstream's, ported rather than
imitated, from `data_toolkit/utils.py` and `data_toolkit/render_cond.py` at cdbb2bb.

    yaw, pitch = sphere_hammersley_sequence(i, n, offset)
    radius     = sqrt(3)/2 / sin(fov/2)
    eye        = radius * (cos(yaw)cos(pitch), sin(yaw)cos(pitch), sin(pitch))

RADIUS AND FOV ARE ONE NUMBER, NOT TWO. sqrt(3)/2 is the half-diagonal of the unit cube, so
that radius is exactly the distance at which the object's bounding sphere fills the frame.
Upstream samples fov in [10, 70] degrees and derives the radius; pass `--fov` here and the
radius follows. Giving both independently is how you get an object that floats in a corner.

SO THE MESH IS NORMALISED INTO THE UNIT CUBE FIRST, because the coupling above assumes it.
The scale and centre are written to the sidecar: a keypoint label in world units is wrong
about this image until it has been through the same transform, and that transform is a
number somebody will need rather than one to re-derive.

UPSTREAM RANDOMISES WHAT THIS TAKES AS ARGUMENTS. `_render_cond` draws `offset` from
`np.random.rand()` and `k` from a uniform over a million samples. Here they are `--offset`
and `--fov`, so a view is reproducible from its index and two numbers rather than from a
seed nobody recorded.

Z-UP, matching both ANNY and upstream's `render_cond.py`, whose camera position is
`(cos(yaw)cos(pitch), sin(yaw)cos(pitch), sin(pitch))`.

Usage:
    python render_view.py <mesh.npz> <out.png> --index 0 --views 8 [--fov 40] [--spp 128]
"""

import argparse
import hashlib
import json
import math
import pathlib
import sys

import numpy as np

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]


def radical_inverse(base, n):
    val, inv_base = 0.0, 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        val += (n % base) * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val


def sphere_hammersley(i, num_samples, offset=(0.0, 0.0)):
    """Upstream's `sphere_hammersley_sequence`, ported exactly. Returns (yaw, pitch).

    The `u < 0.25` branch is upstream's and is not a simplification of anything: it warps the
    first quarter of the sequence so the cameras cluster nearer the horizon than a uniform
    sphere would. Reproducing the distribution means reproducing that too.
    """
    u = i / num_samples + offset[0] / num_samples
    v = radical_inverse(PRIMES[0], i) + offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = math.acos(1 - 2 * u) - math.pi / 2
    phi = v * 2 * math.pi
    return phi, theta


def camera(i, views, fov_deg, offset):
    yaw, pitch = sphere_hammersley(i, views, offset)
    radius = math.sqrt(3) / 2 / math.sin(math.radians(fov_deg) / 2)
    eye = np.array([math.cos(yaw) * math.cos(pitch),
                    math.sin(yaw) * math.cos(pitch),
                    math.sin(pitch)]) * radius
    return eye, yaw, pitch, radius


def normalise(verts):
    """Into the unit cube, centred, which is what the radius formula assumes."""
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    centre = (lo + hi) / 2
    scale = 1.0 / float((hi - lo).max())
    return (verts - centre) * scale, centre, scale


def vertex_normals(verts, faces):
    """Area-weighted, so the body shades smoothly rather than as 27,420 facets."""
    n = np.zeros_like(verts, dtype=np.float64)
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for i in range(3):
        np.add.at(n, faces[:, i], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    # 5,440 of the 19,158 vertices touch no face: the mesh carries the whole basemesh while
    # `Anny.faces` is the body submodel. Nothing rasterises them, so they take +Z, and the
    # count is printed because an unreferenced vertex quietly turned into NaN is how a
    # renderer produces black pixels nobody can explain.
    orphan = ln[:, 0] == 0
    n[orphan], ln[orphan] = (0.0, 0.0, 1.0), 1.0
    return (n / ln).astype(np.float32), int(orphan.sum())


def render(mesh_npz, out_png, index, views, fov_deg, offset, spp, threads, variant):
    import drjit as dr
    import mitsuba as mi

    # METAL IS REACHABLE AND IS DELIBERATELY NOT IN THE FALLBACK ORDER.
    #
    # This list read ("llvm_ad_rgb", "cuda_ad_rgb", "scalar_rgb"), so on Apple silicon it fell
    # through to the CPU and no caller could ask for anything else. Mitsuba 3.9.1 enumerates
    # `metal_ad_rgb` here, and on this film it is worth 60x: 545 ms/image against 32,592 for
    # llvm at one thread, measured by `logbook/scripts/mi_bench_llvm.py` on an M2 Pro.
    #
    # It is still not the default, and the reason is measured rather than cautious. Three
    # renders of this scene at a PINNED seed produced three different sha256 digests, so the
    # divergence is GPU accumulation order and not a seed artefact. A corpus renderer that
    # cannot reproduce a frame is not a corpus renderer, whatever it costs.
    #
    # So `--variant metal_ad_rgb` reaches it for anything that does not need reproducibility --
    # a preview, a look test, a cheap sanity render -- and the default order still lands on the
    # variant the determinism measurement covers. The point is that the choice is now a choice.
    for v in ([variant] if variant else ("llvm_ad_rgb", "cuda_ad_rgb", "scalar_rgb")):
        if v in mi.variants():
            mi.set_variant(v)
            break
    # One thread is what makes this bit-reproducible: parallel llvm and cuda each differ run
    # to run by up to 1/255 on a dozen pixels of a million, and metal differs at any thread
    # count. DRJIT_NUM_THREADS in the environment does not do it; this call does.
    if threads:
        dr.set_thread_count(threads)

    data = np.load(mesh_npz)
    verts, centre, scale = normalise(np.asarray(data["verts"], dtype=np.float64))
    faces = np.asarray(data["faces"], dtype=np.int64)
    normals, orphans = vertex_normals(verts, faces)
    eye, yaw, pitch, radius = camera(index, views, fov_deg, offset)

    mesh = mi.Mesh("body", vertex_count=verts.shape[0], face_count=faces.shape[0],
                   has_vertex_normals=True, has_vertex_texcoords=False)
    mp = mi.traverse(mesh)
    mp["vertex_positions"] = mi.Float(verts.astype(np.float32).reshape(-1))
    mp["faces"] = mi.UInt(faces.astype(np.uint32).reshape(-1))
    mp["vertex_normals"] = mi.Float(normals.reshape(-1))
    mp.update()

    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 6},
        "sensor": {
            "type": "perspective", "fov": fov_deg, "fov_axis": "y",
            "to_world": mi.ScalarTransform4f().look_at(
                origin=[float(x) for x in eye], target=[0, 0, 0], up=[0, 0, 1]),
            "film": {"type": "hdrfilm", "width": 1024, "height": 1024,
                     "rfilter": {"type": "gaussian"}, "pixel_format": "rgb"},
            "sampler": {"type": "independent", "sample_count": spp},
        },
        "body": mesh,
        "bsdf_body": {"type": "ref", "id": "skin"},
        "skin": {"type": "principled", "base_color": {"type": "rgb", "value": [0.76, 0.62, 0.54]},
                 "roughness": 0.55, "metallic": 0.0},
        "world": {"type": "constant", "radiance": {"type": "rgb", "value": 0.35}},
        "key": {"type": "point", "position": [float(eye[0]) * 1.2, float(eye[1]) * 1.2,
                                              float(eye[2]) + 1.5],
                "intensity": {"type": "rgb", "value": 12.0}},
    })

    img = mi.render(scene, spp=spp, seed=0)

    # THE MATTE IS A SECOND PASS, BECAUSE THAT IS WHAT UPSTREAM'S RENDERER DOES.
    #
    # Pixal3D's `preprocess_image` uses an image's alpha when it has one and otherwise runs
    # `briaai/RMBG-2.0`, which is gated and non-commercial and blocklisted here. A render
    # needs no matting model: the silhouette is which rays hit the body. The question is only
    # how to say that in the same terms upstream does.
    #
    # Upstream's `render_cond.py` sets `color_mode = 'RGBA'` and `film_transparent = True`,
    # so the world lights the object and does not appear to camera rays, and the alpha it
    # writes is antialiased coverage. Mitsuba has no equivalent flag, and measuring says why
    # it matters: with a constant emitter and `pixel_format: rgba`, alpha is 1.000 on every
    # pixel, because an infinite emitter counts as a hit. Native alpha is not the same thing.
    #
    # A first version derived alpha from a depth AOV, which gives 0 or 1 and nothing between.
    # That loses the edge: on a 256x256 test the matte pass finds 1,251 partially covered
    # pixels, and thresholding depth calls every one of them solid. So the matte is rendered
    # separately with the emitters removed, where a missed ray returns nothing and coverage
    # comes out fractional exactly as Blender's does.
    #
    # It costs a second render. `max_depth` is 2 because nothing beyond the first hit changes
    # whether a ray hit, and the sample count matches the colour pass so the two agree at the
    # edges rather than nearly agreeing.
    matte_scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 2},
        "sensor": {
            "type": "perspective", "fov": fov_deg, "fov_axis": "y",
            "to_world": mi.ScalarTransform4f().look_at(
                origin=[float(x) for x in eye], target=[0, 0, 0], up=[0, 0, 1]),
            "film": {"type": "hdrfilm", "width": 1024, "height": 1024,
                     "rfilter": {"type": "gaussian"}, "pixel_format": "rgba"},
            "sampler": {"type": "independent", "sample_count": spp},
        },
        "body": mesh,
    })
    alpha = np.array(mi.render(matte_scene, spp=spp, seed=0))[:, :, 3]
    alpha = np.clip(alpha, 0.0, 1.0)

    rgb = np.clip(np.array(img)[:, :, :3], 0.0, None)
    rgba = np.dstack([rgb, alpha]).astype(np.float32)
    covered = int((alpha > 0.5).sum())
    partial = int(((alpha > 0.01) & (alpha < 0.99)).sum())
    if covered == 0:
        raise SystemExit("no pixel hit the body: the camera is pointed at nothing")

    bmp = mi.Bitmap(rgba, pixel_format=mi.Bitmap.PixelFormat.RGBA)
    bmp = bmp.convert(mi.Bitmap.PixelFormat.RGBA, mi.Struct.Type.UInt8, srgb_gamma=True)
    # Synchronous: the JIT variants schedule the write and return, so hashing straight after
    # hashes a file that is not there yet.
    bmp.write(str(out_png))

    digest = hashlib.sha256(pathlib.Path(out_png).read_bytes()).hexdigest()
    side = {
        "generator": "sphere_hammersley_sequence, TencentARC/Pixal3D @ cdbb2bb",
        "index": index, "views": views, "offset": list(offset),
        "yaw_rad": yaw, "pitch_rad": pitch, "pitch_deg": math.degrees(pitch),
        "yaw_deg": math.degrees(yaw), "radius": radius,
        "fov_deg": fov_deg, "camera_angle_x_rad": math.radians(fov_deg),
        "eye": [float(x) for x in eye],
        "normalisation": {"centre": [float(x) for x in centre], "scale": scale},
        "vertices_without_faces": orphans,
        "spp": spp, "threads": threads, "variant": mi.variant(), "sha256": digest,
        "alpha": "matte pass with emitters removed, matching upstream film_transparent",
        "covered_pixels": covered,
        "partial_coverage_pixels": partial,
    }
    pathlib.Path(out_png).with_suffix(".json").write_text(json.dumps(side, indent=2))
    return side


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh_npz")
    ap.add_argument("out_png")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--fov", type=float, default=40.0, help="degrees; radius follows from it")
    ap.add_argument("--offset", type=float, nargs=2, default=(0.0, 0.0))
    ap.add_argument("--spp", type=int, default=128)
    ap.add_argument("--threads", type=int, default=1)
    # `metal_ad_rgb` is accepted and is NOT reproducible -- see the note in render(). Anything
    # writing corpus data wants the default.
    ap.add_argument("--variant", default="llvm_ad_rgb")
    a = ap.parse_args(argv[1:])

    s = render(a.mesh_npz, pathlib.Path(a.out_png), a.index, a.views, a.fov,
               tuple(a.offset), a.spp, a.threads, a.variant)
    print(f"  view {a.index}/{a.views}  yaw {s['yaw_deg']:7.2f}  pitch {s['pitch_deg']:6.2f}  "
          f"radius {s['radius']:.3f}  fov {s['fov_deg']}  sha256 {s['sha256'][:12]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
