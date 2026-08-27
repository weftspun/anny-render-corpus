# mtoon-reference

The differential test behind `mtoon.py`: our MToon 1.0 against `@pixiv/three-vrm`, the
reference implementation, rendered in headless Chromium and compared pixel for pixel.

    npm init -y
    npm install three @pixiv/three-vrm playwright
    npx playwright install chromium
    python ../check_mtoon_reference.py --self-test

The camera is orthographic and framed exactly to the unit sphere, so pixel (u, v) carries
normal (u, v, sqrt(1 - u^2 - v^2)). The comparison needs no plateau finding and no fitting.

## What it measured

three 0.185.1, three-vrm 3.5.5, chromium via swiftshader.

| shadingToonyFactor | px compared | over 4/255 | p99 | max |
| ------------------ | ----------- | ---------- | ------ | ------ |
| 0.9                | 3048        | 0          | 0.0020 | 0.0025 |
| 0.5                | 3048        | 0          | 0.0020 | 0.0021 |
| 0.0                | 3048        | 0          | 0.0020 | 0.0021 |
| 1.0                | 3048        | 2 (0.07%)  | 0.0018 | 0.1351 |

One 8-bit quantisation step is 1/255 = 0.0039, so three of the four agree below the noise
floor of the readback.

**The two pixels at toony 1.0 are the terminator, not the model.** At a hard ramp the
sphere's tessellated normal disagrees with the analytic one by a hair and the step
amplifies it into the full base-to-shade gap. That gap over pi is 0.1337 against a measured
max of 0.1351, which is the cost of one flipped pixel and not a difference in shading.

**THE ONE REAL DIFFERENCE IS A FACTOR OF 1/PI, AND IT IS THE SPEC THAT IS THE OUTLIER.**
Unscaled, their pixels to ours is a near-constant 0.3167 to 0.3183 against
1/pi = 0.31831, flat across every toony value. Two independent implementations apply it by
two different mechanisms: the Godot port writes `vec3 lighting = lightColor / 3.14159;`
into `mtoon_common.gdshaderinc:156` by hand, and three-vrm gets it from three.js's Lambert
BRDF as `RECIPROCAL_PI * diffuseColor`. The VRM 1.0 pseudocode ends
`color = color * lightColor` with no such term, so it is the spec text that omits what
every renderer does rather than a convention either engine invented.

A corpus rendered with our integrator therefore sits pi times brighter than the same
material in any viewer.

That is not cosmetic for this corpus. The tone ladder solves for a dE between the lit and
shade plateaus, and CIELAB is not scale invariant, so the same material measured at two
exposures does not give the same dE. The targets have to be verified at the exposure the
viewer actually uses rather than at ours.

## The context loss that made the mesh render black

Headless Chromium on swiftshader loses the WebGL context across an idle macrotask gap. The
mesh path fetches an OBJ, which is exactly such a gap, and every pixel came back black:
`isContextLost()` true and zero draw calls, with the geometry loaded and the camera correct.

It cost several wrong hypotheses -- backface winding, culling, the camera basis, the drawing
buffer -- and none of them were it. What settled it was a 50 ms `await` in the SPHERE path,
which took a working render from 3220 non-black pixels to zero without touching geometry or
camera at all. Rendering into a `WebGLRenderTarget` did not help either, so it is the context
rather than the drawing buffer.

**So every await in `page.html` happens before the renderer is constructed.** The mesh source
is fetched as text first and `OBJLoader.parse` runs synchronously afterwards. A future edit
that moves an `await` below the `new THREE.WebGLRenderer` line will silently return black
pixels, which is why the ordering carries a comment rather than being left to look arbitrary.

`draws` and `lost` are reported in every result so this failure names itself next time.

## The shape

`testshape.obj` is generated, not committed:

    python ../usda_to_obj.py ../../thebasemesh-stage/models/S_Abacus.usda testshape.obj --scale 2.0

CC0 public domain, 5,840 points and 9,880 triangles, beads occluding beads and a frame
casting onto rods. A sphere is convex and can never self-shadow, so it exercises no part of
the shadow path.

Coverage at 96 square: the sphere fills 78.6 per cent of frame, which is pi/4 and confirms
the unit framing; the abacus fills 47.1 per cent, and turning shadows on adds a second draw
call for the shadow map without changing coverage.
