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

**THE ONE REAL DIFFERENCE IS A FACTOR OF 1/PI.** Before scaling, the ratio of their pixels
to ours is a near-constant 0.3167 to 0.3183 against 1/pi = 0.31831, flat across every
toony value. The VRM 1.0 pseudocode ends `color = color * lightColor` with no such term;
three.js applies the Lambertian convention to direct light. So a corpus rendered with our
integrator sits pi times brighter than the same material in a three-vrm viewer.

That is not cosmetic for this corpus. The tone ladder solves for a dE between the lit and
shade plateaus, and CIELAB is not scale invariant, so the same material measured at two
exposures does not give the same dE. The targets have to be verified at the exposure the
viewer actually uses rather than at ours.
