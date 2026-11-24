# Step 1 prediction (written 2026-11-23, before measuring d(r) beyond r = 16)

**Question.** As distance grows, does the grown world's dimension keep rising toward 3,
plateau below 3, or exceed 3?

**What is already known.** On a 70,000-node world the volume-based estimate d_V(r) reached
about 2.87 at r = 15, tracking the 3D lattice's curve (2.90 at r = 15). The estimator itself
reads low at short range: the 3D lattice's true dimension is 3, yet d_V(5) = 2.69.

**Method fixed in advance.**
- Worlds: 4 independently grown worlds (seeds 0-3) at 22 generations (about 800,000
  nodes), at least 100 random centres each, r up to 30. Smaller worlds (18, 20 generations)
  on the same seeds to check that the curve does not depend on world size.
- Two estimators. d_V(r) = slope of log V(r) vs log r between r and r+1 (the original one).
  d_S(r) = 1 + slope of log S(r) vs log r over r-1..r+1, where S(r) is the number of nodes
  at distance exactly r. On lattices d_S has much less short-range bias (it is exactly 2 on
  the 2D lattice, and 2.96 at r = 5 on the 3D lattice).
- Lattice reference: exact counts on infinite 2D, 3D and 4D lattices (no sampling), plus
  the same sampled code on a periodic 3D lattice (as a code check) and on a 3D lattice
  diluted to degree 4 (the control used in the stiffness runs).
- Error bars: bootstrap over centres within each world; world-to-world spread shown
  separately.
- Saturation check: report V(30) as a fraction of the world. If it exceeds a few percent,
  grow a larger world before drawing conclusions.

**Expectation.** The rule attaches each new node to x and to two of x's neighbours, so
growth is local and makes many triangles; nothing in it obviously fixes the dimension at 3.
My guess: d keeps rising and levels off near 3 (within about ±0.2) — about 55%; keeps
rising past 3 (dimension growing with scale) — about 30%; plateaus clearly below 3 — about
15%. The decisive comparison is the grown curve against the 3D lattice curve measured by
the same estimator, not against the number 3.
