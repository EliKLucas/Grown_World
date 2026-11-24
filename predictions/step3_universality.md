# Step 3 prediction (written 2026-11-23, before any grown-world stiffness data at R > 9)

**Question.** Does the grown world's stiffness exponent θ follow the dimension the world
actually has at each scale (universality), or does the growth process change the physics?

**Known when this was written.**
- Step 1: the grown world's dimension (shell estimator d_S) is 2.72 ± 0.03 at r = 3–6,
  2.89 ± 0.02 at r = 6–12, and 3.00 ± 0.03 from r = 12 to 29. At the same radii the 3D
  lattice reads 2.94, 2.99, 3.00, and the 3D lattice diluted to degree 4 reads 3.17, 3.17,
  3.10.
- Legacy ball-method θ (R = 3–9): 2D −0.17, grown +0.26, 3D +0.52, diluted +0.65. This is the
  same order as the effective dimensions at those radii. The grown value depends noticeably
  on the R = 9 point (+0.18 unweighted without it) and differs between worlds (−0.04 to
  +0.42, each ± 0.15).

**Prediction if universality holds.** The grown world's θ measured over sliding windows of R
rises with R. In windows reaching R ≳ 12, where its dimension has reached 3, it approaches
what the ball method gives for 3D lattices over the same R window (the method is biased, so
compare with the 3D lattice at the same R, not with +0.24).

**What would count against universality.** A grown-world θ that stays flat, or falls, across
windows while its dimension climbs from 2.7 to 3. Or a large-R θ clearly below the 3D lattice's
same-window value once both are three-dimensional. "Clearly" means the difference exceeds
twice its combined error.

**My expectation.** About 60%: θ rises with R as the dimension does, consistent with
universality. About 25%: inconclusive, because errors at R = 10–13 with 40 samples per point
will be large (variance ≈ mean), and the sliding windows may not separate. About 15%: θ
stays clearly below the 3D value at large R. That would be a real result: the grown world's
triangles, hubs or growth history would be changing the physics.

**Caveat committed in advance.** The ball method's θ is biased at small R. The 3D lattice
can only be solved exactly to R ≈ 6–7, while the grown world can reach R ≈ 13. So the
same-window comparison may only be possible up to R ≈ 7. Beyond that the comparison has to
go through the Step 2 calibration (method-θ against dimension), which will itself be
extrapolated.

## Addendum (2026-11-24, while Step 3 was running; before its analysis)

Written after Step 2 finished, with Step 3 partly done. What I had seen of Step 3 data at
this point: only a provisional R = 7–10 grown window (+1.1 ± 0.7, a handful of R = 10
samples) that appeared in a calibration test run. Nothing at R = 11–13.

1. **Expected precision.** At R ≤ 8 the grown world's |ΔE| has a standard deviation about
   equal to its mean, so log mean|ΔE| at a point with n samples has an error of about 1/√n.
   With the planned counts (R = 9–13: about 100, 100, 80, 48, 40) the weighted θ over
   R = 9–13 will carry an error of about ±0.45 before world-to-world spread, and
   three-radius windows at large R ±0.7–1.2. That cannot separate 0.24 (the published 3D
   value) from 0.5 (what the ball method gives for 3D at R ≤ 7). So the "inconclusive"
   outcome is likely on statistics alone, whatever the physics. Halving the R = 9–13 error
   would need about 4× the samples there (roughly 12 more hours on this machine).
2. **"θ rises with R" is not a clean signature.** Step 2 shows the ball method's θ for the
   2D lattice *falls* with R at fixed dimension (about 0 at R = 3–6, about −0.4 at
   R = 10–14). If the same small-size bias acts on the grown world, a falling bias and a
   rising dimension pull in opposite directions. The comparison that controls for this is
   the grown world's θ against the lattice line at the same window and effective dimension
   (Step 2's figure), which only exists up to R = 7. Beyond R = 7 there is no 3D-lattice
   comparison at all.
3. **Test committed now for large R.** Over R = 9–13 the grown θ is "consistent with 3D"
   if it lies within 2 errors of the range 0.24–0.5 (the published value to the ball
   method's small-R 3D reading). It counts against universality only if it is more than
   2 errors outside that range.
