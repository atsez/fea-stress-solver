# 2D finite element stress solver with a Gaussian process surrogate

A plane-stress finite element solver written from scratch in Python (NumPy, SciPy), verified against
exact solutions, validated against handbook stress concentration data, and used to train a Gaussian
process surrogate for elliptical holes in finite-width plates.

## Headline results

- **Verified.** Patch test passes to machine precision, Q8 elements reproduce pure bending exactly,
  and on the exact Kirsch problem the energy-norm error converges at order 0.997 (Q4) and 1.999 (Q8)
  against theory 1 and 2. 16 automated tests cover all of this.
- **Validated.** Net-section stress concentration factors for a finite-width plate sit within 1.21%
  of Howland's solution across d/W = 0.05 to 0.6, inside the 2.05% spread between the three
  handbook references themselves. Elliptical holes match Inglis to within 0.5%.
- **Fast.** Vectorised element assembly into a sparse global matrix solves 77,090 degrees of freedom
  (12,672 Q8 elements) in 5.6 seconds in pure Python.
- **Surrogate.** A Gaussian process trained on 60 solver runs (Latin hypercube) predicts 40 held-out
  cases within 0.02%, with every test point inside its 95% band. A prediction takes 8.7 microseconds
  against 0.77 seconds for a full solve, about 88,790 times faster.

Full tables are in [`out/results.md`](out/results.md).

## Layout

| File | What it does |
|---|---|
| `fea/elements.py` | Q4 and Q8 isoparametric shape functions, Gauss rules, edge shape functions |
| `fea/solver.py` | Vectorised element stiffness, sparse assembly, traction loads, boundary conditions, sparse solve, nodal stress recovery, energy-norm error |
| `fea/mesh.py` | Structured mesher for a quarter plate with a circular or elliptical hole, graded towards the hole; rectangle and distorted patch meshes |
| `fea/analytic.py` | Kirsch, Inglis, Howland (Peterson fit), Heywood and handbook polynomial references, exact pure bending |
| `fea/problems.py` | Patch test, pure bending, exact Kirsch problem, finite-width plate |
| `tests/` | 16 pytest checks (shape functions, rigid body modes, patch test, bending, equilibrium of the Kirsch field, convergence order, global equilibrium) |
| `study.py` | Every study, written to `out/results.json` and CSVs |
| `report.py` | Charts and `out/results.md`, every number read from the run files |
| `run_all.sh` | Tests, studies, report in one command (about 2.5 minutes) |

## Reproduce

```
pip install -r requirements.txt
./run_all.sh
```

## The method in plain English

- **Elements.** Each element is a square in its own coordinates (xi, eta from -1 to 1). Shape functions blend node values into a field inside it, and the same functions map the square onto the real element, which is what isoparametric means. Q4 has 4 corner nodes and a bilinear field. Q8 adds mid-side nodes, so its field is quadratic and its edges can follow a curved hole.
- **Stiffness.** Strain comes from displacement through the B matrix, stress from strain through the plane-stress D matrix, and the element stiffness is the integral of B'DB over the element. Gauss quadrature evaluates that integral exactly for the polynomials involved (2 x 2 points for Q4, 3 x 3 for Q8). All elements are processed at once with NumPy, then assembled into a sparse global matrix.
- **Loads and supports.** A traction on an edge is turned into consistent nodal forces by integrating it against the edge shape functions. A quarter of the plate is modelled, with symmetry supports on the two cut lines.
- **Stress recovery.** Stresses are evaluated at each element's nodes and averaged between neighbouring elements.
- **Verification vs validation.** Verification asks whether the code solves the equations correctly (patch test, exact bending, exact Kirsch problem with measured convergence order). Validation asks whether the answer is right for a real engineering geometry (finite-width plates against Howland and Heywood, ellipses against Inglis).
- **Energy-norm error.** The square root of the strain energy of the stress error, relative to the strain energy of the exact field. Theory says it falls in proportion to h for Q4 and h squared for Q8, where h is element size.
- **Shear locking.** A bilinear Q4 element cannot bend without also shearing, so in bending it is far too stiff. Q8 does not have this problem. The bending test shows both.
- **Surrogate.** The verified solver generates training data over hole size and ellipse shape. A Gaussian process learns the map from geometry to Kt and reports its own uncertainty. It is tested on 40 solves it never saw.

## Limits

Linear elastic, plane stress, small strain, static load. Kt values are elastic stress concentration
factors, so they do not include plasticity, notch sensitivity or fatigue effects. Kt is a smooth
function of two inputs, so the surrogate shows the workflow (verified solver, design of experiments,
held-out test, calibrated uncertainty) rather than a hard learning problem.
