# SAtom

**Sensitivity Analysis using the Kolmogorov-Smirnov 2-sample test (TOM method)**

A Python implementation of the SAtom method originally developed in MATLAB by
Torben Østergård and further developed by Markus Schaffer at Aalborg University.

## Installation

```bash
# From local source (editable / development mode)
pip install -e ./satom_python

# Or build and install
pip install ./satom_python
```

## Quick start

```python
import numpy as np
from satom import SAtom

# Example: Ishigami-Homma function (3 inputs, Uniform[-π, π])
rng = np.random.default_rng(123)
N = 10_000
X = rng.uniform(-np.pi, np.pi, size=(N, 3))
a, b = 2, 1
Y = np.sin(X[:, 0]) + a * np.sin(X[:, 1])**2 + b * X[:, 2]**4 * np.sin(X[:, 0])

# Run SAtom
KS2_mean, KS2 = SAtom(X, Y, J=2000, seed=42)

print("KS2 mean distances:", KS2_mean)
# → ranking reveals relative importance of each input
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | array (N, nIn) | required | Input sample matrix |
| `Y` | array (N,) or (N, nOut) | required | Output sample matrix |
| `dummyYN` | bool | `False` | Append a random-permutation dummy column |
| `J` | int | `100` | Number of random subsamples |
| `checkInterval` | int | `0` | Interactive convergence check interval (0 = off) |
| `N` | int or None | `None` | Number of rows to use (None = all) |
| `plotYN` | bool | `False` | Show boxplot and convergence figures |
| `SScompare` | int | `0` | Comparison mode: 0, 1, or 2 |
| `X_label` | list[str] or None | `None` | Labels for input variables |
| `seed` | int | `42` | Random seed for reproducibility |

## Returns

- **`KS2_mean`** – `np.ndarray (nIn,)` – Mean KS2 distance per input variable.
- **`KS2`** – `np.ndarray (J, nIn)` – Full KS2 distance matrix for all repetitions.

## References

> Østergård, T., Jensen, R.L., and Maagaard, S.E. (2017)
> *Interactive Building Design Space Exploration Using Regionalized
> Sensitivity Analysis.* Proc. 15th IBPSA, San Francisco, USA.

## License

BSD 2-Clause – see [LICENSE](LICENSE).
