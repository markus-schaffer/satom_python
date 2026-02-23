"""
SAtom – Sensitivity Analysis using the Kolmogorov-Smirnov 2-sample test (TOM method).

Translated from MATLAB (SAtom v1.0.0) by Torben Østergård and Markus Schaffer, Aalborg University.

References
----------
[1] Østergård, T., Jensen, R.L., and Maagaard, S.E. (2017)
    Interactive Building Design Space Exploration Using Regionalized
    Sensitivity Analysis. Proc. 15th IBPSA, San Francisco, USA.

License
-------
BSD 2-Clause – see LICENSE file for full text.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

try:
    from numba import njit

    _HAS_NUMBA = True
except ModuleNotFoundError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # noqa: D103 – stub
        """Identity decorator when Numba is not installed."""
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


# =========================================================================
# Numba-accelerated KS row computation
# =========================================================================


@njit(cache=True)
def _ks_row_mode01(mask, orderMat, uniformCDF, N, nIn):
    """KS row for SScompare 0 or 1 (one-sample vs uniform).

    Fuses indexing + cumsum + max into a single pass per column,
    eliminating temporary array allocations.
    """
    ks_row = np.empty(nIn, dtype=np.float64)
    n2 = 0.0
    for i in range(N):
        if mask[i]:
            n2 += 1.0
    for col in range(nIn):
        running_sum = 0.0
        max_d = 0.0
        for row in range(N):
            idx = orderMat[row, col]
            if mask[idx]:
                running_sum += 1.0
            cdf2 = running_sum / n2
            d = abs(uniformCDF[row] - cdf2)
            if d > max_d:
                max_d = d
        ks_row[col] = max_d
    return ks_row


@njit(cache=True)
def _ks_row_mode2(maskB, maskN, orderMat, N, nIn):
    """KS row for SScompare 2 (behavioural vs non-behavioural).

    Fuses both CDFs + max into a single pass per column.
    """
    ks_row = np.empty(nIn, dtype=np.float64)
    n1 = 0.0
    n2 = 0.0
    for i in range(N):
        if maskB[i]:
            n1 += 1.0
        if maskN[i]:
            n2 += 1.0
    for col in range(nIn):
        sum1 = 0.0
        sum2 = 0.0
        max_d = 0.0
        for row in range(N):
            idx = orderMat[row, col]
            if maskB[idx]:
                sum1 += 1.0
            if maskN[idx]:
                sum2 += 1.0
            d = abs(sum1 / n1 - sum2 / n2)
            if d > max_d:
                max_d = d
        ks_row[col] = max_d
    return ks_row


@njit(cache=True)
def _build_mask(randStart, YsortedIdx, N, nOut, subsetSize):
    """Build the behavioural mask for one iteration (Numba-accelerated).

    Parameters
    ----------
    randStart : 1-D int array of length nOut, 1-based start positions.
    YsortedIdx : 2-D int array (nOut, N), 0-based original row indices
                 sorted by each output.
    N, nOut, subsetSize : int

    Returns
    -------
    maskB : bool array of length N
    remaining : int – number of True values in maskB
    """
    maskB = np.ones(N, dtype=np.bool_)
    for idxY in range(nOut):
        idxStart = randStart[idxY]  # 1-based
        maskTmp = np.zeros(N, dtype=np.bool_)
        for k in range(subsetSize + 1):
            sortedPos = (idxStart - 1 + k) % N
            origIdx = YsortedIdx[idxY, sortedPos]
            maskTmp[origIdx] = True
        for i in range(N):
            if not maskTmp[i]:
                maskB[i] = False
    remaining = 0
    for i in range(N):
        if maskB[i]:
            remaining += 1
    return maskB, remaining


# =========================================================================
# Helper: Switch matplotlib backend
# =========================================================================


def _switch_backend(gui: str) -> None:
    """Force-switch the matplotlib backend, reimporting pyplot."""
    global plt
    with contextlib.suppress(ValueError):
        matplotlib.use(gui, force=True)
    plt = matplotlib.pyplot


def SAtom(
    X: ArrayLike,
    Y: ArrayLike,
    dummyYN: bool = False,
    J: int = 100,
    checkInterval: int = 0,
    N: int | None = None,
    plotYN: bool = False,
    SScompare: int = 0,
    X_label: Sequence[str] | None = None,
    seed: int = 42,
    use_numba: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """TOM sensitivity analysis using the Kolmogorov-Smirnov 2-sample test.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_inputs)
        Input matrix – each column is one input variable.
    Y : array-like, shape (n_samples,) or (n_samples, n_outputs)
        Output matrix – each column is one output variable.
    dummyYN : bool
        If ``True`` a dummy (random-permutation) column is appended to *X*.
        Default: False
    J : int
        Maximum number of repeated randomly selected subsamples.  When
        *checkInterval* > 0, this serves as the upper limit; the user can
        stop earlier once convergence is visually confirmed.  Default: 100
    checkInterval : int
        Interval (in completed iterations) at which an interactive
        convergence plot is shown.  Set to 0 to disable.  Default: 0
    N : int or None
        Number of rows to use.  ``None`` → ``X.shape[0]``.  Default: None
    plotYN : bool
        Plot figures at the end.  Default: False
    SScompare : int
        Subsets used to calculate the maximum KS2 distance *d*:
        ``0`` – Non-behavioural vs. all (default),
        ``1`` – Behavioural vs. all,
        ``2`` – Non-behavioural vs. behavioural.
    X_label : sequence of str or None
        Labels for each input variable in *X*.  Default: None (auto-generated).
    seed : int
        Random seed for reproducibility.  Default: 42
    use_numba : bool
        If ``True`` (default), use Numba JIT-compiled inner loop for
        faster KS computation.  Set to ``False`` to use the pure-NumPy
        fallback (useful for benchmarking or environments without Numba).

    Returns
    -------
    KS2_mean : np.ndarray, shape (n_inputs,)
        Mean KS2 distances across all completed repetitions.
    KS2 : np.ndarray, shape (actual_J, n_inputs)
        Raw KS2 distance matrix for every completed repetition.

    Examples
    --------
    >>> from satom import SAtom
    >>> KS2_mean, KS2 = SAtom(X, Y[:, 0], J=1000, seed=42)
    >>> KS2_mean, KS2 = SAtom(X, Y[:, 0], J=200, plotYN=True, SScompare=1)

    See also: scipy.stats.ks_2samp
    """

    # Constants
    J_MIN = 100

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("X must be a 2-D array.")
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y matrices must have the same number of rows!")

    if N is None:
        N = X.shape[0]
    if not isinstance(N, (int, np.integer)) or N <= 0 or N != int(N):
        raise ValueError("N must be a positive integer.")
    if N > X.shape[0]:
        raise ValueError(
            f"N ({N}) cannot be greater than the number of rows in X ({X.shape[0]})."
        )

    if not isinstance(J, (int, np.integer)) or J <= 0 or J != int(J):
        raise ValueError("J must be a positive integer.")

    if SScompare not in (0, 1, 2):
        raise ValueError("SScompare must be 0, 1, or 2.")

    if (
        not isinstance(checkInterval, (int, np.integer))
        or checkInterval < 0
        or checkInterval != int(checkInterval)
    ):
        raise ValueError("checkInterval must be a non-negative integer.")

    # Resolve Numba availability
    if use_numba and not _HAS_NUMBA:
        warnings.warn(
            "Numba is not installed – falling back to pure-NumPy mode. "
            "Install numba (pip install satom[fast]) for ~5-10× speedup.",
            stacklevel=2,
        )
        use_numba = False

    # Subset data (double precision throughout)
    X = X[:N].copy()
    Y = Y[:N].copy()
    nIn = X.shape[1]
    nOut = Y.shape[1]

    # ------------------------------------------------------------------
    # Predictor names
    # ------------------------------------------------------------------
    if X_label is None or (isinstance(X_label, (list, tuple)) and len(X_label) == 0):
        predictorNames = [f"x{i + 1}" for i in range(nIn)]
    else:
        predictorNames = list(X_label)
        if len(predictorNames) != nIn:
            raise ValueError(
                f"X_label must have {nIn} elements to match the number of "
                f"input columns."
            )

    # ------------------------------------------------------------------
    # Initialize random stream for reproducibility
    # ------------------------------------------------------------------
    rng = np.random.RandomState(seed)

    # Warn if J is below minimum and adjust
    if J < J_MIN:
        warnings.warn(
            f"J ({J}) is below minimum value. Increasing J to {J_MIN} for "
            f"convergence.",
            stacklevel=2,
        )
        J = J_MIN

    # ------------------------------------------------------------------
    # Add dummy variable (MATLAB: randperm(N) → 1-based [1..N])
    # ------------------------------------------------------------------
    if dummyYN:
        dummy_col = (rng.permutation(N) + 1).astype(np.float64)[:, np.newaxis]
        X = np.hstack([X, dummy_col])
        nIn += 1
        predictorNames.append("dummy")

    # ------------------------------------------------------------------
    # Pre-computations
    # ------------------------------------------------------------------
    KS = np.zeros((J, nIn), dtype=np.float64)

    subsetSize = max(1, int(np.floor(0.5 ** (1.0 / nOut) * N)))

    # Precompute sorted index arrays for each output
    # YsortedIdx[idxY, k] = 0-based original row at position k when
    # output idxY is sorted ascending
    YsortedIdx = np.empty((nOut, N), dtype=np.intp)
    for idxY in range(nOut):
        YsortedIdx[idxY, :] = np.argsort(Y[:, idxY], kind="mergesort")

    # Precompute sorted indices for each input column
    orderMat = np.empty((N, nIn), dtype=np.uint32)
    for i in range(nIn):
        orderMat[:, i] = np.argsort(X[:, i], kind="mergesort")

    # Precompute uniform CDF for SScompare 0 and 1
    uniformCDF: np.ndarray | None = None
    if SScompare in (0, 1):
        uniformCDF = np.arange(1, N + 1, dtype=np.float64) / float(N)

    # Cache-aware chunk size for NumPy fallback (8 bytes per float64)
    if not use_numba:
        cacheBytes = 32e6
        chunkSize = max(1, int(np.floor(cacheBytes / (8 * N * 2))))

    # MATLAB: randStarts = ceil(rand(J*10, nOut) * N) → 1-based [1..N]
    randStarts = np.ceil(rng.rand(J * 10, nOut) * N).astype(np.intp)

    rep = 0
    randIdx = 0
    maxAttempts = J * 10
    attempts = 0
    userStopped = False

    # ------------------------------------------------------------------
    # Interactive convergence setup
    # ------------------------------------------------------------------
    originalBackend = None
    convFig: plt.Figure | None = None
    convAx: plt.Axes | None = None
    if checkInterval > 0:
        originalBackend = matplotlib.get_backend()
        _switch_backend("QtAgg")
        plt.ion()
        convFig, convAx = plt.subplots(1, 1, num="TOM: Interactive Convergence Check")
        nextCheck = checkInterval

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while rep < J and attempts < maxAttempts and not userStopped:
        attempts += 1

        # Grow random-start buffer if exhausted
        if randIdx >= randStarts.shape[0]:
            newStarts = np.ceil(rng.rand(J * 10, nOut) * N).astype(np.intp)
            randStarts = np.concatenate([randStarts, newStarts], axis=0)

        # Build behavioural mask
        if use_numba:
            maskB, remaining = _build_mask(
                randStarts[randIdx], YsortedIdx, N, nOut, subsetSize
            )
        else:
            maskB = np.ones(N, dtype=bool)
            for idxY in range(nOut):
                idxStart = randStarts[randIdx, idxY]
                sortedPositions = np.mod(
                    np.arange(idxStart - 1, idxStart - 1 + subsetSize + 1), N
                )
                origIndices = YsortedIdx[idxY, sortedPositions]
                maskTmp = np.zeros(N, dtype=bool)
                maskTmp[origIndices] = True
                maskB &= maskTmp
            remaining = int(maskB.sum())

        randIdx += 1

        if remaining == 0 or remaining == N:
            continue

        # --- Compute KS row -----------------------------------------------
        if use_numba:
            # Numba-accelerated: single fused pass per column
            if SScompare == 0:  # Non-behavioural vs. all
                maskN = ~maskB
                KS[rep, :] = _ks_row_mode01(maskN, orderMat, uniformCDF, N, nIn)
            elif SScompare == 1:  # Behavioural vs. all
                KS[rep, :] = _ks_row_mode01(maskB, orderMat, uniformCDF, N, nIn)
            else:  # SScompare == 2
                maskN = ~maskB
                KS[rep, :] = _ks_row_mode2(maskB, maskN, orderMat, N, nIn)
        else:
            # Pure-NumPy fallback with cache-aware chunking
            if SScompare == 0:  # Non-behavioural vs. all
                maskN = ~maskB
                n2 = float(N - remaining)
                for colStart in range(0, nIn, chunkSize):
                    colEnd = min(colStart + chunkSize, nIn)
                    cols = slice(colStart, colEnd)
                    m2_chunk = maskN[orderMat[:, cols]]
                    cdf2_chunk = np.cumsum(m2_chunk, axis=0, dtype=np.float64) / n2
                    KS[rep, cols] = np.max(
                        np.abs(uniformCDF[:, np.newaxis] - cdf2_chunk), axis=0
                    )
            elif SScompare == 1:  # Behavioural vs. all
                n2 = float(remaining)
                for colStart in range(0, nIn, chunkSize):
                    colEnd = min(colStart + chunkSize, nIn)
                    cols = slice(colStart, colEnd)
                    m2_chunk = maskB[orderMat[:, cols]]
                    cdf2_chunk = np.cumsum(m2_chunk, axis=0, dtype=np.float64) / n2
                    KS[rep, cols] = np.max(
                        np.abs(uniformCDF[:, np.newaxis] - cdf2_chunk), axis=0
                    )
            else:  # SScompare == 2
                maskN = ~maskB
                n1 = float(remaining)
                n2 = float(N - remaining)
                for colStart in range(0, nIn, chunkSize):
                    colEnd = min(colStart + chunkSize, nIn)
                    cols = slice(colStart, colEnd)
                    m1_chunk = maskB[orderMat[:, cols]]
                    m2_chunk = maskN[orderMat[:, cols]]
                    cdf1_chunk = np.cumsum(m1_chunk, axis=0, dtype=np.float64) / n1
                    cdf2_chunk = np.cumsum(m2_chunk, axis=0, dtype=np.float64) / n2
                    KS[rep, cols] = np.max(np.abs(cdf1_chunk - cdf2_chunk), axis=0)

        rep += 1

        # Interactive checkpoint
        if checkInterval > 0 and rep == nextCheck:
            userStopped = _convergence_checkpoint(
                KS, rep, predictorNames, convFig, convAx
            )
            nextCheck = rep + checkInterval

    # ------------------------------------------------------------------
    # Post-loop
    # ------------------------------------------------------------------
    if rep == 0:
        raise RuntimeError(
            "No valid iterations completed. "
            "All randomly selected behavioral subsets were empty."
        )

    actualJ = rep

    if userStopped:
        print(f"Stopped by user at J = {actualJ} (convergence confirmed).")
    elif attempts >= maxAttempts:
        warnings.warn(
            f"Maximum attempts reached before completing all J repetitions. "
            f"Completed {actualJ} of {J}.",
            stacklevel=2,
        )

    # Close interactive figure and restore original backend
    if convFig is not None:
        plt.close(convFig)
        plt.ioff()
    if originalBackend is not None:
        _switch_backend(originalBackend)

    # Trim to completed iterations
    KS = KS[:actualJ]

    KS2_J_mean = (
        np.cumsum(KS, axis=0)
        / np.arange(1, actualJ + 1, dtype=np.float64)[:, np.newaxis]
    )

    # Output (already float64)
    KS2_mean = np.mean(KS, axis=0)
    KS2 = KS

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    if plotYN:
        rank = np.argsort(KS2_mean)[::-1]
        KS_ranked = KS2[:, rank]
        predictorNames_ranked = [predictorNames[i] for i in rank]

        fig_box, ax_box = plt.subplots(num="TOM: Boxplot, ranked")
        ax_box.boxplot(KS_ranked, tick_labels=predictorNames_ranked)
        ax_box.tick_params(axis="x", rotation=90)
        ax_box.set_ylabel(r"Smirnov distances $D_{ij}$")
        fig_box.tight_layout()

        KS2_J_mean_ranked = KS2_J_mean[:, rank]
        fig_conv, ax_conv = plt.subplots(num="TOM: Convergence J")
        for idx in range(KS2_J_mean_ranked.shape[1]):
            ax_conv.plot(
                KS2_J_mean_ranked[:, idx],
                linewidth=1,
                label=predictorNames_ranked[idx],
            )
        ax_conv.set_xlabel("Number of repetitions, $J$")
        ax_conv.set_ylabel(r"Mean of $D_{ij}$")
        if actualJ > 10:
            ax_conv.set_xlim(10, actualJ)
        ax_conv.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small")
        fig_conv.tight_layout()
        plt.show()

    return KS2_mean, KS2


# =========================================================================
# Helper: Interactive convergence checkpoint
# =========================================================================


def _convergence_checkpoint(
    KS: np.ndarray,
    currentJ: int,
    predictorNames: list[str],
    fig: plt.Figure,
    ax: plt.Axes,
) -> bool:
    """Display running convergence plot and ask user to continue or stop."""

    KS_partial = KS[:currentJ]
    cumMean = (
        np.cumsum(KS_partial, axis=0)
        / np.arange(1, currentJ + 1, dtype=np.float64)[:, np.newaxis]
    )

    currentMean = np.mean(KS_partial, axis=0)
    rank = np.argsort(currentMean)[::-1]
    cumMean_ranked = cumMean[:, rank]
    predictorNames_ranked = [predictorNames[i] for i in rank]

    ax.clear()
    for idx in range(cumMean_ranked.shape[1]):
        ax.plot(cumMean_ranked[:, idx], linewidth=1, label=predictorNames_ranked[idx])

    ax.set_xlabel("Number of repetitions, $J$")
    ax.set_ylabel(r"Cumulative mean of $D_{ij}$")
    ax.set_title(f"Convergence check at J = {currentJ}")
    if currentJ > 10:
        ax.set_xlim(10, currentJ)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small")
    fig.tight_layout()

    plt.show(block=False)
    plt.pause(0.5)

    answer = input(
        f"\nJ = {currentJ} completed. "
        "Inspect the convergence plot.\n"
        "Type 'stop' to finish or press Enter to continue: "
    )
    return answer.strip().lower() == "stop"
