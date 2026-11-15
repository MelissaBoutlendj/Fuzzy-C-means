"""
Fuzzy C-means (FCM) lab -- M2 VMI, Advanced Pattern Recognition.

Usage:
    python3 fcm_tp.py                                        # default images
    python3 fcm_tp.py --gray milky-way-nvg.jpg --color milky-way.jpg

All figures are written to the results/ folder; numerical results are printed.
"""
import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, figures are saved to disk
import matplotlib.pyplot as plt
from skimage import io, color, data, img_as_float, transform


# ---------------------------------------------------------------------------
# 1. Fuzzy C-means algorithm (generic: works on any data X of shape (N, d))
# ---------------------------------------------------------------------------
def fcm(X, c, m=2.0, eps=1e-3, max_iter=300, seed=0):
    """
    Fuzzy C-means (Dunn 1973, Bezdek 1981), following the lab appendix.

    Parameters
    ----------
    X        : (N, d) array, N data points of dimension d
    c        : number of clusters
    m        : fuzzifier (m > 1); m -> 1 gives hard (k-means-like) memberships
    eps      : stopping criterion, max_ij |u_ij^(k+1) - u_ij^(k)| < eps
    max_iter : safety limit on the number of iterations
    seed     : random seed for the initialisation of U

    Returns
    -------
    C      : (c, d) cluster centres
    U      : (N, c) membership degrees, each row sums to 1
    n_iter : number of iterations performed
    J_hist : list of the objective J_m at each iteration
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:                      # 1-D data -> column vector
        X = X[:, None]
    N = X.shape[0]
    rng = np.random.default_rng(seed)

    # Step 1: random initialisation of U, rows normalised to sum to 1
    U = rng.random((N, c))
    U /= U.sum(axis=1, keepdims=True)

    J_hist = []
    for k in range(1, max_iter + 1):
        # Step 2: centres c_j = sum_i u_ij^m x_i / sum_i u_ij^m
        Um = U ** m
        C = (Um.T @ X) / Um.sum(axis=0)[:, None]

        # Distances ||x_i - c_j|| for all points and clusters, shape (N, c).
        # A small lower bound avoids division by zero when a point lies on a centre.
        D = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        D = np.fmax(D, 1e-10)

        # Objective function J_m = sum_ij u_ij^m ||x_i - c_j||^2 (monitoring only)
        J_hist.append(float((Um * D ** 2).sum()))

        # Step 3: memberships u_ij = 1 / sum_k (d_ij / d_ik)^(2/(m-1)).
        # Equivalent vectorised form: u_ij = d_ij^(-2/(m-1)) / sum_k d_ik^(-2/(m-1))
        inv = D ** (-2.0 / (m - 1.0))
        U_new = inv / inv.sum(axis=1, keepdims=True)

        # Step 4: stop when U no longer changes significantly
        delta = np.abs(U_new - U).max()
        U = U_new
        if delta < eps:
            break
    return C, U, k, J_hist


def sort_clusters(C, U):
    """Reorder clusters by increasing centre intensity (dark -> bright) for readability."""
    order = np.argsort(C.sum(axis=1))
    return C[order], U[:, order]


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def resize(img, max_side):
    """Downscale so that the largest side is at most max_side pixels (faster FCM)."""
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = transform.resize(img, (int(h * s), int(w * s)), anti_aliasing=True)
    return img


def load_gray(path, max_side=512):
    """Load a grey-level image in [0, 1]; fall back to a skimage test image."""
    if path and os.path.exists(path):
        img = img_as_float(io.imread(path))
        name = os.path.splitext(os.path.basename(path))[0]
    else:
        img = img_as_float(data.camera())
        name = "cameraman"
    if img.ndim == 3:                    # RGB file -> convert to grey levels
        img = color.rgb2gray(img[..., :3])
    return resize(img, max_side), name


def load_color(path, max_side=512):
    """Load an RGB image in [0, 1]; fall back to a skimage test image."""
    if path and os.path.exists(path):
        img = img_as_float(io.imread(path))
        name = os.path.splitext(os.path.basename(path))[0]
    else:
        img = img_as_float(data.coffee())
        name = "coffee"
    if img.ndim == 2:
        img = color.gray2rgb(img)
    return resize(img[..., :3], max_side), name


def savefig(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ->", path)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def exp_1d(out):
    """Sanity check on 1-D data, as in the appendix example (20 points, 3 clusters, m=2)."""
    print("\n[1] 1-D validation")
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 0.6, 7), rng.normal(5, 0.6, 7),
                        rng.normal(10, 0.6, 6)])
    xs = np.linspace(X.min() - 1, X.max() + 1, 400)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, eps in zip(axes, (0.3, 0.01)):
        C, U, it, J = fcm(X, c=3, m=2.0, eps=eps, seed=1)
        C, U = sort_clusters(C, U)
        # Properties of a fuzzy partition (appendix):
        assert np.allclose(U.sum(axis=1), 1)                     # sum_j u_ij = 1
        assert np.all((U.sum(axis=0) > 0) & (U.sum(axis=0) < len(X)))  # no empty cluster
        assert all(a >= b - 1e-9 for a, b in zip(J, J[1:]))      # J_m never increases
        print(f"  eps={eps}: {it} iterations, centres={np.round(C.ravel(), 2)}")
        # Continuous membership functions induced by the final centres (m = 2)
        Dg = np.fmax(np.abs(xs[:, None] - C.ravel()[None, :]), 1e-10)
        inv = Dg ** -2.0
        Ug = inv / inv.sum(axis=1, keepdims=True)
        for j, col in enumerate(["tab:blue", "tab:green", "tab:purple"]):
            ax.plot(xs, Ug[:, j], color=col)
            ax.axvline(C[j, 0], color=col, ls="--", lw=1)
        ax.scatter(X, np.zeros_like(X), c=U.argmax(1), cmap="viridis", zorder=3)
        ax.set_title(f"$\\varepsilon$={eps}: {it} iterations")
        ax.set_xlabel("x"); ax.set_ylabel("membership degree")
    fig.suptitle("1-D FCM (m = 2): membership functions of the 3 clusters")
    savefig(fig, out, "01_fcm_1d.png")


def exp_gray(img, name, out):
    """Grey-level segmentation with K=2 and K=3: membership heatmaps + crisp segmentation."""
    X = img.reshape(-1, 1)               # each pixel = one 1-D data point
    results = {}
    for K in (2, 3):
        print(f"\n[2] Grey levels, K={K}")
        t = time.time()
        C, U, it, J = fcm(X, c=K, m=2.0, eps=1e-3, seed=0)
        C, U = sort_clusters(C, U)
        print(f"  {it} iterations, {time.time()-t:.2f}s, centres={np.round(C.ravel(), 3)}")
        results[K] = (C, U, it, J)

        fig, axes = plt.subplots(1, K + 2, figsize=(3.4 * (K + 2), 3.4))
        axes[0].imshow(img, cmap="gray"); axes[0].set_title("Image")
        for j in range(K):
            im = axes[j + 1].imshow(U[:, j].reshape(img.shape), cmap="jet", vmin=0, vmax=1)
            axes[j + 1].set_title(f"$u_{{{j+1}}}$ (centre={C[j,0]:.2f})")
        fig.colorbar(im, ax=axes[1:K + 1], fraction=0.02)
        lab = U.argmax(1).reshape(img.shape)          # defuzzification by maximum membership
        axes[-1].imshow(lab, cmap="viridis"); axes[-1].set_title("Defuzzified (argmax)")
        for a in axes: a.axis("off")
        savefig(fig, out, f"02_{name}_K{K}_heatmaps.png")

        if K == 2:
            # Uncertainty map 1 - max_j u_ij (high near region boundaries) + convergence of J_m
            fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
            unc = 1 - U.max(1).reshape(img.shape)
            im = ax[0].imshow(unc, cmap="magma"); ax[0].set_title("Uncertainty 1 - max$_j$ u")
            fig.colorbar(im, ax=ax[0], fraction=0.046)
            ax[1].plot(J, marker="o", ms=3); ax[1].set_yscale("log")
            ax[1].set_title("Convergence of $J_m$"); ax[1].set_xlabel("iteration")
            ax[0].axis("off")
            savefig(fig, out, f"03_{name}_K2_uncertainty_convergence.png")
    return results


def exp_alpha(img, name, U, out):
    """Alpha-cuts of the brightest cluster (K=2): A_alpha = {x : u(x) >= alpha}."""
    print("\n[3] Alpha-cuts")
    u = U[:, -1].reshape(img.shape)
    alphas = [0.1, 0.5, 0.9]
    fig, axes = plt.subplots(1, len(alphas) + 1, figsize=(3.4 * (len(alphas) + 1), 3.4))
    axes[0].imshow(u, cmap="jet", vmin=0, vmax=1); axes[0].set_title("$u$ bright cluster")
    for a, alpha in zip(axes[1:], alphas):
        cut = u >= alpha
        a.imshow(cut, cmap="gray")
        a.set_title(f"$\\alpha$={alpha}: {100*cut.mean():.1f}% px")
        print(f"  alpha={alpha}: {100*cut.mean():.1f}% of the pixels")
    for a in axes: a.axis("off")
    savefig(fig, out, f"04_{name}_alpha_cuts.png")


def exp_m(img, name, out):
    """Influence of the fuzzifier m (K=2)."""
    print("\n[4] Influence of m")
    X = img.reshape(-1, 1)
    ms = [1.2, 1.5, 2.0, 3.0, 5.0]
    fig, axes = plt.subplots(1, len(ms), figsize=(3.2 * len(ms), 3.4))
    stats = []
    for a, m in zip(axes, ms):
        C, U, it, _ = fcm(X, c=2, m=m, eps=1e-3, seed=0)
        C, U = sort_clusters(C, U)
        frac_amb = np.mean(U.max(1) < 0.8)  # "ambiguous" pixels: no membership >= 0.8
        stats.append((m, it, frac_amb))
        print(f"  m={m}: {it} it., centres={np.round(C.ravel(),3)}, "
              f"ambiguous pixels (max u<0.8)={100*frac_amb:.1f}%")
        a.imshow(U[:, -1].reshape(img.shape), cmap="jet", vmin=0, vmax=1)
        a.set_title(f"m={m} ({100*frac_amb:.0f}% ambiguous)"); a.axis("off")
    savefig(fig, out, f"05_{name}_influence_m.png")
    return stats


def exp_eps_init(img, name, out):
    """Influence of epsilon and of the random initialisation (K=3)."""
    print("\n[5] Influence of epsilon and initialisation (K=3)")
    X = img.reshape(-1, 1)
    for eps in (0.3, 0.05, 0.01, 1e-3, 1e-4):
        t = time.time()
        C, U, it, J = fcm(X, c=3, m=2.0, eps=eps, seed=0)
        print(f"  eps={eps}: {it} it., J={J[-1]:.3f}, {time.time()-t:.2f}s")
    cents = []
    for seed in range(5):
        C, U, it, J = fcm(X, c=3, m=2.0, eps=1e-3, seed=seed)
        C, _ = sort_clusters(C, U)
        cents.append(C.ravel())
        print(f"  seed={seed}: {it} it., centres={np.round(C.ravel(),3)}, J={J[-1]:.3f}")
    cents = np.array(cents)
    print(f"  std of the centres across initialisations: {np.round(cents.std(0), 4)}")


def exp_color(img, name, out):
    """Colour image: each pixel is an RGB vector (d = 3), K=3. No change to fcm() needed."""
    print("\n[6] Colour image (RGB, d=3)")
    X = img.reshape(-1, 3)
    K = 3
    t = time.time()
    C, U, it, _ = fcm(X, c=K, m=2.0, eps=1e-3, seed=0)
    C, U = sort_clusters(C, U)
    print(f"  {it} iterations, {time.time()-t:.2f}s, centres (RGB)={np.round(C, 3).tolist()}")
    fig, axes = plt.subplots(1, K + 2, figsize=(3.4 * (K + 2), 3.4))
    axes[0].imshow(img); axes[0].set_title("Image")
    for j in range(K):
        axes[j + 1].imshow(U[:, j].reshape(img.shape[:2]), cmap="jet", vmin=0, vmax=1)
        axes[j + 1].set_title(f"$u_{{{j+1}}}$")
    # Fuzzy recolouring: each pixel takes the membership-weighted colour sum_j u_ij c_j
    recol = (U @ C).reshape(img.shape)
    axes[-1].imshow(np.clip(recol, 0, 1)); axes[-1].set_title("Recolouring $\\sum_j u_{ij} c_j$")
    for a in axes: a.axis("off")
    savefig(fig, out, f"06_{name}_colour_K{K}.png")


def main():
    p = argparse.ArgumentParser(description="Fuzzy C-means image segmentation lab")
    p.add_argument("--gray", default="milky-way-nvg.jpg", help="grey-level image")
    p.add_argument("--color", default="milky-way.jpg", help="colour image")
    p.add_argument("--out", default="results", help="output folder for figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    exp_1d(a.out)
    img, name = load_gray(a.gray)
    print(f"\nGrey-level image: {name}, size {img.shape}")
    res = exp_gray(img, name, a.out)
    exp_alpha(img, name, res[2][1], a.out)
    exp_m(img, name, a.out)
    exp_eps_init(img, name, a.out)
    cimg, cname = load_color(a.color)
    print(f"\nColour image: {cname}, size {cimg.shape}")
    exp_color(cimg, cname, a.out)


if __name__ == "__main__":
    main()
