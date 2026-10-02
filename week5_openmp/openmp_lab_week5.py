# ch 1
# import time
# import numpy as np
# import numba
# from numba import njit, prange

# print(f"Hardware Threads Detected: {numba.config.NUMBA_NUM_THREADS}")

# @njit(parallel=True)
# def monte_carlo_pi(n_samples):
#     inside_circle = 0

#     for i in prange(n_samples):
#         x = np.random.uniform(0.0, 1.0)
#         y = np.random.uniform(0.0, 1.0)

#         if x * x + y * y <= 1.0:
#             inside_circle += 1

#     return (4.0 * inside_circle) / n_samples


# # JIT Warmup
# _ = monte_carlo_pi(10_000)

# SAMPLES = 120_000_000

# thread_counts = [1, 2, 4, 8, numba.config.NUMBA_NUM_THREADS]

# # Remove duplicates and invalid thread counts
# thread_counts = sorted(
#     list(set([
#         t for t in thread_counts
#         if t <= numba.config.NUMBA_NUM_THREADS
#     ]))
# )

# print(f"{'Threads':<10} | {'Time (s)':<12} | {'Speedup':<10} | {'Efficiency (%)':<15}")
# print("-" * 55)

# t1_baseline = None

# for t in thread_counts:
#     numba.set_num_threads(t)

#     start = time.perf_counter()
#     pi_est = monte_carlo_pi(SAMPLES)
#     elapsed = time.perf_counter() - start

#     if t == 1:
#         t1_baseline = elapsed
#         speedup = 1.0
#         efficiency = 100.0
#     else:
#         speedup = t1_baseline / elapsed
#         efficiency = (speedup / t) * 100.0

#     print(
#         f"{t:<10} | {elapsed:<12.4f} | "
#         f"{speedup:<10.2f}x | {efficiency:<15.1f}"
#     )

# print(f"\nEstimated Pi: {pi_est}")


# ch 2
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# from numba import njit, prange


# @njit(parallel=True)
# def render_mandelbrot_rows(h, w, max_iter):
#     img = np.zeros((h, w), dtype=np.int32)

#     # Row-parallel loop
#     for r in prange(h):
#         cy = -1.2 + (r / h) * 2.4

#         for c in range(w):
#             cx = -2.0 + (c / w) * 2.5

#             z_real, z_imag = 0.0, 0.0
#             it = 0

#             while (
#                 z_real * z_real + z_imag * z_imag <= 4.0
#                 and it < max_iter
#             ):
#                 next_real = z_real * z_real - z_imag * z_imag + cx
#                 z_imag = 2.0 * z_real * z_imag + cy
#                 z_real = next_real
#                 it += 1

#             img[r, c] = it

#     return img


# @njit(parallel=True)
# def render_mandelbrot_cols(h, w, max_iter):
#     img = np.zeros((h, w), dtype=np.int32)

#     # Column-parallel loop
#     for c in prange(w):
#         cx = -2.0 + (c / w) * 2.5

#         for r in range(h):
#             cy = -1.2 + (r / h) * 2.4

#             z_real, z_imag = 0.0, 0.0
#             it = 0

#             while (
#                 z_real * z_real + z_imag * z_imag <= 4.0
#                 and it < max_iter
#             ):
#                 next_real = z_real * z_real - z_imag * z_imag + cx
#                 z_imag = 2.0 * z_real * z_imag + cy
#                 z_real = next_real
#                 it += 1

#             img[r, c] = it

#     return img


# # Warmup JIT
# _ = render_mandelbrot_rows(100, 100, 50)
# _ = render_mandelbrot_cols(100, 100, 50)

# H, W, MAX_IT = 2500, 2500, 1000


# # Row-parallel benchmark
# t0 = time.perf_counter()
# grid_rows = render_mandelbrot_rows(H, W, MAX_IT)
# t_rows = time.perf_counter() - t0


# # Column-parallel benchmark
# t1 = time.perf_counter()
# grid_cols = render_mandelbrot_cols(H, W, MAX_IT)
# t_cols = time.perf_counter() - t1


# print(f"Row-Parallel Render Time: {t_rows:.3f} s")
# print(f"Column-Parallel Render Time: {t_cols:.3f} s")


# # Save visual artifact
# plt.figure(figsize=(8, 8))
# plt.imshow(
#     grid_rows,
#     cmap="magma",
#     extent=[-2.0, 0.5, -1.2, 1.2]
# )
# plt.title(f"Mandelbrot {H}x{W} (Render: {t_rows:.2f}s)")
# plt.axis("off")
# plt.savefig(
#     "mandelbrot_output.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# print("Saved image: mandelbrot_output.png")


# ch 3
import time
import numpy as np
from numba import njit, prange


@njit(parallel=True)
def heat_step(u, u_next, alpha=0.20):
    rows, cols = u.shape

    for i in prange(1, rows - 1):
        for j in range(1, cols - 1):
            u_next[i, j] = u[i, j] + alpha * (
                u[i+1, j]
                + u[i-1, j]
                + u[i, j+1]
                + u[i, j-1]
                - 4.0 * u[i, j]
            )


GRID_SIZE = 1500
STEPS = 300

# FIRST TEST: float64
u = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
u_next = np.zeros_like(u)

# Dirichlet Boundary conditions
u[0, :] = 100.0
u[:, 0] = 100.0
u_next[0, :] = 100.0
u_next[:, 0] = 100.0

# Warmup
heat_step(u, u_next)

# Benchmark Execution
start = time.perf_counter()

for step in range(STEPS):
    heat_step(u, u_next)

    # Pointer swap
    u, u_next = u_next, u

elapsed = time.perf_counter() - start

cells_per_sec = (
    GRID_SIZE * GRID_SIZE * STEPS
) / elapsed / 1e6

print("=== FLOAT32 RESULT ===")
print(f"Heat Diffusion Complete: {elapsed:.3f} s")
print(f"Throughput: {cells_per_sec:.2f} Megacells/sec")