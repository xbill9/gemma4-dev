#!/usr/bin/env python3
"""RETRACTED DATA — do not regenerate the published chart from this file.

The numbers hard-coded below came from ``jax_e_benchmark_sweep.py``, which had
three methodology defects (see that file's module docstring and
``jax_e_benchmark_sweep_v2.py``):

  1. prefill was timed on an un-jitted call, so it measured dispatch overhead
     (tell: 512x more context moved "prefill" only 1.43x, 544 ms -> 779 ms);
  2. "decode step latency" was total_scan_time/16 with the prefill inside it;
  3. the decode steps ran with NO KV cache, so they never attended to history.

The B=1 column is additionally self-inconsistent: reconstructed wall time is
304.6 ms for one sequence versus 109.1 ms for two, i.e. the smaller batch is
2.8x slower in absolute terms. MXU underutilization cannot produce that, so the
"2.7x per-user speedup" derived from it does not hold.

Re-run ``jax_e_benchmark_sweep_v2.py`` on a TPU VM and replot from its
``--json-out`` before publishing any chart.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

# Set dark theme style
plt.style.use('dark_background')
fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
fig.suptitle('Gemma 4 E2B QAT JAX — Cloud TPU v6e-1 Benchmark Performance', fontsize=18, fontweight='bold', y=0.96, color='#4A90E2')

# Raw Benchmark Data
# (users, context, prefill_ms, decode_step_ms, agg_tok_s, per_user_tok_s)
data = [
    # B = 1
    (1, 8, 544.2, 19.04, 52.5, 52.5),
    (1, 16, 703.6, 19.04, 52.5, 52.5),
    (1, 32, 568.3, 19.04, 52.5, 52.5),
    (1, 64, 519.6, 19.04, 52.5, 52.5),
    (1, 128, 582.6, 19.16, 52.2, 52.2),
    (1, 256, 629.8, 19.15, 52.2, 52.2),
    (1, 512, 642.0, 19.38, 51.6, 51.6),
    (1, 1024, 643.4, 19.80, 50.5, 50.5),
    (1, 2048, 645.5, 20.90, 47.8, 47.8),
    (1, 4096, 779.2, 24.16, 41.4, 41.4),
    # B = 2
    (2, 8, 593.6, 6.82, 293.4, 146.7),
    (2, 16, 614.6, 6.81, 293.6, 146.8),
    (2, 32, 627.6, 6.81, 293.7, 146.8),
    (2, 64, 653.4, 6.81, 293.7, 146.9),
    (2, 128, 647.7, 6.93, 288.7, 144.4),
    (2, 256, 635.9, 6.90, 290.0, 145.0),
    (2, 512, 647.4, 7.47, 267.7, 133.8),
    (2, 1024, 626.6, 8.19, 244.2, 122.1),
    (2, 2048, 727.2, 11.54, 173.3, 86.6),
    # B = 4
    (4, 8, 655.6, 6.92, 577.8, 144.5),
    (4, 16, 654.4, 6.92, 577.7, 144.4),
    (4, 32, 658.6, 6.92, 577.8, 144.4),
    (4, 64, 658.2, 6.92, 577.7, 144.4),
    (4, 128, 630.5, 6.99, 572.2, 143.0),
    (4, 256, 656.6, 7.15, 559.6, 139.9),
    (4, 512, 659.0, 8.10, 494.1, 123.5),
    (4, 1024, 696.9, 10.66, 375.2, 93.8),
    # B = 8
    (8, 8, 651.2, 6.96, 1150.2, 143.8),
    (8, 16, 655.0, 6.95, 1150.5, 143.8),
    (8, 32, 648.3, 6.95, 1150.6, 143.8),
    (8, 64, 653.9, 6.95, 1150.4, 143.8),
    (8, 128, 656.4, 7.30, 1096.5, 137.1),
    (8, 256, 643.5, 7.96, 1004.8, 125.6),
    (8, 512, 645.2, 10.44, 766.1, 95.8),
    # B = 16
    (16, 8, 600.3, 7.20, 2223.3, 139.0),
    (16, 16, 597.1, 7.19, 2225.1, 139.1),
    (16, 32, 594.6, 7.19, 2224.6, 139.0),
    (16, 64, 595.2, 7.19, 2225.1, 139.1),
    (16, 128, 602.5, 8.32, 1924.0, 120.2),
    (16, 256, 643.6, 10.17, 1572.5, 98.3),
    # B = 32
    (32, 8, 608.4, 8.07, 3965.8, 123.9),
    (32, 16, 604.5, 8.07, 3965.0, 123.9),
    (32, 32, 596.3, 8.07, 3965.4, 123.9),
    (32, 64, 605.2, 8.07, 3966.2, 123.9),
    (32, 128, 621.9, 10.35, 3090.6, 96.6),
    # B = 64
    (64, 8, 649.2, 9.85, 6496.4, 101.5),
    (64, 16, 712.7, 9.85, 6496.8, 101.5),
    (64, 32, 707.6, 9.85, 6494.3, 101.5),
    (64, 64, 701.5, 9.86, 6494.0, 101.5),
]

users_list = [1, 2, 4, 8, 16, 32, 64]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5', '#9B59B6']

# ------------------------------------------------------------------------------
# Subplot 1: Aggregate Throughput (tok/s) vs Concurrent Users (B)
# ------------------------------------------------------------------------------
ax1 = axs[0, 0]
for idx, b in enumerate(users_list):
    b_data = [d for d in data if d[0] == b]
    ctxs = [d[1] for d in b_data]
    aggs = [d[4] for d in b_data]
    ax1.plot(ctxs, aggs, marker='o', linewidth=2, label=f'Users = {b}', color=colors[idx])

ax1.set_xscale('log', base=2)
ax1.set_xlabel('Context Length (tokens)', fontsize=12, labelpad=8)
ax1.set_ylabel('Aggregate Serving Throughput (tok/s)', fontsize=12, labelpad=8)
ax1.set_title('Aggregate Serving Throughput vs Context Length', fontsize=14, pad=12, color='#4ECDC4')
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend(loc='upper right', framealpha=0.6, fontsize=9)
ax1.annotate('Peak: 6,496.8 tok/s\n(64 users @ 64 ctx)', xy=(64, 6494.0), xytext=(32, 5000),
             arrowprops=dict(facecolor='#FF6B6B', shrink=0.05, width=1.5, headwidth=8),
             fontsize=10, fontweight='bold', color='#FF6B6B', bbox=dict(boxstyle="round,pad=0.3", fc="#1A1A1A", ec="#FF6B6B", lw=1))

# ------------------------------------------------------------------------------
# Subplot 2: Per-User Throughput (tok/s/user) vs Concurrent Users (B)
# ------------------------------------------------------------------------------
ax2 = axs[0, 1]
b_short_ctx = []
per_user_tok_s_short = []
for b in users_list:
    # average for short contexts (8..64)
    b_data = [d[5] for d in data if d[0] == b and d[1] <= 64]
    if b_data:
        b_short_ctx.append(b)
        per_user_tok_s_short.append(np.mean(b_data))

bars = ax2.bar([str(b) for b in b_short_ctx], per_user_tok_s_short, color='#45B7D1', edgecolor='#FFFFFF', alpha=0.85, width=0.55)
ax2.set_xlabel('Concurrent Users (Batch Size B)', fontsize=12, labelpad=8)
ax2.set_ylabel('Per-User Token Speed (tok/s/user)', fontsize=12, labelpad=8)
ax2.set_title('Per-User Throughput Boost via TPU Vectorization', fontsize=14, pad=12, color='#45B7D1')
ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 3, f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#FFFFFF')

ax2.annotate('Single User: 52.5 tok/s\n(MXU Underutilized)', xy=(0, 52.5), xytext=(0.5, 90),
             arrowprops=dict(facecolor='#FF6B6B', shrink=0.05, width=1.5, headwidth=8),
             fontsize=9, color='#FF6B6B', bbox=dict(boxstyle="round,pad=0.3", fc="#1A1A1A", ec="#FF6B6B", lw=1))

ax2.annotate('~2.7x Speedup per User!\n(144-147 tok/s/user @ B=2..8)', xy=(2, 146.8), xytext=(2.2, 120),
             arrowprops=dict(facecolor='#2ECC71', shrink=0.05, width=1.5, headwidth=8),
             fontsize=9, fontweight='bold', color='#2ECC71', bbox=dict(boxstyle="round,pad=0.3", fc="#1A1A1A", ec="#2ECC71", lw=1))

# ------------------------------------------------------------------------------
# Subplot 3: Decode Step Latency (ms) vs Context Length
# ------------------------------------------------------------------------------
ax3 = axs[1, 0]
for idx, b in enumerate([1, 2, 4, 8, 16, 32, 64]):
    b_data = [d for d in data if d[0] == b]
    ctxs = [d[1] for d in b_data]
    latencies = [d[3] for d in b_data]
    ax3.plot(ctxs, latencies, marker='s', linewidth=2, label=f'B = {b}', color=colors[idx])

ax3.set_xscale('log', base=2)
ax3.set_xlabel('Context Length (tokens)', fontsize=12, labelpad=8)
ax3.set_ylabel('Decode Step Latency (ms/token)', fontsize=12, labelpad=8)
ax3.set_title('Decode Step Latency vs Context Length', fontsize=14, pad=12, color='#FFEEAD')
ax3.grid(True, linestyle='--', alpha=0.3)
ax3.legend(loc='upper left', framealpha=0.6, fontsize=9)

# ------------------------------------------------------------------------------
# Subplot 4: TPU HBM Memory Frontier (Max Reachable Context vs Users)
# ------------------------------------------------------------------------------
ax4 = axs[1, 1]
max_context_map = {
    1: 4096,
    2: 2048,
    4: 1024,
    8: 512,
    16: 256,
    32: 128,
    64: 64,
    128: 0
}

u_keys = [1, 2, 4, 8, 16, 32, 64]
max_ctx_vals = [max_context_map[k] for k in u_keys]

ax4.plot([str(k) for k in u_keys], max_ctx_vals, marker='D', markersize=8, color='#E74C3C', linewidth=3, linestyle='-')
ax4.fill_between([str(k) for k in u_keys], max_ctx_vals, color='#E74C3C', alpha=0.25)
ax4.set_xlabel('Concurrent Users (Batch Size B)', fontsize=12, labelpad=8)
ax4.set_ylabel('Max Reachable Context (tokens)', fontsize=12, labelpad=8)
ax4.set_title('TPU v6e 32GB HBM KV Cache Capacity Frontier', fontsize=14, pad=12, color='#E74C3C')
ax4.grid(True, linestyle='--', alpha=0.3)

for x_i, y_i in zip([str(k) for k in u_keys], max_ctx_vals):
    lbl = f'{y_i} tok' if y_i >= 1024 else f'{y_i}'
    ax4.text(x_i, y_i * 1.15, lbl, ha='center', va='bottom', fontsize=10, fontweight='bold', color='#E74C3C')

ax4.set_yscale('log', base=2)
ax4.set_ylim(32, 8192)

plt.tight_layout(rect=[0, 0, 1, 0.94])
# Writers use a bare filename in the CWD so an archived run dir is never overwritten;
# ARTIFACT_DIR redirects it. This was an absolute path to one machine's checkout.
out_path = os.path.join(os.getenv('ARTIFACT_DIR', '.'), 'gemma4_tpu_v6e_benchmark.png')
plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none')
print(f"Successfully generated plot at {out_path}")
