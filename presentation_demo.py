"""
presentation_demo.py — Live, animated visualization for SIH judges.
NOW SIMULATING BOTH POLICIES PROPERLY (not theoretical formulas)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

from env import SpectrumEnv
from policies import FixedSweepPolicy, WhittlePolicy
from whittle import WhittleIndexer

# --- Configuration ---
T_STEPS = 150
PD, PFA = 0.90, 0.05
SEED = 42  # Fixed seed for reproducibility

# --- Load Real Data ---
TRANSITIONS_FILE = "env_ready_transitions.npz"
if not os.path.exists(TRANSITIONS_FILE):
    raise FileNotFoundError(f"Could not find {TRANSITIONS_FILE}")

print(f"Loading real transition probabilities from {TRANSITIONS_FILE}...")
real_data = np.load(TRANSITIONS_FILE)
p01_real = real_data["p01"]
p11_real = real_data["p11"]
N_BANDS = len(p01_real)

print(f"Loaded real data for {N_BANDS} bands.")

# --- Create TWO identical environments (same seed) ---
# This ensures both policies face the EXACT SAME ground truth
env_whittle = SpectrumEnv(n_bands=N_BANDS, pd=PD, pfa=PFA, seed=SEED, p01=p01_real, p11=p11_real)
env_fixed = SpectrumEnv(n_bands=N_BANDS, pd=PD, pfa=PFA, seed=SEED, p01=p01_real, p11=p11_real)

env_whittle.reset()
env_fixed.reset()

# --- Initialize BOTH policies ---
indexer = WhittleIndexer(p01_real, p11_real)
policy_whittle = WhittlePolicy(N_BANDS, PD, PFA, p01_real, p11_real, indexer=indexer)
policy_fixed = FixedSweepPolicy(N_BANDS)

# --- Data buffers ---
true_active_history = np.zeros((T_STEPS, N_BANDS))
whittle_intercepts = np.zeros(T_STEPS)
fixed_intercepts = np.zeros(T_STEPS)

# --- Animation Setup ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})
fig.suptitle("Live EW Spectrum Scheduling: Whittle Index vs. Fixed Sweep (Real TSRD Data)", fontsize=14, fontweight='bold')

# Top Plot: Spectrum Heatmap
im = ax1.imshow(np.zeros((1, N_BANDS)), aspect='auto', cmap='hot', vmin=0, vmax=1, extent=[0, N_BANDS, 0, 1])
ax1.set_ylabel("Time (Steps)")
ax1.set_title("Ground Truth Spectrum Activity (Bright = Active)")
ax1.set_yticks([])

# Bottom Plot: Cumulative Intercepts
line_fixed, = ax2.plot([], [], 'r-', linewidth=2, label='Fixed Sweep (Baseline)')
line_whittle, = ax2.plot([], [], 'b-', linewidth=2, label='Whittle Index (Ours)')
ax2.set_xlim(0, T_STEPS)
ax2.set_ylim(0, 80)
ax2.set_xlabel("Timestep")
ax2.set_ylabel("Cumulative True Intercepts")
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

def init():
    return [im, line_fixed, line_whittle]

def update(frame):
    # ==========================================
    # ACTUALLY RUN BOTH POLICIES (not formulas!)
    # ==========================================
    
    # 1. Whittle Policy
    band_w = policy_whittle.choose_band(frame)
    res_w = env_whittle.step(band_w)
    policy_whittle.observe(band_w, res_w["observed_hit"])
    
    is_intercept_w = res_w["true_active"][band_w] and res_w["observed_hit"]
    whittle_intercepts[frame] = (whittle_intercepts[frame-1] if frame > 0 else 0) + (1 if is_intercept_w else 0)
    
    # 2. Fixed Sweep Policy (REAL SIMULATION)
    band_f = policy_fixed.choose_band(frame)
    res_f = env_fixed.step(band_f)
    policy_fixed.observe(band_f, res_f["observed_hit"])
    
    is_intercept_f = res_f["true_active"][band_f] and res_f["observed_hit"]
    fixed_intercepts[frame] = (fixed_intercepts[frame-1] if frame > 0 else 0) + (1 if is_intercept_f else 0)
    
    # Update Heatmap (show Whittle's perspective)
    current_row = res_w["true_active"].astype(float).reshape(1, -1)
    current_row[0, band_w] = 0.8  # Highlight sensed band
    
    global true_active_history
    true_active_history = np.vstack([true_active_history[1:], current_row])
    
    im.set_data(true_active_history)
    im.set_extent([0, N_BANDS, 0, frame+1])
    ax1.set_ylim(0, frame+1)
    
    # Update Line Chart with ACTUAL simulation results
    x_data = np.arange(frame + 1)
    line_whittle.set_data(x_data, whittle_intercepts[:frame+1])
    line_fixed.set_data(x_data, fixed_intercepts[:frame+1])
    
    ax2.set_title(f"Step {frame+1}/{T_STEPS} | Whittle: {int(whittle_intercepts[frame])} vs Fixed Sweep: {int(fixed_intercepts[frame])}")
    
    return [im, line_fixed, line_whittle]

ani = FuncAnimation(fig, update, frames=T_STEPS, init_func=init, blit=False, interval=100, repeat=False)
plt.tight_layout()
plt.show()