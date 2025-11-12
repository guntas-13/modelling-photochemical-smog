"""
Animated GIF Generator for Photochemical Smog Modeling
Creates time-progressing animations for all model results
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import shutil
import sys
import os

# Add parent directory to path to import latex.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from latex import latexify

import warnings
warnings.filterwarnings('ignore')

# Set publication style
latexify(columns=2)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Save high-quality MP4s when possible
SAVE_MP4 = True
ffmpeg_path = shutil.which('ffmpeg')
ffmpeg_available = ffmpeg_path is not None
if SAVE_MP4 and not ffmpeg_available:
    print('\nWARNING: ffmpeg not found in PATH. MP4 exports will be skipped.')
    print('Install ffmpeg (e.g. `brew install ffmpeg`) to enable high-quality MP4 output.\n')

# Colors
COLORS = {
    'NO': '#1f77b4', 'NO2': '#d62728', 'O3': '#2ca02c',
    'OH': '#ff7f0e', 'HO2': '#17becf', 'RO2': '#e377c2',
    'CO': '#8c564b', 'HCHO': '#9467bd', 'ALK': '#2ca02c', 'OLE': '#bcbd22'
}

# ============================================================================
# COMMON FUNCTIONS
# ============================================================================

def solar_intensity(t):
    """Solar intensity factor (0-1) based on time of day"""
    t_solar = t + 5
    if 6 <= t_solar <= 18:
        return np.sin(np.pi * (t_solar - 6) / 12)
    return 0.0

def add_daylight_shading(ax, t_max):
    """Add yellow shading for daylight hours"""
    ax.axvspan(6, min(18, t_max), alpha=0.15, color='yellow', zorder=0)

# ============================================================================
# MODEL 1: BASIC PHOTOCHEMICAL CYCLE
# ============================================================================

T = 288.0
O2 = 210000.0

k1_max = 0.508 * 60
k2 = 3.9e-6 * np.exp(510/T) * 60
k3 = 3.1e3 * np.exp(-1450/T) * 60
k4 = 1.34e4 * 60
k5 = 5.6e2 * np.exp(584/T) * 60

E_NO_m1 = 0.02
E_NO2_m1 = 0.01

def k1_photolysis(t):
    return k1_max * solar_intensity(t)

def model1_odes(C, t):
    NO, NO2, O3, O = C
    k1 = k1_photolysis(t)
    R1 = k1 * NO2
    R2 = k2 * O * O2
    R3 = k3 * O3 * NO
    R4 = k4 * NO2 * O
    R5 = k5 * NO * O
    dNO_dt = R1 - R3 + R4 - R5 + E_NO_m1
    dNO2_dt = -R1 + R3 - R4 + R5 + E_NO2_m1
    dO3_dt = R2 - R3
    dO_dt = R1 - R2 - R4 - R5
    return [dNO_dt, dNO2_dt, dO3_dt, dO_dt]

def model1_odes_1(y, t):
    NO, NO2, O3 = y
    k1 = k1_photolysis(t)
    dNO_dt = k1 * NO2 - k3 * NO * O3 + E_NO_m1
    dNO2_dt = -k1 * NO2 + k3 * NO * O3 + E_NO2_m1
    dO3_dt = k1 * NO2 - k3 * NO * O3
    return [dNO_dt, dNO2_dt, dO3_dt]

# ============================================================================
# MODEL 2: REFINED WITH VOCs
# ============================================================================

H2O = 15000.0
M = 1.0e6

k1_max_m2 = k1_max
k4_max = 0.0328 * 60
k7_max = 0.00284 * 60

k2_m2 = k2
k3_m2 = k3
k5_m2 = 1.0e5 * 60
k6 = 4.4e2 * 60
k8 = 19200.0 * 60
k9 = 4700.0 * 60
k10 = 89142.0 * 60
k11 = 0.136 * 60
k12 = 1.2e4 * 60
k13 = 1.2e4 * 60
k14 = 3700.0 * 60
k15 = 1.477e15 * 10**(-11.6*T/(17.4+T)) * (280/T)**2 * 60

E_NO_m2 = 0.02
E_NO2_m2 = 0.01
E_CO = 0.02
E_HCHO = 0.03
E_ALK = 0.1
E_OLE = 0.0

def k4_photolysis(t):
    return k4_max * solar_intensity(t)

def k7_photolysis(t):
    return k7_max * solar_intensity(t)

def model2_odes(C, t):
    NO, NO2, O3, O, CO, HCHO, ALK, OLE, OH, HO2, RO2 = C
    k1 = k1_max_m2 * solar_intensity(t)
    k4 = k4_photolysis(t)
    k7 = k7_photolysis(t)
    if k5_m2 * H2O > 0:
        O1D = k4 * O3 / (k5_m2 * H2O)
    else:
        O1D = 0.0
    R1 = k1 * NO2
    R2 = k2_m2 * O * O2
    R3 = k3_m2 * O3 * NO
    R4 = k4 * O3
    R5 = k5_m2 * O1D * H2O
    R6 = k6 * CO * OH
    R7 = k7 * HCHO
    R8 = k8 * HCHO * OH
    R9 = k9 * ALK * OH
    R10 = k10 * OLE * OH
    R11 = k11 * OLE * O3
    R12 = k12 * HO2 * NO
    R13 = k13 * RO2 * NO
    R14 = k14 * HO2 * HO2
    R15 = k15 * OH * NO2
    dNO_dt = R1 - R3 - R12 - R13 + E_NO_m2
    dNO2_dt = -R1 + R3 + R12 + R13 - R15 + E_NO2_m2
    dO3_dt = R2 - R3 - R4 - R11
    dO_dt = R1 - R2
    dCO_dt = -R6 + R7 + R8 + E_CO
    dHCHO_dt = 0.5 * R11 - R7 - R8 + E_HCHO
    dALK_dt = -R9 + E_ALK
    dOLE_dt = -R10 - R11 + E_OLE
    dOH_dt = 2*R5 - R6 - R8 - R9 - R10 - R15 + R12
    dHO2_dt = R6 + 2*R7 + R8 + 0.5*R11 + R13 - R12 - 2*R14
    dRO2_dt = R9 + R10 + 0.5*R11 - R13
    return [dNO_dt, dNO2_dt, dO3_dt, dO_dt, dCO_dt, dHCHO_dt,
            dALK_dt, dOLE_dt, dOH_dt, dHO2_dt, dRO2_dt]

# ============================================================================
# SOLVE MODELS
# ============================================================================

print("="*70)
print("ANIMATED GIF GENERATOR - ALL MODELS")
print("="*70)

C0_m1 = [0.1, 0.05, 0.0, 0.0]
t_span = np.linspace(0, 19, 2000)

sol_m1 = odeint(model1_odes, C0_m1, t_span)
sol_m1_1 = odeint(model1_odes_1, C0_m1[:3], t_span)

NO_m1 = sol_m1[:, 0]
NO2_m1 = sol_m1[:, 1]
O3_m1 = sol_m1[:, 2]
O_m1 = sol_m1[:, 3]

NO_m1_1 = sol_m1_1[:, 0]
NO2_m1_1 = sol_m1_1[:, 1]
O3_m1_1 = sol_m1_1[:, 2]

t_clock = t_span + 5

C0_m2 = [0.1, 0.05, 0.0, 0.0, 0.1, 0.01, 1.0, 0.2, 0.0, 0.0, 0.0]
sol_m2 = odeint(model2_odes, C0_m2, t_span, rtol=1e-6, atol=1e-8)

NO_m2 = sol_m2[:, 0]
NO2_m2 = sol_m2[:, 1]
O3_m2 = sol_m2[:, 2]
O_m2 = sol_m2[:, 3]
CO_m2 = sol_m2[:, 4]
HCHO_m2 = sol_m2[:, 5]
ALK_m2 = sol_m2[:, 6]
OLE_m2 = sol_m2[:, 7]
OH_m2 = sol_m2[:, 8]
HO2_m2 = sol_m2[:, 9]
RO2_m2 = sol_m2[:, 10]

# ============================================================================
# ANIMATION 1: model1_results.gif
# ============================================================================

print("\n[1/4] Creating animation: model1_results.gif")

fig1 = plt.figure(figsize=(14, 10))
gs1 = GridSpec(2, 2, figure=fig1, hspace=0.3, wspace=0.3)

ax1_1 = fig1.add_subplot(gs1[0, 0])
ax1_2 = fig1.add_subplot(gs1[0, 1])
ax1_3 = fig1.add_subplot(gs1[1, 0])
ax1_4 = fig1.add_subplot(gs1[1, 1])

# Initialize plot elements
line1_NO, = ax1_1.plot([], [], color=COLORS['NO'], linewidth=2.5, label='NO')
line1_NO2, = ax1_1.plot([], [], color=COLORS['NO2'], linewidth=2.5, label=r'NO$_2$')
line1_O3, = ax1_2.plot([], [], color=COLORS['O3'], linewidth=3)
line1_O, = ax1_3.plot([], [], 'm-', linewidth=2.5)
line1_NO_norm, = ax1_4.plot([], [], color=COLORS['NO'], linewidth=2, label='NO')
line1_NO2_norm, = ax1_4.plot([], [], color=COLORS['NO2'], linewidth=2, label=r'NO$_2$')
line1_O3_norm, = ax1_4.plot([], [], color=COLORS['O3'], linewidth=2, label=r'O$_3$')
peak1_marker, = ax1_2.plot([], [], 'r*', markersize=15)

# Set up axes
for ax in [ax1_1, ax1_2, ax1_3, ax1_4]:
    ax.set_xlim(5, 24)
    ax.grid(alpha=0.3)

ax1_1.set_ylim(0, max(NO_m1.max(), NO2_m1.max()) * 1.1)
ax1_1.set_xlabel('Time (hours)', fontweight='bold')
ax1_1.set_ylabel('Concentration (ppm)', fontweight='bold')
ax1_1.set_title(r'(a) Primary Pollutants: NO and NO$_2$', fontweight='bold')
ax1_1.legend(framealpha=0.9)

ax1_2.set_ylim(0, O3_m1.max() * 1.1)
ax1_2.set_xlabel('Time (hours)', fontweight='bold')
ax1_2.set_ylabel('Concentration (ppm)', fontweight='bold')
ax1_2.set_title(r'(b) Secondary Pollutant: Ozone (O$_3$)', fontweight='bold')

ax1_3.set_ylim(0, O_m1.max() * 1.1)
ax1_3.set_xlabel('Time (hours)', fontweight='bold')
ax1_3.set_ylabel(r'Concentration (ppm)', fontweight='bold')
ax1_3.set_title('(c) Reactive Intermediate: Atomic Oxygen (O)', fontweight='bold')

ax1_4.set_ylim(0, 1.1)
ax1_4.set_xlabel('Time (hours)', fontweight='bold')
ax1_4.set_ylabel('Normalized Concentration', fontweight='bold')
ax1_4.set_title('(d) Comparative Dynamics (Normalized)', fontweight='bold')
ax1_4.legend(framealpha=0.9)

fig1.suptitle('Model 1: Basic Photochemical Cycle', fontsize=14, fontweight='bold')

def animate1(frame):
    idx = int((frame / num_frames) * len(t_clock))
    if idx == 0:
        idx = 1
    
    t_current = t_clock[:idx]
    
    line1_NO.set_data(t_current, NO_m1[:idx])
    line1_NO2.set_data(t_current, NO2_m1[:idx])
    line1_O3.set_data(t_current, O3_m1[:idx])
    line1_O.set_data(t_current, O_m1[:idx])
    line1_NO_norm.set_data(t_current, NO_m1[:idx]/NO_m1.max())
    line1_NO2_norm.set_data(t_current, NO2_m1[:idx]/NO2_m1.max())
    line1_O3_norm.set_data(t_current, O3_m1[:idx]/O3_m1.max())
    
    # Update fill
    for coll in ax1_2.collections:
        coll.remove()
    ax1_2.fill_between(t_current, 0, O3_m1[:idx], color=COLORS['O3'], alpha=0.2)
    
    peak_idx = O3_m1.argmax()
    if idx >= peak_idx:
        peak1_marker.set_data([t_clock[peak_idx]], [O3_m1[peak_idx]])
    else:
        peak1_marker.set_data([], [])
    
    t_max = t_current[-1]
    for ax in [ax1_1, ax1_2, ax1_3, ax1_4]:
        for patch in [p for p in ax.patches if len(p.get_facecolor()) > 0 and p.get_facecolor()[0] > 0.9]:
            patch.remove()
        add_daylight_shading(ax, t_max)
    
    return line1_NO, line1_NO2, line1_O3, line1_O, line1_NO_norm, line1_NO2_norm, line1_O3_norm, peak1_marker

num_frames = 150
fps = 20

anim1 = FuncAnimation(fig1, animate1, frames=num_frames, interval=1000/fps, blit=True, repeat=True)
writer1 = PillowWriter(fps=fps)
anim1.save('model1_results.gif', writer=writer1, dpi=300)
print(f"✓ Saved: model1_results.gif ({os.path.getsize('model1_results.gif') / (1024*1024):.2f} MB)")
plt.close(fig1)
if SAVE_MP4 and ffmpeg_available:
    try:
        mp4_writer = FFMpegWriter(fps=fps, metadata={'artist': 'animate_all_models'}, bitrate=8000)
        anim1.save('model1_results.mp4', writer=mp4_writer, dpi=300)
        print(f"✓ Saved: model1_results.mp4 ({os.path.getsize('model1_results.mp4') / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"Failed to save model1_results.mp4: {e}")

# ============================================================================
# ANIMATION 2: model1_results_1.gif
# ============================================================================

print("\n[2/4] Creating animation: model1_results_1.gif")

fig2 = plt.figure(figsize=(14, 10))
gs2 = GridSpec(2, 2, figure=fig2, hspace=0.3, wspace=0.3)

ax2_1 = fig2.add_subplot(gs2[0, 0])
ax2_2 = fig2.add_subplot(gs2[0, 1])
ax2_3 = fig2.add_subplot(gs2[1, :])

line2_NO, = ax2_1.plot([], [], color=COLORS['NO'], linewidth=2.5, label='NO')
line2_NO2, = ax2_1.plot([], [], color=COLORS['NO2'], linewidth=2.5, label=r'NO$_2$')
line2_O3, = ax2_2.plot([], [], color=COLORS['O3'], linewidth=3)
peak2_marker, = ax2_2.plot([], [], 'r*', markersize=15)
line2_NO_c, = ax2_3.plot([], [], color=COLORS['NO'], linewidth=2, label='NO')
line2_NO2_c, = ax2_3.plot([], [], color=COLORS['NO2'], linewidth=2, label=r'NO$_2$')
line2_O3_c, = ax2_3.plot([], [], color=COLORS['O3'], linewidth=2, label=r'O$_3$')

for ax in [ax2_1, ax2_2, ax2_3]:
    ax.set_xlim(5, 24)
    ax.grid(alpha=0.3)

ax2_1.set_ylim(0, max(NO_m1_1.max(), NO2_m1_1.max()) * 1.1)
ax2_1.set_xlabel('Time (hours)', fontweight='bold')
ax2_1.set_ylabel('Concentration (ppm)', fontweight='bold')
ax2_1.set_title(r'(a) Primary Pollutants: NO and NO$_2$', fontweight='bold')
ax2_1.legend(framealpha=0.9)

ax2_2.set_ylim(0, O3_m1_1.max() * 1.1)
ax2_2.set_xlabel('Time (hours)', fontweight='bold')
ax2_2.set_ylabel('Concentration (ppm)', fontweight='bold')
ax2_2.set_title(r'(b) Secondary Pollutant: Ozone (O$_3$)', fontweight='bold')

ax2_3.set_ylim(0, max(NO_m1_1.max(), NO2_m1_1.max(), O3_m1_1.max()) * 1.1)
ax2_3.set_xlabel('Time (hours)', fontweight='bold')
ax2_3.set_ylabel('Concentration', fontweight='bold')
ax2_3.set_title('(c) Comparative Dynamics', fontweight='bold')
ax2_3.legend(framealpha=0.9)

fig2.suptitle('Model 1: Basic Photochemical Cycle', fontsize=14, fontweight='bold')

def animate2(frame):
    idx = int((frame / num_frames) * len(t_clock))
    if idx == 0:
        idx = 1
    
    t_current = t_clock[:idx]
    
    line2_NO.set_data(t_current, NO_m1_1[:idx])
    line2_NO2.set_data(t_current, NO2_m1_1[:idx])
    line2_O3.set_data(t_current, O3_m1_1[:idx])
    line2_NO_c.set_data(t_current, NO_m1_1[:idx])
    line2_NO2_c.set_data(t_current, NO2_m1_1[:idx])
    line2_O3_c.set_data(t_current, O3_m1_1[:idx])
    
    for coll in ax2_2.collections:
        coll.remove()
    ax2_2.fill_between(t_current, 0, O3_m1_1[:idx], color=COLORS['O3'], alpha=0.2)
    
    peak_idx = O3_m1_1.argmax()
    if idx >= peak_idx:
        peak2_marker.set_data([t_clock[peak_idx]], [O3_m1_1[peak_idx]])
    else:
        peak2_marker.set_data([], [])
    
    t_max = t_current[-1]
    for ax in [ax2_1, ax2_2, ax2_3]:
        for patch in [p for p in ax.patches if len(p.get_facecolor()) > 0 and p.get_facecolor()[0] > 0.9]:
            patch.remove()
        add_daylight_shading(ax, t_max)
    
    return line2_NO, line2_NO2, line2_O3, line2_NO_c, line2_NO2_c, line2_O3_c, peak2_marker

anim2 = FuncAnimation(fig2, animate2, frames=num_frames, interval=1000/fps, blit=True, repeat=True)
writer2 = PillowWriter(fps=fps)
anim2.save('model1_results_1.gif', writer=writer2, dpi=300)
print(f"✓ Saved: model1_results_1.gif ({os.path.getsize('model1_results_1.gif') / (1024*1024):.2f} MB)")
plt.close(fig2)
if SAVE_MP4 and ffmpeg_available:
    try:
        mp4_writer = FFMpegWriter(fps=fps, metadata={'artist': 'animate_all_models'}, bitrate=8000)
        anim2.save('model1_results_1.mp4', writer=mp4_writer, dpi=300)
        print(f"✓ Saved: model1_results_1.mp4 ({os.path.getsize('model1_results_1.mp4') / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"Failed to save model1_results_1.mp4: {e}")

# ============================================================================
# ANIMATION 3: model2_results.gif
# ============================================================================

print("\n[3/4] Creating animation: model2_results.gif")

fig3 = plt.figure(figsize=(16, 12))
gs3 = GridSpec(3, 3, figure=fig3, hspace=0.35, wspace=0.35)

axes3 = [fig3.add_subplot(gs3[i, j]) for i in range(3) for j in range(3)]

# Initialize all lines
lines3 = {}
lines3['O3'], = axes3[0].plot([], [], color=COLORS['O3'], linewidth=3)
peak3_marker, = axes3[0].plot([], [], 'r*', markersize=12)
lines3['NO'], = axes3[1].plot([], [], color=COLORS['NO'], linewidth=2.5, label='NO')
lines3['NO2'], = axes3[1].plot([], [], color=COLORS['NO2'], linewidth=2.5, label=r'NO$_2$')
lines3['OH'], = axes3[2].plot([], [], color=COLORS['OH'], linewidth=2, label=r'OH ($\times 10^6$)')
lines3['HO2'], = axes3[2].plot([], [], color=COLORS['HO2'], linewidth=2, label=r'HO$_2$ ($\times 10^3$)')
lines3['RO2'], = axes3[2].plot([], [], color=COLORS['RO2'], linewidth=2, label=r'RO$_2$ ($\times 10^3$)')
lines3['ALK'], = axes3[3].plot([], [], color=COLORS['ALK'], linewidth=2.5, label='ALK')
lines3['OLE'], = axes3[3].plot([], [], color=COLORS['OLE'], linewidth=2.5, label='OLE')
lines3['HCHO'], = axes3[4].plot([], [], color=COLORS['HCHO'], linewidth=2.5, label='HCHO')
lines3['CO'], = axes3[4].plot([], [], color=COLORS['CO'], linewidth=2.5, label='CO')
lines3['O3_path'], = axes3[5].plot([], [], 'g--', linewidth=2, label=r'O$_3$ + NO', alpha=0.7)
lines3['HO2_path'], = axes3[5].plot([], [], color=COLORS['HO2'], linewidth=2, label=r'HO$_2$ + NO')
lines3['RO2_path'], = axes3[5].plot([], [], color=COLORS['RO2'], linewidth=2, label=r'RO$_2$ + NO')
lines3['NOx'], = axes3[6].plot([], [], 'darkred', linewidth=3, label=r'NO$_x$ (NO + NO$_2$)')
lines3['NO_dash'], = axes3[6].plot([], [], color=COLORS['NO'], linewidth=1.5, linestyle='--', alpha=0.6)
lines3['NO2_dash'], = axes3[6].plot([], [], color=COLORS['NO2'], linewidth=1.5, linestyle='--', alpha=0.6)
lines3['Ox'], = axes3[7].plot([], [], 'darkblue', linewidth=3)
lines3['ratio'], = axes3[8].plot([], [], 'purple', linewidth=2.5)

for ax in axes3:
    ax.set_xlim(5, 24)
    ax.grid(alpha=0.3)

axes3[0].set_ylim(0, O3_m2.max() * 1.1)
axes3[0].set_xlabel('Time (hours)', fontweight='bold')
axes3[0].set_ylabel('Concentration (ppm)', fontweight='bold')
axes3[0].set_title('(a) Ozone Formation', fontweight='bold')

axes3[1].set_ylim(0, max(NO_m2.max(), NO2_m2.max()) * 1.1)
axes3[1].set_xlabel('Time (hours)', fontweight='bold')
axes3[1].set_ylabel('Concentration (ppm)', fontweight='bold')
axes3[1].set_title('(b) Nitrogen Oxides', fontweight='bold')
axes3[1].legend()

axes3[2].set_ylim(0, max(OH_m2.max()*1e6, HO2_m2.max()*1e3, RO2_m2.max()*1e3) * 1.1)
axes3[2].set_xlabel('Time (hours)', fontweight='bold')
axes3[2].set_ylabel('Scaled Concentration', fontweight='bold')
axes3[2].set_title('(c) Radical Species', fontweight='bold')
axes3[2].legend()

axes3[3].set_ylim(0, max(ALK_m2.max(), OLE_m2.max()) * 1.1)
axes3[3].set_xlabel('Time (hours)', fontweight='bold')
axes3[3].set_ylabel('Concentration (ppm)', fontweight='bold')
axes3[3].set_title('(d) Volatile Organic Compounds', fontweight='bold')
axes3[3].legend()

axes3[4].set_ylim(0, max(HCHO_m2.max(), CO_m2.max()) * 1.1)
axes3[4].set_xlabel('Time (hours)', fontweight='bold')
axes3[4].set_ylabel('Concentration (ppm)', fontweight='bold')
axes3[4].set_title('(e) Secondary Products', fontweight='bold')
axes3[4].legend()

rate_O3 = k3_m2 * O3_m2 * NO_m2
rate_HO2 = k12 * HO2_m2 * NO_m2
rate_RO2 = k13 * RO2_m2 * NO_m2
axes3[5].set_ylim(0, max(rate_O3.max(), rate_HO2.max(), rate_RO2.max()) * 1.1)
axes3[5].set_xlabel('Time (hours)', fontweight='bold')
axes3[5].set_ylabel('Rate (ppm/h)', fontweight='bold')
axes3[5].set_title(r'(f) NO $\to$ NO$_2$ Pathways', fontweight='bold')
axes3[5].legend()

NOx_m2 = NO_m2 + NO2_m2
axes3[6].set_ylim(0, NOx_m2.max() * 1.1)
axes3[6].set_xlabel('Time (hours)', fontweight='bold')
axes3[6].set_ylabel('Concentration (ppm)', fontweight='bold')
axes3[6].set_title(r'(g) NO$_x$ Budget', fontweight='bold')
axes3[6].legend()

Ox = O3_m2 + NO2_m2
axes3[7].set_ylim(0, Ox.max() * 1.1)
axes3[7].set_xlabel('Time (hours)', fontweight='bold')
axes3[7].set_ylabel('Concentration (ppm)', fontweight='bold')
axes3[7].set_title(r'(h) Oxidant Capacity (O$_x$)', fontweight='bold')

axes3[8].set_ylim(0, 5)
axes3[8].set_xlabel('Time (hours)', fontweight='bold')
axes3[8].set_ylabel('Ratio', fontweight='bold')
axes3[8].set_title(r'(i) NO/NO$_2$ Ratio', fontweight='bold')

fig3.suptitle('Model 2: Refined with VOCs and Radicals', fontsize=14, fontweight='bold')

def animate3(frame):
    idx = int((frame / num_frames) * len(t_clock))
    if idx == 0:
        idx = 1
    
    t_current = t_clock[:idx]
    
    lines3['O3'].set_data(t_current, O3_m2[:idx])
    lines3['NO'].set_data(t_current, NO_m2[:idx])
    lines3['NO2'].set_data(t_current, NO2_m2[:idx])
    lines3['OH'].set_data(t_current, OH_m2[:idx]*1e6)
    lines3['HO2'].set_data(t_current, HO2_m2[:idx]*1e3)
    lines3['RO2'].set_data(t_current, RO2_m2[:idx]*1e3)
    lines3['ALK'].set_data(t_current, ALK_m2[:idx])
    lines3['OLE'].set_data(t_current, OLE_m2[:idx])
    lines3['HCHO'].set_data(t_current, HCHO_m2[:idx])
    lines3['CO'].set_data(t_current, CO_m2[:idx])
    lines3['O3_path'].set_data(t_current, rate_O3[:idx])
    lines3['HO2_path'].set_data(t_current, rate_HO2[:idx])
    lines3['RO2_path'].set_data(t_current, rate_RO2[:idx])
    lines3['NOx'].set_data(t_current, NOx_m2[:idx])
    lines3['NO_dash'].set_data(t_current, NO_m2[:idx])
    lines3['NO2_dash'].set_data(t_current, NO2_m2[:idx])
    lines3['Ox'].set_data(t_current, Ox[:idx])
    NO2_safe = np.where(NO2_m2[:idx] > 1e-6, NO2_m2[:idx], 1e-6)
    lines3['ratio'].set_data(t_current, NO_m2[:idx] / NO2_safe)
    
    for coll in axes3[0].collections:
        coll.remove()
    axes3[0].fill_between(t_current, 0, O3_m2[:idx], color=COLORS['O3'], alpha=0.2)
    
    peak_idx = O3_m2.argmax()
    if idx >= peak_idx:
        peak3_marker.set_data([t_clock[peak_idx]], [O3_m2[peak_idx]])
    else:
        peak3_marker.set_data([], [])
    
    t_max = t_current[-1]
    for ax in axes3:
        for patch in [p for p in ax.patches if len(p.get_facecolor()) > 0 and p.get_facecolor()[0] > 0.9]:
            patch.remove()
        add_daylight_shading(ax, t_max)
    
    return tuple(lines3.values()) + (peak3_marker,)

anim3 = FuncAnimation(fig3, animate3, frames=num_frames, interval=1000/fps, blit=True, repeat=True)
writer3 = PillowWriter(fps=fps)
anim3.save('model2_results.gif', writer=writer3, dpi=300)
print(f"✓ Saved: model2_results.gif ({os.path.getsize('model2_results.gif') / (1024*1024):.2f} MB)")
plt.close(fig3)
if SAVE_MP4 and ffmpeg_available:
    try:
        mp4_writer = FFMpegWriter(fps=fps, metadata={'artist': 'animate_all_models'}, bitrate=12000)
        anim3.save('model2_results.mp4', writer=mp4_writer, dpi=300)
        print(f"✓ Saved: model2_results.mp4 ({os.path.getsize('model2_results.mp4') / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"Failed to save model2_results.mp4: {e}")

# ============================================================================
# ANIMATION 4: comparison_models.gif
# ============================================================================

print("\n[4/4] Creating animation: comparison_models.gif")

fig4 = plt.figure(figsize=(14, 10))
gs4 = GridSpec(2, 2, figure=fig4, hspace=0.3, wspace=0.3)

ax4_1 = fig4.add_subplot(gs4[0, :])
ax4_2 = fig4.add_subplot(gs4[1, 0])
ax4_3 = fig4.add_subplot(gs4[1, 1])

line4_O3_m1, = ax4_1.plot([], [], 'g--', linewidth=3, label='Model 1 (No VOCs)', alpha=0.7)
line4_O3_m2, = ax4_1.plot([], [], 'g-', linewidth=3, label='Model 2 (With VOCs)')
line4_NO_m1, = ax4_2.plot([], [], 'b--', linewidth=2.5, label='Model 1', alpha=0.7)
line4_NO_m2, = ax4_2.plot([], [], 'b-', linewidth=2.5, label='Model 2')
line4_NO2_m1, = ax4_3.plot([], [], 'r--', linewidth=2.5, label='Model 1', alpha=0.7)
line4_NO2_m2, = ax4_3.plot([], [], 'r-', linewidth=2.5, label='Model 2')

for ax in [ax4_1, ax4_2, ax4_3]:
    ax.set_xlim(5, 24)
    ax.grid(alpha=0.3)

ax4_1.set_ylim(0, max(O3_m1.max(), O3_m2.max()) * 1.1)
ax4_1.set_xlabel('Time (hours)', fontweight='bold')
ax4_1.set_ylabel(r'O$_3$ Concentration (ppm)', fontweight='bold')
ax4_1.set_title('(a) Ozone: Model Comparison', fontweight='bold')
ax4_1.legend(framealpha=0.9)

ax4_2.set_ylim(0, max(NO_m1.max(), NO_m2.max()) * 1.1)
ax4_2.set_xlabel('Time (hours)', fontweight='bold')
ax4_2.set_ylabel('NO Concentration (ppm)', fontweight='bold')
ax4_2.set_title('(b) NO: Model Comparison', fontweight='bold')
ax4_2.legend(framealpha=0.9)

ax4_3.set_ylim(0, max(NO2_m1.max(), NO2_m2.max()) * 1.1)
ax4_3.set_xlabel('Time (hours)', fontweight='bold')
ax4_3.set_ylabel(r'NO$_2$ Concentration (ppm)', fontweight='bold')
ax4_3.set_title(r'(c) NO$_2$: Model Comparison', fontweight='bold')
ax4_3.legend(framealpha=0.9)

fig4.suptitle('Model Comparison: Impact of VOCs on Ozone Formation', fontsize=14, fontweight='bold')

def animate4(frame):
    idx = int((frame / num_frames) * len(t_clock))
    if idx == 0:
        idx = 1
    
    t_current = t_clock[:idx]
    
    line4_O3_m1.set_data(t_current, O3_m1[:idx])
    line4_O3_m2.set_data(t_current, O3_m2[:idx])
    line4_NO_m1.set_data(t_current, NO_m1[:idx])
    line4_NO_m2.set_data(t_current, NO_m2[:idx])
    line4_NO2_m1.set_data(t_current, NO2_m1[:idx])
    line4_NO2_m2.set_data(t_current, NO2_m2[:idx])
    
    t_max = t_current[-1]
    for ax in [ax4_1, ax4_2, ax4_3]:
        for patch in [p for p in ax.patches if len(p.get_facecolor()) > 0 and p.get_facecolor()[0] > 0.9]:
            patch.remove()
        add_daylight_shading(ax, t_max)
    
    return line4_O3_m1, line4_O3_m2, line4_NO_m1, line4_NO_m2, line4_NO2_m1, line4_NO2_m2

anim4 = FuncAnimation(fig4, animate4, frames=num_frames, interval=1000/fps, blit=True, repeat=True)
writer4 = PillowWriter(fps=fps)
anim4.save('comparison_models.gif', writer=writer4, dpi=300)
print(f"✓ Saved: comparison_models.gif ({os.path.getsize('comparison_models.gif') / (1024*1024):.2f} MB)")
plt.close(fig4)
if SAVE_MP4 and ffmpeg_available:
    try:
        mp4_writer = FFMpegWriter(fps=fps, metadata={'artist': 'animate_all_models'}, bitrate=8000)
        anim4.save('comparison_models.mp4', writer=mp4_writer, dpi=300)
        print(f"✓ Saved: comparison_models.mp4 ({os.path.getsize('comparison_models.mp4') / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"Failed to save comparison_models.mp4: {e}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ALL ANIMATIONS COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  1. model1_results.gif")
print("  2. model1_results_1.gif")
print("  3. model2_results.gif")
print("  4. comparison_models.gif")
print(f"\nAnimation settings:")
print(f"  - Frames: {num_frames}")
print(f"  - FPS: {fps}")
print(f"  - Duration: {num_frames/fps:.1f} seconds each")
print("="*70)
