"""
Photochemical Smog Modeling: Model 1 and Model 2
Combined script with publication-quality figures
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from latex import latexify
latexify(columns=2)
import warnings
warnings.filterwarnings('ignore')

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16

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
    t_solar = t + 5  # Convert from simulation time to clock time
    if 6 <= t_solar <= 18:
        return np.sin(np.pi * (t_solar - 6) / 12)
    return 0.0

def add_daylight_shading(ax):
    """Add yellow shading for daylight hours"""
    ax.axvspan(6, 18, alpha=0.15, color='yellow', zorder=0)

# ============================================================================
# MODEL 1: BASIC PHOTOCHEMICAL CYCLE
# ============================================================================

# Parameters
T = 288.0  # K
O2 = 210000.0  # ppm

k1_max = 0.508 * 60  # h^-1
k2 = 3.9e-6 * np.exp(510/T) * 60
k3 = 3.1e3 * np.exp(-1450/T) * 60
k4 = 1.34e4 * 60
k5 = 5.6e2 * np.exp(584/T) * 60

E_NO_m1 = 0.02  # ppm/h
E_NO2_m1 = 0.01

def k1_photolysis(t):
    """NO2 photolysis rate"""
    return k1_max * solar_intensity(t)

def model1_odes(C, t):
    """Model 1: Basic 4-species system"""
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

# Solve Model 1
print("="*70)
print("MODEL 1: BASIC PHOTOCHEMICAL CYCLE")
print("="*70)

C0_m1 = [0.1, 0.05, 0.0, 0.0]  # [NO, NO2, O3, O]
t_span = np.linspace(0, 19, 2000)  # 5:00 AM to midnight

sol_m1 = odeint(model1_odes, C0_m1, t_span)
NO_m1 = sol_m1[:, 0]
NO2_m1 = sol_m1[:, 1]
O3_m1 = sol_m1[:, 2]
O_m1 = sol_m1[:, 3]
t_clock = t_span + 5

print(f"Peak O3: {O3_m1.max():.6f} ppm at {t_clock[O3_m1.argmax()]:.1f}:00")
print(f"Peak NO2: {NO2_m1.max():.4f} ppm")
print(f"Min NO: {NO_m1.min():.4f} ppm")

# ============================================================================
# MODEL 2: REFINED WITH VOCs
# ============================================================================

# Additional parameters
H2O = 15000.0  # ppm
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
    """O3 photolysis rate"""
    return k4_max * solar_intensity(t)

def k7_photolysis(t):
    """HCHO photolysis rate"""
    return k7_max * solar_intensity(t)

def model2_odes(C, t):
    """Model 2: 11-species system with VOCs and radicals"""
    NO, NO2, O3, O, CO, HCHO, ALK, OLE, OH, HO2, RO2 = C
    
    k1 = k1_max_m2 * solar_intensity(t)
    k4 = k4_photolysis(t)
    k7 = k7_photolysis(t)
    
    # Quasi-steady-state for O(1D)
    if k5_m2 * H2O > 0:
        O1D = k4 * O3 / (k5_m2 * H2O)
    else:
        O1D = 0.0
    
    # Reaction rates
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
    
    # ODEs
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

# Solve Model 2
print("\n" + "="*70)
print("MODEL 2: REFINED WITH VOCs AND RADICALS")
print("="*70)

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

print(f"Peak O3: {O3_m2.max():.6f} ppm at {t_clock[O3_m2.argmax()]:.1f}:00")
print(f"Peak OH: {OH_m2.max():.2e} ppm")
print(f"Peak HO2: {HO2_m2.max():.2e} ppm")
print(f"Enhancement factor: {O3_m2.max()/O3_m1.max():.1f}×")

# ============================================================================
# FIGURE 1: MODEL 1 RESULTS
# ============================================================================

fig1 = plt.figure(figsize=(14, 10))
gs1 = GridSpec(2, 2, figure=fig1, hspace=0.3, wspace=0.3)

# Plot 1: NO and NO2
ax1 = fig1.add_subplot(gs1[0, 0])
ax1.plot(t_clock, NO_m1, color=COLORS['NO'], linewidth=2.5, label='NO')
ax1.plot(t_clock, NO2_m1, color=COLORS['NO2'], linewidth=2.5, label=r'NO$_2$')
add_daylight_shading(ax1)
ax1.set_xlabel('Time (hours)', fontweight='bold')
ax1.set_ylabel('Concentration (ppm)', fontweight='bold')
ax1.set_title(r'(a) Primary Pollutants: NO and NO$_2$', fontweight='bold')
ax1.legend(framealpha=0.9)
ax1.grid(alpha=0.3)
ax1.set_xlim(5, 24)

# Plot 2: Ozone
ax2 = fig1.add_subplot(gs1[0, 1])
ax2.plot(t_clock, O3_m1, color=COLORS['O3'], linewidth=3)
ax2.fill_between(t_clock, 0, O3_m1, color=COLORS['O3'], alpha=0.2)
add_daylight_shading(ax2)
peak_idx = O3_m1.argmax()
ax2.plot(t_clock[peak_idx], O3_m1[peak_idx], 'r*', markersize=15)
ax2.set_xlabel('Time (hours)', fontweight='bold')
ax2.set_ylabel('Concentration (ppm)', fontweight='bold')
ax2.set_title(r'(b) Secondary Pollutant: Ozone (O$_3$)', fontweight='bold')
ax2.grid(alpha=0.3)
ax2.set_xlim(5, 24)

# Plot 3: Atomic Oxygen
ax3 = fig1.add_subplot(gs1[1, 0])
ax3.plot(t_clock, O_m1 * 1e6, 'm-', linewidth=2.5)
add_daylight_shading(ax3)
ax3.set_xlabel('Time (hours)', fontweight='bold')
ax3.set_ylabel(r'Concentration (ppm $\times$ 10$^6$)', fontweight='bold')
ax3.set_title('(c) Reactive Intermediate: Atomic Oxygen (O)', fontweight='bold')
ax3.grid(alpha=0.3)
ax3.set_xlim(5, 24)

# Plot 4: Normalized comparison
ax4 = fig1.add_subplot(gs1[1, 1])
ax4.plot(t_clock, NO_m1/NO_m1.max(), color=COLORS['NO'], linewidth=2, label='NO')
ax4.plot(t_clock, NO2_m1/NO2_m1.max(), color=COLORS['NO2'], linewidth=2, label=r'NO$_2$')
ax4.plot(t_clock, O3_m1/O3_m1.max(), color=COLORS['O3'], linewidth=2, label=r'O$_3$')
add_daylight_shading(ax4)
ax4.set_xlabel('Time (hours)', fontweight='bold')
ax4.set_ylabel('Normalized Concentration', fontweight='bold')
ax4.set_title('(d) Comparative Dynamics (Normalized)', fontweight='bold')
ax4.legend(framealpha=0.9)
ax4.grid(alpha=0.3)
ax4.set_xlim(5, 24)
ax4.set_ylim(0, 1.1)

fig1.suptitle('Model 1: Basic Photochemical Cycle', fontsize=14, fontweight='bold')
plt.savefig('model1_results.pdf', dpi=300, bbox_inches='tight')
print("\n✓ Saved: model1_results.pdf")
plt.show()

# ============================================================================
# FIGURE 2: MODEL 2 COMPREHENSIVE RESULTS
# ============================================================================

fig2 = plt.figure(figsize=(16, 12))
gs2 = GridSpec(3, 3, figure=fig2, hspace=0.35, wspace=0.35)

# Plot 1: Ozone
ax1 = fig2.add_subplot(gs2[0, 0])
ax1.plot(t_clock, O3_m2, color=COLORS['O3'], linewidth=3)
ax1.fill_between(t_clock, 0, O3_m2, color=COLORS['O3'], alpha=0.2)
add_daylight_shading(ax1)
peak_idx = O3_m2.argmax()
ax1.plot(t_clock[peak_idx], O3_m2[peak_idx], 'r*', markersize=12)
ax1.set_xlabel('Time (hours)', fontweight='bold')
ax1.set_ylabel('Concentration (ppm)', fontweight='bold')
ax1.set_title('(a) Ozone Formation', fontweight='bold')
ax1.grid(alpha=0.3)
ax1.set_xlim(5, 24)

# Plot 2: NO and NO2
ax2 = fig2.add_subplot(gs2[0, 1])
ax2.plot(t_clock, NO_m2, color=COLORS['NO'], linewidth=2.5, label='NO')
ax2.plot(t_clock, NO2_m2, color=COLORS['NO2'], linewidth=2.5, label=r'NO$_2$')
add_daylight_shading(ax2)
ax2.set_xlabel('Time (hours)', fontweight='bold')
ax2.set_ylabel('Concentration (ppm)', fontweight='bold')
ax2.set_title('(b) Nitrogen Oxides', fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xlim(5, 24)

# Plot 3: Radicals
ax3 = fig2.add_subplot(gs2[0, 2])
ax3.plot(t_clock, OH_m2*1e6, color=COLORS['OH'], linewidth=2, label=r'OH ($\times 10^6$)')
ax3.plot(t_clock, HO2_m2*1e3, color=COLORS['HO2'], linewidth=2, label=r'HO$_2$ ($\times 10^3$)')
ax3.plot(t_clock, RO2_m2*1e3, color=COLORS['RO2'], linewidth=2, label=r'RO$_2$ ($\times 10^3$)')
add_daylight_shading(ax3)
ax3.set_xlabel('Time (hours)', fontweight='bold')
ax3.set_ylabel('Scaled Concentration', fontweight='bold')
ax3.set_title('(c) Radical Species', fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)
ax3.set_xlim(5, 24)

# Plot 4: VOCs
ax4 = fig2.add_subplot(gs2[1, 0])
ax4.plot(t_clock, ALK_m2, color=COLORS['ALK'], linewidth=2.5, label='ALK')
ax4.plot(t_clock, OLE_m2, color=COLORS['OLE'], linewidth=2.5, label='OLE')
add_daylight_shading(ax4)
ax4.set_xlabel('Time (hours)', fontweight='bold')
ax4.set_ylabel('Concentration (ppm)', fontweight='bold')
ax4.set_title('(d) Volatile Organic Compounds', fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)
ax4.set_xlim(5, 24)

# Plot 5: Secondary products
ax5 = fig2.add_subplot(gs2[1, 1])
ax5.plot(t_clock, HCHO_m2, color=COLORS['HCHO'], linewidth=2.5, label='HCHO')
ax5.plot(t_clock, CO_m2, color=COLORS['CO'], linewidth=2.5, label='CO')
add_daylight_shading(ax5)
ax5.set_xlabel('Time (hours)', fontweight='bold')
ax5.set_ylabel('Concentration (ppm)', fontweight='bold')
ax5.set_title('(e) Secondary Products', fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3)
ax5.set_xlim(5, 24)

# Plot 6: NO→NO2 pathways
ax6 = fig2.add_subplot(gs2[1, 2])
rate_O3 = k3_m2 * O3_m2 * NO_m2
rate_HO2 = k12 * HO2_m2 * NO_m2
rate_RO2 = k13 * RO2_m2 * NO_m2
ax6.plot(t_clock, rate_O3, 'g--', linewidth=2, label=r'O$_3$ + NO', alpha=0.7)
ax6.plot(t_clock, rate_HO2, color=COLORS['HO2'], linewidth=2, label=r'HO$_2$ + NO')
ax6.plot(t_clock, rate_RO2, color=COLORS['RO2'], linewidth=2, label=r'RO$_2$ + NO')
add_daylight_shading(ax6)
ax6.set_xlabel('Time (hours)', fontweight='bold')
ax6.set_ylabel('Rate (ppm/h)', fontweight='bold')
ax6.set_title(r'(f) NO $\to$ NO$_2$ Pathways', fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)
ax6.set_xlim(5, 24)

# Plot 7: NOx budget
ax7 = fig2.add_subplot(gs2[2, 0])
NOx_m2 = NO_m2 + NO2_m2
ax7.plot(t_clock, NOx_m2, 'darkred', linewidth=3, label=r'NO$_x$ (NO + NO$_2$)')
ax7.plot(t_clock, NO_m2, color=COLORS['NO'], linewidth=1.5, linestyle='--', alpha=0.6)
ax7.plot(t_clock, NO2_m2, color=COLORS['NO2'], linewidth=1.5, linestyle='--', alpha=0.6)
add_daylight_shading(ax7)
ax7.set_xlabel('Time (hours)', fontweight='bold')
ax7.set_ylabel('Concentration (ppm)', fontweight='bold')
ax7.set_title(r'(g) NO$_x$ Budget', fontweight='bold')
ax7.legend()
ax7.grid(alpha=0.3)
ax7.set_xlim(5, 24)

# Plot 8: Oxidant capacity
ax8 = fig2.add_subplot(gs2[2, 1])
Ox = O3_m2 + NO2_m2
ax8.plot(t_clock, Ox, 'darkblue', linewidth=3)
add_daylight_shading(ax8)
ax8.set_xlabel('Time (hours)', fontweight='bold')
ax8.set_ylabel('Concentration (ppm)', fontweight='bold')
ax8.set_title(r'(h) Oxidant Capacity (O$_x$)', fontweight='bold')
ax8.grid(alpha=0.3)
ax8.set_xlim(5, 24)

# Plot 9: NO/NO2 ratio
ax9 = fig2.add_subplot(gs2[2, 2])
NO2_safe = np.where(NO2_m2 > 1e-6, NO2_m2, 1e-6)
ratio = NO_m2 / NO2_safe
ax9.plot(t_clock, ratio, 'purple', linewidth=2.5)
add_daylight_shading(ax9)
ax9.set_xlabel('Time (hours)', fontweight='bold')
ax9.set_ylabel('Ratio', fontweight='bold')
ax9.set_title(r'(i) NO/NO$_2$ Ratio', fontweight='bold')
ax9.grid(alpha=0.3)
ax9.set_xlim(5, 24)
ax9.set_ylim(0, 5)

fig2.suptitle('Model 2: Refined with VOCs and Radicals', fontsize=14, fontweight='bold')
plt.savefig('model2_results.pdf', dpi=300, bbox_inches='tight')
print("✓ Saved: model2_results.pdf")
plt.show()

# ============================================================================
# FIGURE 3: MODEL COMPARISON
# ============================================================================

fig3 = plt.figure(figsize=(14, 10))
gs3 = GridSpec(2, 2, figure=fig3, hspace=0.3, wspace=0.3)

# Ozone comparison - spanning both top columns
ax1 = fig3.add_subplot(gs3[0, :])
ax1.plot(t_clock, O3_m1, 'g--', linewidth=3, label='Model 1 (No VOCs)', alpha=0.7)
ax1.plot(t_clock, O3_m2, 'g-', linewidth=3, label='Model 2 (With VOCs)')
add_daylight_shading(ax1)
ax1.set_xlabel('Time (hours)', fontweight='bold')
ax1.set_ylabel(r'O$_3$ Concentration (ppm)', fontweight='bold')
ax1.set_title('(a) Ozone: Model Comparison', fontweight='bold')
ax1.legend(framealpha=0.9)
ax1.grid(alpha=0.3)
ax1.set_xlim(5, 24)

# NO comparison
ax2 = fig3.add_subplot(gs3[1, 0])
ax2.plot(t_clock, NO_m1, 'b--', linewidth=2.5, label='Model 1', alpha=0.7)
ax2.plot(t_clock, NO_m2, 'b-', linewidth=2.5, label='Model 2')
add_daylight_shading(ax2)
ax2.set_xlabel('Time (hours)', fontweight='bold')
ax2.set_ylabel('NO Concentration (ppm)', fontweight='bold')
ax2.set_title('(b) NO: Model Comparison', fontweight='bold')
ax2.legend(framealpha=0.9)
ax2.grid(alpha=0.3)
ax2.set_xlim(5, 24)

# NO2 comparison
ax3 = fig3.add_subplot(gs3[1, 1])
ax3.plot(t_clock, NO2_m1, 'r--', linewidth=2.5, label='Model 1', alpha=0.7)
ax3.plot(t_clock, NO2_m2, 'r-', linewidth=2.5, label='Model 2')
add_daylight_shading(ax3)
ax3.set_xlabel('Time (hours)', fontweight='bold')
ax3.set_ylabel(r'NO$_2$ Concentration (ppm)', fontweight='bold')
ax3.set_title(r'(c) NO$_2$: Model Comparison', fontweight='bold')
ax3.legend(framealpha=0.9)
ax3.grid(alpha=0.3)
ax3.set_xlim(5, 24)

fig3.suptitle('Model Comparison: Impact of VOCs on Ozone Formation', 
             fontsize=14, fontweight='bold')
plt.savefig('comparison_models.pdf', dpi=300, bbox_inches='tight')
print("✓ Saved: comparison_models.pdf")
plt.show()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print("\nMODEL 1:")
print(f"  Peak O₃: {O3_m1.max():.6f} ppm at {t_clock[O3_m1.argmax()]:.1f}:00")
print(f"  Peak NO₂: {NO2_m1.max():.4f} ppm")
print(f"  Min NO: {NO_m1.min():.4f} ppm")

print("\nMODEL 2:")
idx_peak = O3_m2.argmax()
print(f"  Peak O₃: {O3_m2.max():.6f} ppm at {t_clock[idx_peak]:.1f}:00")
print(f"  Peak OH: {OH_m2.max():.2e} ppm at {t_clock[OH_m2.argmax()]:.1f}:00")
print(f"  Peak HO₂: {HO2_m2.max():.2e} ppm")
print(f"  Peak RO₂: {RO2_m2.max():.2e} ppm")
print(f"  Min NO: {NO_m2.min():.6f} ppm")

print("\nCOMPARISON:")
print(f"  O₃ Enhancement: {O3_m2.max()/O3_m1.max():.1f}×")
print(f"  Additional O₃: {(O3_m2.max()-O3_m1.max())*1000:.1f} ppb")

print("\nVOC CONSUMPTION (Model 2):")
print(f"  ALK: {(1-ALK_m2[-1]/C0_m2[6])*100:.1f}% consumed")
print(f"  OLE: {(1-OLE_m2[-1]/C0_m2[7])*100:.1f}% consumed")

print("\nNO→NO₂ PATHWAYS at peak O₃ (Model 2):")
rate_O3_peak = k3_m2 * O3_m2[idx_peak] * NO_m2[idx_peak]
rate_HO2_peak = k12 * HO2_m2[idx_peak] * NO_m2[idx_peak]
rate_RO2_peak = k13 * RO2_m2[idx_peak] * NO_m2[idx_peak]
total = rate_O3_peak + rate_HO2_peak + rate_RO2_peak
if total > 0:
    print(f"  Via O₃: {rate_O3_peak/total*100:.1f}%")
    print(f"  Via HO₂: {rate_HO2_peak/total*100:.1f}%")
    print(f"  Via RO₂: {rate_RO2_peak/total*100:.1f}%")

print("\n" + "="*70)
print("All figures saved successfully!")
print("="*70)