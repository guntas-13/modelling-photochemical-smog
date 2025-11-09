"""
GRS (Generic Reaction Set) Model for Photochemical Smog
Simplified 5-species model based on Jacobson's framework
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
# from latex import latexify
# latexify(columns=2)

# ============================================================================
# GRS MODEL
# ============================================================================

def solar_intensity(t, k_max):
    """Solar intensity function (sinusoidal, 6-18h)"""
    hour = t % 24
    if 6 <= hour <= 18:
        return k_max * np.sin(np.pi * (hour - 6) / 12)
    return 0.0

def grs_model(y, t, params):
    """
    GRS Model ODEs
    Species: ROC, RP, NO, NO2, O3
    
    Reactions:
    1. ROC + hv -> RP + ROC       (k1)
    2. RP + NO -> NO2             (k2)
    3. NO2 + hv -> NO + O3        (k3)
    4. NO + O3 -> NO2             (k4)
    5. RP + RP -> RP              (k5)
    6. RP + NO2 -> SGN            (k6)
    7. RP + NO2 -> SNGN           (k7)
    """
    ROC, RP, NO, NO2, O3 = y
    
    # Time-dependent photolysis rates
    k1 = solar_intensity(t, params['k1_max'])
    k3 = solar_intensity(t, params['k3_max'])
    
    # Extract parameters
    k2 = params['k2']
    k4 = params['k4']
    k5 = params['k5']
    k6 = params['k6']
    k7 = params['k7']
    E_ROC = params['E_ROC']
    E_NO = params['E_NO']
    
    # Reaction rates
    R1 = k1 * ROC
    R2 = k2 * RP * NO
    R3 = k3 * NO2
    R4 = k4 * NO * O3
    R5 = k5 * RP * RP
    R6 = k6 * RP * NO2
    R7 = k7 * RP * NO2
    
    # ODEs
    dROC_dt = -R1 + E_ROC
    dRP_dt = R1 - R2 - 2*R5 - R6 - R7
    dNO_dt = R3 - R2 - R4 + E_NO
    dNO2_dt = R2 + R4 - R3 - R6 - R7
    dO3_dt = R3 - R4
    
    return [dROC_dt, dRP_dt, dNO_dt, dNO2_dt, dO3_dt]

# ============================================================================
# SIMULATION
# ============================================================================

# Parameters (from Carrasco-Venegas mapping)
params = {
    'k1_max': 0.00284,      # min^-1 (ROC photolysis)
    'k2': 12000.0,          # ppm^-1 min^-1 (RP + NO)
    'k3_max': 0.508,        # min^-1 (NO2 photolysis)
    'k4': 20.0,             # ppm^-1 min^-1 (NO + O3)
    'k5': 3700.0,           # ppm^-1 min^-1 (RP termination)
    'k6': 10000.0,          # ppm^-1 min^-1 (HNO3 formation)
    'k7': 2000.0,           # ppm^-1 min^-1 (PAN formation)
    'E_ROC': 0.03 / 60,     # ppm/min (VOC emissions)
    'E_NO': 0.02 / 60       # ppm/min (NO emissions)
}

# Initial conditions [ROC, RP, NO, NO2, O3]
y0 = [0.020, 1.0e-5, 0.100, 0.050, 0.020]

# Time (24 hours)
t = np.linspace(0, 24, 24 * 60 + 1)

# Solve
print("Running GRS Model...")
solution = odeint(grs_model, y0, t, args=(params,))

ROC = solution[:, 0]
RP = solution[:, 1]
NO = solution[:, 2]
NO2 = solution[:, 3]
O3 = solution[:, 4]

print(f"Peak O3: {O3.max():.4f} ppm at t={t[O3.argmax()]:.1f}h")
print(f"Peak RP: {RP.max():.2e} ppm")

# ============================================================================
# PLOTTING
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('GRS Model: Generic Reaction Set for Photochemical Smog', 
             fontsize=14, fontweight='bold')

# Add day/night shading
def add_shading(ax):
    for day in range(3):
        ax.axvspan(day*24, day*24+6, alpha=0.05, color='gray')
        ax.axvspan(day*24+18, (day+1)*24, alpha=0.05, color='gray')

# Plot 1: Primary species
ax1 = axes[0, 0]
ax1.plot(t, NO, 'b-', linewidth=2, label='NO')
ax1.plot(t, NO2, 'r-', linewidth=2, label=r'NO$_2$')
ax1.plot(t, O3, 'g-', linewidth=2, label=r'O$_3$')
add_shading(ax1)
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Concentration (ppm)')
ax1.set_title(r'Primary Species: NO-NO$_2$-O$_3$')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_xlim(0, 24)

# Plot 2: ROC and RP
ax2 = axes[0, 1]
ax2.plot(t, ROC, 'purple', linewidth=2, label='ROC (VOCs)')
ax2_twin = ax2.twinx()
ax2_twin.semilogy(t, RP, 'orange', linewidth=2, label='RP (Radicals)')
add_shading(ax2)
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('ROC (ppm)', color='purple')
ax2_twin.set_ylabel('RP (ppm, log)', color='orange')
ax2.set_title('ROC and RP Evolution')
ax2.tick_params(axis='y', labelcolor='purple')
ax2_twin.tick_params(axis='y', labelcolor='orange')
ax2.grid(alpha=0.3)
ax2.set_xlim(0, 24)

# Plot 3: O3 focus
ax3 = axes[1, 0]
ax3.plot(t, O3, 'g-', linewidth=3)
ax3.fill_between(t, 0, O3, alpha=0.3, color='green')
add_shading(ax3)
peak_idx = O3.argmax()
ax3.plot(t[peak_idx], O3[peak_idx], 'r*', markersize=15)
ax3.annotate(f'Peak: {O3.max():.3f} ppm\n@{t[peak_idx]:.1f}h',
            xy=(t[peak_idx], O3[peak_idx]),
            xytext=(t[peak_idx]+5, O3[peak_idx]-0.01),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, fontweight='bold')
ax3.set_xlabel('Time (hours)')
ax3.set_ylabel(r'O$_3$ Concentration (ppm)')
ax3.set_title('Ozone Production')
ax3.grid(alpha=0.3)
ax3.set_xlim(0, 24)

# Plot 4: Diurnal pattern
ax4 = axes[1, 1]
hours = np.arange(24)
NO_avg = [NO[(t >= h) & (t < h+1)].mean() for h in hours]
NO2_avg = [NO2[(t >= h) & (t < h+1)].mean() for h in hours]
O3_avg = [O3[(t >= h) & (t < h+1)].mean() for h in hours]

ax4.plot(hours, NO_avg, 'o-', linewidth=2, markersize=6, label='NO')
ax4.plot(hours, NO2_avg, 's-', linewidth=2, markersize=6, label=r'NO$_2$')
ax4.plot(hours, O3_avg, '^-', linewidth=2, markersize=6, label=r'O$_3$')
ax4.axvspan(0, 6, alpha=0.1, color='gray')
ax4.axvspan(18, 24, alpha=0.1, color='gray')
ax4.set_xlabel('Hour of Day')
ax4.set_ylabel('Avg Concentration (ppm)')
ax4.set_title('Average Diurnal Pattern')
ax4.legend()
ax4.grid(alpha=0.3)
ax4.set_xlim(0, 24)
ax4.set_xticks([0, 6, 12, 18, 24])

plt.tight_layout()
plt.savefig('grs_model_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot saved: grs_model_results.png")
plt.show()

# ============================================================================
# KEY INSIGHTS
# ============================================================================

print("\n" + "="*60)
print("GRS MODEL INSIGHTS")
print("="*60)
print("\nReaction Competition:")
print(f"  • R2 (RP+NO→NO$_2$): Produces NO$_2$, enables O$_3$ formation")
print(f"  • R4 (NO+O$_3$→NO$_2$): Consumes O$_3$, competes with R2")
print(f"  • When NO high: R2 dominates → net O$_3$ production")
print(f"  • When O$_3$ high: R4 dominates → O$_3$ titration")
print(f"  • R6+R7: Remove NOₓ → O$_3$ production stops")

print(f"\nPeak Timing:")
print(f"  • NO peaks: Night (emissions + low photolysis)")
print(f"  • NO$_2$ peaks: Evening (accumulated from reactions)")
print(f"  • O$_3$ peaks: Afternoon ({t[peak_idx]:.1f}h, {O3.max():.3f} ppm)")
print(f"  • RP peaks: Noon (maximum photolysis)")

print("\n" + "="*60)