import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
from latex import latexify
latexify(columns=2)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARAMETERS
# ============================================================================

# Temperature (K)
T = 288.0

# Rate constants (converted to per hour and ppm units)
k1_max = 0.508 * 60  # Max photolysis rate (1/h)
k2 = 3.9e-6 * np.exp(510/T) * 60  # O3 formation (ppm^-1 h^-1)
k3 = 3.1e3 * np.exp(-1450/T) * 60  # NO oxidation by O3 (ppm^-1 h^-1)
k4 = 1.34e4 * 60  # NO2 reduction by O (ppm^-1 h^-1)
k5 = 5.6e2 * np.exp(584/T) * 60  # NO oxidation by O (ppm^-1 h^-1)

# Constant concentrations
O2 = 210000.0  # ppm (atmospheric oxygen)

# Emission rates (ppm/h)
E_NO = 0.02
E_NO2 = 0.01

# Initial conditions [NO, NO2, O3, O] at t=0 (5:00 AM)
C0 = [0.1, 0.05, 0.0, 0.0]

# ============================================================================
# SOLAR RADIATION FUNCTION
# ============================================================================

def k1_photolysis(t):
    """
    Photolysis rate constant for NO2 as function of time
    Solar radiation: 6:00 AM to 6:00 PM
    
    Parameters:
        t : float, time in hours (0 = 5:00 AM)
    
    Returns:
        float : photolysis rate constant (1/h)
    """
    t_solar = t + 5  # Convert to clock time
    
    if 6 <= t_solar <= 18:
        # Sine function peaks at noon
        return k1_max * np.sin(np.pi * (t_solar - 6) / 12)
    else:
        return 0.0

# ============================================================================
# DIFFERENTIAL EQUATIONS
# ============================================================================

def photochemical_model(C, t):
    """
    System of ODEs for photochemical smog formation
    
    Parameters:
        C : list, concentrations [NO, NO2, O3, O]
        t : float, time (hours since 5:00 AM)
    
    Returns:
        list : derivatives [dNO/dt, dNO2/dt, dO3/dt, dO/dt]
    """
    NO, NO2, O3, O = C
    
    # Get time-dependent photolysis rate
    k1 = k1_photolysis(t)
    
    # Reaction rates (R1-R5)
    R1 = k1 * NO2
    R2 = k2 * O * O2
    R3 = k3 * O3 * NO
    R4 = k4 * NO2 * O
    R5 = k5 * NO * O
    
    # Species mass balances
    dNO_dt = R1 - R3 + R4 - R5 + E_NO
    dNO2_dt = -R1 + R3 - R4 + R5 + E_NO2
    dO3_dt = R2 - R3
    dO_dt = R1 - R2 - R4 - R5
    
    return [dNO_dt, dNO2_dt, dO3_dt, dO_dt]

# ============================================================================
# SOLVE ODEs
# ============================================================================

# Time span: 19 hours (5:00 AM to midnight)
t_span = np.linspace(0, 19, 1000)

# Solve system
solution = odeint(photochemical_model, C0, t_span)

# Extract solutions
NO = solution[:, 0]
NO2 = solution[:, 1]
O3 = solution[:, 2]
O = solution[:, 3]

# Convert time to clock hours
t_clock = t_span + 5

# ============================================================================
# VISUALIZATION
# ============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: NO and NO2
ax1 = axes[0, 0]
ax1.plot(t_clock, NO, 'b-', linewidth=2, label='NO')
ax1.plot(t_clock, NO2, 'r-', linewidth=2, label=r'NO$_2$')
ax1.axvspan(6, 18, alpha=0.2, color='yellow', label='Daylight')
ax1.set_xlabel('Time (hours)', fontsize=12)
ax1.set_ylabel('Concentration (ppm)', fontsize=12)
ax1.set_title(r'Primary Pollutants: NO and NO$_2$', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(5, 24)

# Plot 2: Ozone
ax2 = axes[0, 1]
ax2.plot(t_clock, O3, 'g-', linewidth=2.5, label=r'O$_3$')
ax2.axvspan(6, 18, alpha=0.2, color='yellow', label='Daylight')
ax2.set_xlabel('Time (hours)', fontsize=12)
ax2.set_ylabel('Concentration (ppm)', fontsize=12)
ax2.set_title(r'Secondary Pollutant: Ozone (O$_3$)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(5, 24)

# Plot 3: Atomic Oxygen
ax3 = axes[1, 0]
ax3.plot(t_clock, O * 1e6, 'm-', linewidth=2, label=r'O (x10$^6$)')
ax3.axvspan(6, 18, alpha=0.2, color='yellow', label='Daylight')
ax3.set_xlabel('Time (hours)', fontsize=12)
ax3.set_ylabel(r'Concentration (ppm x 10$^6$)', fontsize=12)
ax3.set_title(r'Reactive Intermediate: Atomic Oxygen (O)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(5, 24)

# Plot 4: All species together (normalized)
ax4 = axes[1, 1]
ax4.plot(t_clock, NO / np.max(NO), 'b-', linewidth=2, label=r'NO (norm.)')
ax4.plot(t_clock, NO2 / np.max(NO2), 'r-', linewidth=2, label=r'NO$_2$ (norm.)')
ax4.plot(t_clock, O3 / np.max(O3), 'g-', linewidth=2, label=r'O$_3$ (norm.)')
ax4.axvspan(6, 18, alpha=0.2, color='yellow', label='Daylight')
ax4.set_xlabel('Time (hours)', fontsize=12)
ax4.set_ylabel('Normalized Concentration', fontsize=12)
ax4.set_title('Comparative Dynamics (Normalized)', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(5, 24)

plt.tight_layout()
# plt.savefig('iteration1_photochemical_smog.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("ITERATION 1: SIMPLE PHOTOCHEMICAL SMOG MODEL - RESULTS SUMMARY")
print("="*70)

print(f"\nTemperature: {T} K")
print(f"\nRate Constants:")
print(f"  k1_max (photolysis): {k1_max:.3f} h⁻¹")
print(f"  k2 (O3 formation): {k2:.6e} ppm⁻¹h⁻¹")
print(f"  k3 (NO oxidation): {k3:.6e} ppm⁻¹h⁻¹")

print(f"\nInitial Concentrations (5:00 AM):")
print(f"  [NO]₀  = {C0[0]:.3f} ppm")
print(f"  [NO₂]₀ = {C0[1]:.3f} ppm")
print(f"  [O₃]₀  = {C0[2]:.3f} ppm")

print(f"\nPeak Concentrations:")
idx_max_O3 = np.argmax(O3)
print(f"  Max [O₃]  = {np.max(O3):.4f} ppm at {t_clock[idx_max_O3]:.1f}:00")
print(f"  Max [NO₂] = {np.max(NO2):.4f} ppm at {t_clock[np.argmax(NO2)]:.1f}:00")
print(f"  Min [NO]  = {np.min(NO):.4f} ppm at {t_clock[np.argmin(NO)]:.1f}:00")

print(f"\nFinal Concentrations (midnight):")
print(f"  [NO]  = {NO[-1]:.4f} ppm")
print(f"  [NO₂] = {NO2[-1]:.4f} ppm")
print(f"  [O₃]  = {O3[-1]:.4f} ppm")

print("\n" + "="*70)

# ============================================================================
# PARAMETERS
# ============================================================================

# Temperature (K)
T = 288.0

# Constants
O2 = 210000.0      # ppm (atmospheric oxygen)
H2O = 15000.0      # ppm (water vapor)
M = 1.0e6          # ppm (air, third body)

# Maximum photolysis rates (1/h)
k1_max = 0.508 * 60           # NO2 photolysis
k4_max = 0.0328 * 60          # O3 photolysis
k7_max = 0.00284 * 60         # HCHO photolysis

# Temperature-dependent rate constants
k2 = 3.9e-6 * np.exp(510/T) * 60           # O + O2 -> O3
k3 = 3.1e3 * np.exp(-1450/T) * 60          # O3 + NO -> NO2
k5 = 1.0e5 * 60                             # O(1D) + H2O -> 2OH
k6 = 4.4e2 * 60                             # CO + OH -> HO2
k8 = 19200.0 * 60                           # HCHO + OH -> HO2
k9 = 4700.0 * 60                            # ALK + OH -> RO2
k10 = 89142.0 * 60                          # OLE + OH -> RO2
k11 = 0.136 * 60                            # OLE + O3 -> products
k12 = 1.2e4 * 60                            # HO2 + NO -> NO2 + OH
k13 = 1.2e4 * 60                            # RO2 + NO -> NO2 + HO2
k14 = 3700.0 * 60                           # HO2 + HO2 -> H2O2
k15 = 1.477e15 * 10**(-11.6*T/(17.4+T)) * (280/T)**2 * 60  # OH + NO2 -> HNO3

# Emission rates (ppm/h)
E_NO = 0.02
E_NO2 = 0.01
E_CO = 0.02
E_HCHO = 0.03
E_ALK = 0.1
E_OLE = 0.0

# Initial conditions [NO, NO2, O3, O, CO, HCHO, ALK, OLE, OH, HO2, RO2]
C0 = [0.1, 0.05, 0.0, 0.0, 0.1, 0.01, 1.0, 0.2, 0.0, 0.0, 0.0]

# ============================================================================
# SOLAR RADIATION FUNCTIONS
# ============================================================================

def solar_intensity(t):
    """Solar intensity factor (0-1) based on time of day"""
    t_solar = t + 5  # Convert to clock time
    if 6 <= t_solar <= 18:
        return np.sin(np.pi * (t_solar - 6) / 12)
    else:
        return 0.0

def k1_photolysis(t):
    """NO2 photolysis rate"""
    return k1_max * solar_intensity(t)

def k4_photolysis(t):
    """O3 photolysis rate"""
    return k4_max * solar_intensity(t)

def k7_photolysis(t):
    """HCHO photolysis rate"""
    return k7_max * solar_intensity(t)

# ============================================================================
# DIFFERENTIAL EQUATIONS
# ============================================================================

def refined_photochemical_model(C, t):
    """
    Refined system of ODEs for photochemical smog with VOCs and radicals
    
    Species order: [NO, NO2, O3, O, CO, HCHO, ALK, OLE, OH, HO2, RO2]
    """
    NO, NO2, O3, O, CO, HCHO, ALK, OLE, OH, HO2, RO2 = C
    
    # Get time-dependent photolysis rates
    k1 = k1_photolysis(t)
    k4 = k4_photolysis(t)
    k7 = k7_photolysis(t)
    
    # Calculate O(1D) concentration (quasi-steady-state approximation)
    if k5 * H2O > 0:
        O1D = k4 * O3 / (k5 * H2O)
    else:
        O1D = 0.0
    
    # Reaction rates
    R1 = k1 * NO2
    R2 = k2 * O * O2
    R3 = k3 * O3 * NO
    R4 = k4 * O3
    R5 = k5 * O1D * H2O
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
    
    # Species mass balances
    dNO_dt = R1 - R3 - R12 - R13 + E_NO
    
    dNO2_dt = -R1 + R3 + R12 + R13 - R15 + E_NO2
    
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
# SOLVE ODEs
# ============================================================================

# Time span: 19 hours (5:00 AM to midnight)
t_span = np.linspace(0, 19, 2000)

# Solve system
print("Solving refined photochemical model...")
solution = odeint(refined_photochemical_model, C0, t_span, rtol=1e-6, atol=1e-8)

# Extract solutions
NO = solution[:, 0]
NO2 = solution[:, 1]
O3 = solution[:, 2]
O = solution[:, 3]
CO = solution[:, 4]
HCHO = solution[:, 5]
ALK = solution[:, 6]
OLE = solution[:, 7]
OH = solution[:, 8]
HO2 = solution[:, 9]
RO2 = solution[:, 10]

# Convert time to clock hours
t_clock = t_span + 5

# ============================================================================
# VISUALIZATION
# ============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(16, 12))

# Create grid for subplots
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Ozone comparison (will compare with Iteration 1 later)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t_clock, O3, 'g-', linewidth=2.5, label=r'O$_3$ (with VOCs)')
ax1.axvspan(6, 18, alpha=0.2, color='yellow')
ax1.set_xlabel('Time (hours)', fontsize=11)
ax1.set_ylabel('Concentration (ppm)', fontsize=11)
ax1.set_title('Ozone Formation', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(5, 24)

# Plot 2: NO and NO2
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t_clock, NO, 'b-', linewidth=2, label=r'NO')
ax2.plot(t_clock, NO2, 'r-', linewidth=2, label=r'NO$_2$')
ax2.axvspan(6, 18, alpha=0.2, color='yellow')
ax2.set_xlabel('Time (hours)', fontsize=11)
ax2.set_ylabel('Concentration (ppm)', fontsize=11)
ax2.set_title('Nitrogen Oxides', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(5, 24)

# Plot 3: Radicals (OH, HO2, RO2)
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(t_clock, OH * 1e6, 'purple', linewidth=2, label=r'OH (x 10$^6$)')
ax3.plot(t_clock, HO2 * 1e3, 'orange', linewidth=2, label=r'HO$_2$ (x 10$^3$)')
ax3.plot(t_clock, RO2 * 1e3, 'brown', linewidth=2, label=r'RO$_2$ (x 10$^3$)')
ax3.axvspan(6, 18, alpha=0.2, color='yellow')
ax3.set_xlabel('Time (hours)', fontsize=11)
ax3.set_ylabel('Scaled Concentration', fontsize=11)
ax3.set_title('Radical Species', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(5, 24)

# Plot 4: VOCs (ALK, OLE)
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(t_clock, ALK, 'darkgreen', linewidth=2, label='ALK (Alkanes)')
ax4.plot(t_clock, OLE, 'olive', linewidth=2, label='OLE (Olefins)')
ax4.axvspan(6, 18, alpha=0.2, color='yellow')
ax4.set_xlabel('Time (hours)', fontsize=11)
ax4.set_ylabel('Concentration (ppm)', fontsize=11)
ax4.set_title('Volatile Organic Compounds', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(5, 24)

# Plot 5: Secondary products (HCHO, CO)
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(t_clock, HCHO, 'magenta', linewidth=2, label='HCHO')
ax5.plot(t_clock, CO, 'gray', linewidth=2, label='CO')
ax5.axvspan(6, 18, alpha=0.2, color='yellow')
ax5.set_xlabel('Time (hours)', fontsize=11)
ax5.set_ylabel('Concentration (ppm)', fontsize=11)
ax5.set_title('Secondary Products', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.set_xlim(5, 24)

# Plot 6: O3 production mechanisms
ax6 = fig.add_subplot(gs[1, 2])
# Calculate contributions
NO_to_NO2_via_HO2 = k12 * HO2 * NO * 60  # Convert to ppm/h
NO_to_NO2_via_RO2 = k13 * RO2 * NO * 60
NO_to_NO2_via_O3 = k3 * O3 * NO * 60
ax6.plot(t_clock, NO_to_NO2_via_O3, 'g--', linewidth=2, label=r'O$_3$ + NO', alpha=0.7)
ax6.plot(t_clock, NO_to_NO2_via_HO2, 'orange', linewidth=2, label=r'HO$_2$ + NO')
ax6.plot(t_clock, NO_to_NO2_via_RO2, 'brown', linewidth=2, label=r'RO$_2$ + NO')
ax6.axvspan(6, 18, alpha=0.2, color='yellow')
ax6.set_xlabel('Time (hours)', fontsize=11)
ax6.set_ylabel('Rate (ppm/h)', fontsize=11)
ax6.set_title(r'NO $\to$ NO$_2$ Pathways', fontsize=12, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)
ax6.set_xlim(5, 24)

# Plot 7: NOx budget
ax7 = fig.add_subplot(gs[2, 0])
NOx = NO + NO2
ax7.plot(t_clock, NOx, 'darkred', linewidth=2.5, label=r'NO$_x$ (NO + NO$_2$)')
ax7.plot(t_clock, NO, 'b--', linewidth=1.5, label=r'NO', alpha=0.6)
ax7.plot(t_clock, NO2, 'r--', linewidth=1.5, label=r'NO$_2$', alpha=0.6)
ax7.axvspan(6, 18, alpha=0.2, color='yellow')
ax7.set_xlabel('Time (hours)', fontsize=11)
ax7.set_ylabel('Concentration (ppm)', fontsize=11)
ax7.set_title(r'NO$_x$ Budget', fontsize=12, fontweight='bold')
ax7.legend(fontsize=10)
ax7.grid(True, alpha=0.3)
ax7.set_xlim(5, 24)

# Plot 8: Oxidant capacity (O3 + NO2)
ax8 = fig.add_subplot(gs[2, 1])
Ox = O3 + NO2
ax8.plot(t_clock, Ox, 'darkblue', linewidth=2.5, label=r'O$_x$ (O$_3$ + NO$_2$)')
ax8.axvspan(6, 18, alpha=0.2, color='yellow')
ax8.set_xlabel('Time (hours)', fontsize=11)
ax8.set_ylabel('Concentration (ppm)', fontsize=11)
ax8.set_title('Oxidant Capacity', fontsize=12, fontweight='bold')
ax8.legend(fontsize=10)
ax8.grid(True, alpha=0.3)
ax8.set_xlim(5, 24)

# Plot 9: Key ratios
ax9 = fig.add_subplot(gs[2, 2])
# Avoid division by zero
NO2_safe = np.where(NO2 > 1e-6, NO2, 1e-6)
NO_NO2_ratio = NO / NO2_safe
VOC_NOx_ratio = (ALK + OLE) / (NO + NO2_safe)
ax9.plot(t_clock, NO_NO2_ratio, 'purple', linewidth=2, label=r'NO/NO$_2$')
ax9.axvspan(6, 18, alpha=0.2, color='yellow')
ax9.set_xlabel('Time (hours)', fontsize=11)
ax9.set_ylabel('Ratio', fontsize=11)
ax9.set_title(r'NO/NO$_2$ Ratio', fontsize=12, fontweight='bold')
ax9.legend(fontsize=10)
ax9.grid(True, alpha=0.3)
ax9.set_xlim(5, 24)
ax9.set_ylim(0, 5)

plt.suptitle('Iteration 2: Refined Photochemical Smog Model with VOCs and Radicals', 
             fontsize=14, fontweight='bold', y=0.995)

# plt.savefig('iteration2_refined_model.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# COMPARISON WITH ITERATION 1 (Re-run simple model for comparison)
# ============================================================================

def simple_model(C, t):
    """Simple model from Iteration 1 for comparison"""
    NO, NO2, O3, O = C
    k1 = k1_photolysis(t)
    
    R1 = k1 * NO2
    R2 = k2 * O * O2
    R3 = k3 * O3 * NO
    
    dNO_dt = R1 - R3 + E_NO
    dNO2_dt = -R1 + R3 + E_NO2
    dO3_dt = R2 - R3
    dO_dt = R1 - R2
    
    return [dNO_dt, dNO2_dt, dO3_dt, dO_dt]

C0_simple = [0.1, 0.05, 0.0, 0.0]
solution_simple = odeint(simple_model, C0_simple, t_span)

NO_simple = solution_simple[:, 0]
NO2_simple = solution_simple[:, 1]
O3_simple = solution_simple[:, 2]

# Comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Ozone comparison
ax1 = axes[0, 0]
ax1.plot(t_clock, O3_simple, 'g--', linewidth=2.5, label='Simple (No VOCs)', alpha=0.7)
ax1.plot(t_clock, O3, 'g-', linewidth=2.5, label='Refined (With VOCs)')
ax1.axvspan(6, 18, alpha=0.2, color='yellow')
ax1.set_xlabel('Time (hours)', fontsize=12)
ax1.set_ylabel('Concentration (ppm)', fontsize=12)
ax1.set_title('Ozone: Model Comparison', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(5, 24)

# NO comparison
ax2 = axes[0, 1]
ax2.plot(t_clock, NO_simple, 'b--', linewidth=2, label='Simple', alpha=0.7)
ax2.plot(t_clock, NO, 'b-', linewidth=2, label='Refined')
ax2.axvspan(6, 18, alpha=0.2, color='yellow')
ax2.set_xlabel('Time (hours)', fontsize=12)
ax2.set_ylabel('Concentration (ppm)', fontsize=12)
ax2.set_title('NO: Model Comparison', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(5, 24)

# NO2 comparison
ax3 = axes[1, 0]
ax3.plot(t_clock, NO2_simple, 'r--', linewidth=2, label='Simple', alpha=0.7)
ax3.plot(t_clock, NO2, 'r-', linewidth=2, label='Refined')
ax3.axvspan(6, 18, alpha=0.2, color='yellow')
ax3.set_xlabel('Time (hours)', fontsize=12)
ax3.set_ylabel('Concentration (ppm)', fontsize=12)
ax3.set_title(r'NO$_2$: Model Comparison', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(5, 24)

# Ozone enhancement factor
ax4 = axes[1, 1]
O3_enhancement = np.where(O3_simple > 0.001, O3 / O3_simple, 1.0)
ax4.plot(t_clock, O3_enhancement, 'purple', linewidth=2.5)
ax4.axhline(y=1, color='gray', linestyle='--', label='No enhancement')
ax4.axvspan(6, 18, alpha=0.2, color='yellow')
ax4.set_xlabel('Time (hours)', fontsize=12)
ax4.set_ylabel('Enhancement Factor', fontsize=12)
ax4.set_title(r'O$_3$ Enhancement due to VOCs', fontsize=13, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(5, 24)
# ax4.set_ylim(0, 5)

plt.suptitle('Iteration 1 vs Iteration 2: Impact of VOCs on Ozone Formation', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
# plt.savefig('iteration_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("ITERATION 2: REFINED MODEL WITH VOCs - RESULTS SUMMARY")
print("="*80)

print(f"\nModel Configuration:")
print(f"  Temperature: {T} K")
print(f"  Species tracked: 11 (NO, NO₂, O₃, O, CO, HCHO, ALK, OLE, OH, HO₂, RO₂)")
print(f"  Reactions included: 15")

print(f"\nInitial Conditions (5:00 AM):")
print(f"  [NO]₀   = {C0[0]:.3f} ppm")
print(f"  [NO₂]₀  = {C0[1]:.3f} ppm")
print(f"  [O₃]₀   = {C0[2]:.3f} ppm")
print(f"  [CO]₀   = {C0[4]:.3f} ppm")
print(f"  [ALK]₀  = {C0[6]:.3f} ppm")
print(f"  [OLE]₀  = {C0[7]:.3f} ppm")

print(f"\nPeak Concentrations:")
idx_O3_max = np.argmax(O3)
idx_OH_max = np.argmax(OH)
idx_HO2_max = np.argmax(HO2)
idx_RO2_max = np.argmax(RO2)

print(f"  Max [O₃]   = {np.max(O3):.4f} ppm at {t_clock[idx_O3_max]:.1f}:00")
print(f"  Max [OH]   = {np.max(OH)*1e6:.4f} × 10⁻⁶ ppm at {t_clock[idx_OH_max]:.1f}:00")
print(f"  Max [HO₂]  = {np.max(HO2)*1e3:.4f} × 10⁻³ ppm at {t_clock[idx_HO2_max]:.1f}:00")
print(f"  Max [RO₂]  = {np.max(RO2)*1e3:.4f} × 10⁻³ ppm at {t_clock[idx_RO2_max]:.1f}:00")
print(f"  Max [NO₂]  = {np.max(NO2):.4f} ppm at {t_clock[np.argmax(NO2)]:.1f}:00")
print(f"  Min [NO]   = {np.min(NO):.4f} ppm at {t_clock[np.argmin(NO)]:.1f}:00")

print(f"\nVOC Consumption:")
ALK_consumed = C0[6] - ALK[-1]
OLE_consumed = C0[7] - OLE[-1]
print(f"  ALK consumed: {ALK_consumed:.4f} ppm ({ALK_consumed/C0[6]*100:.1f}%)")
print(f"  OLE consumed: {OLE_consumed:.4f} ppm ({OLE_consumed/C0[7]*100:.1f}%)")

print(f"\nComparison with Simple Model (Iteration 1):")
O3_simple_max = np.max(O3_simple)
O3_refined_max = np.max(O3)
enhancement = O3_refined_max / O3_simple_max if O3_simple_max > 0 else 0
print(f"  Simple model max O₃:  {O3_simple_max:.4f} ppm")
print(f"  Refined model max O₃: {O3_refined_max:.4f} ppm")
print(f"  Enhancement factor:   {enhancement:.2f}x")
print(f"  Additional O₃ formed: {(O3_refined_max - O3_simple_max)*1000:.2f} ppb")

print(f"\nNOₓ Budget at Peak O₃ Time:")
idx_peak = idx_O3_max
print(f"  [NO]:    {NO[idx_peak]:.4f} ppm")
print(f"  [NO₂]:   {NO2[idx_peak]:.4f} ppm")
print(f"  [NOₓ]:   {(NO[idx_peak] + NO2[idx_peak]):.4f} ppm")
print(f"  NO/NO₂:  {NO[idx_peak]/NO2[idx_peak]:.4f}")

print(f"\nRadical Concentrations at Peak O₃ Time:")
print(f"  [OH]:    {OH[idx_peak]*1e6:.4f} × 10⁻⁶ ppm")
print(f"  [HO₂]:   {HO2[idx_peak]*1e3:.4f} × 10⁻³ ppm")
print(f"  [RO₂]:   {RO2[idx_peak]*1e3:.4f} × 10⁻³ ppm")

print(f"\nNO → NO₂ Conversion Pathways (at peak O₃):")
rate_O3 = k3 * O3[idx_peak] * NO[idx_peak] * 60
rate_HO2 = k12 * HO2[idx_peak] * NO[idx_peak] * 60
rate_RO2 = k13 * RO2[idx_peak] * NO[idx_peak] * 60
total_rate = rate_O3 + rate_HO2 + rate_RO2
if total_rate > 0:
    print(f"  Via O₃:    {rate_O3:.6f} ppm/h ({rate_O3/total_rate*100:.1f}%)")
    print(f"  Via HO₂:   {rate_HO2:.6f} ppm/h ({rate_HO2/total_rate*100:.1f}%)")
    print(f"  Via RO₂:   {rate_RO2:.6f} ppm/h ({rate_RO2/total_rate*100:.1f}%)")
    print(f"  Total:     {total_rate:.6f} ppm/h")

print("\n" + "="*80)