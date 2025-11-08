"""
Mathematical Modeling of Photochemical Smog Formation
Final Version with Enhanced Plots

Author: Your Name
Date: 2025
"""

import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
from latex import latexify
latexify(columns=2)
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Color scheme
COLORS = {
    'NO': '#1f77b4',      # Blue
    'NO2': '#d62728',     # Red
    'O3': '#2ca02c',      # Green
    'HCHO': '#9467bd',    # Purple
    'CO': '#8c564b',      # Brown
    'OH': '#ff7f0e',      # Orange
    'HO2': '#17becf',     # Cyan
    'RO2': '#e377c2',     # Pink
    'night': 'gray'
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def solar_intensity(t, k_max):
    """
    Calculate solar intensity factor based on time of day.
    
    Parameters:
    -----------
    t : float
        Time in hours (can be > 24 for multi-day simulations)
    k_max : float
        Maximum rate constant at solar noon
        
    Returns:
    --------
    float : Photolysis rate constant at time t
    """
    hour = t % 24
    if 6 <= hour <= 18:
        return k_max * np.sin(np.pi * (hour - 6) / 12)
    return 0.0

def add_day_night_shading(ax, t_max):
    """Add subtle shading to indicate day/night periods on plot."""
    for day in range(int(t_max // 24) + 1):
        # Night before sunrise (0-6h)
        ax.axvspan(day * 24, day * 24 + 6, alpha=0.05, color=COLORS['night'], zorder=0)
        # Night after sunset (18-24h)
        ax.axvspan(day * 24 + 18, (day + 1) * 24, alpha=0.05, color=COLORS['night'], zorder=0)

def format_axis(ax, xlabel='', ylabel='', title='', grid=True):
    """Apply consistent formatting to axis."""
    if xlabel:
        ax.set_xlabel(xlabel, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontweight='bold')
    if title:
        ax.set_title(title, fontweight='bold', pad=10)
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ============================================================================
# MODEL 1: SIMPLE NO-NO2-O3 PHOTOCHEMICAL CYCLE
# ============================================================================

def model1_odes(y, t, params):
    """
    Model 1: Three-species photochemical cycle (NO, NO2, O3).
    
    Parameters:
    -----------
    y : array
        Concentrations [NO, NO2, O3] in ppm
    t : float
        Time in hours
    params : dict
        Dictionary containing rate constants and emissions
        
    Returns:
    --------
    list : Time derivatives [dNO/dt, dNO2/dt, dO3/dt]
    """
    NO, NO2, O3 = y
    
    # Time-dependent photolysis rate
    k1 = solar_intensity(t, params['k1_max'])
    k3 = params['k3']
    E_NO = params['E_NO']
    E_NO2 = params['E_NO2']
    
    # Differential equations
    dNO_dt = k1 * NO2 - k3 * NO * O3 + E_NO
    dNO2_dt = -k1 * NO2 + k3 * NO * O3 + E_NO2
    dO3_dt = k1 * NO2 - k3 * NO * O3
    
    return [dNO_dt, dNO2_dt, dO3_dt]

def run_model1():
    """Run Model 1 simulation and create enhanced plots."""
    print("\n" + "="*70)
    print("MODEL 1: SIMPLE PHOTOCHEMICAL CYCLE (NO-NO₂-O₃)")
    print("="*70)
    
    # Parameters
    params = {
        'k1_max': 0.508,      # min^-1 (maximum NO2 photolysis rate)
        'k3': 20.0,           # ppm^-1 min^-1 (NO + O3 reaction)
        'E_NO': 0.02 / 60,    # ppm/min (converted from ppm/h)
        'E_NO2': 0.01 / 60    # ppm/min
    }
    
    # Initial conditions (ppm)
    y0 = [0.100, 0.050, 0.020]  # NO, NO2, O3
    
    # Time array (0 to 48 hours, 1-minute resolution)
    t = np.linspace(0, 48, 48 * 60 + 1)
    
    # Solve ODEs
    print("Solving ODEs using LSODA algorithm...")
    solution = odeint(model1_odes, y0, t, args=(params,))
    
    NO = solution[:, 0]
    NO2 = solution[:, 1]
    O3 = solution[:, 2]
    
    # Print statistics
    print(f"\nResults:")
    print(f"  Peak O3: {O3.max():.4f} ppm at t = {t[O3.argmax()]:.1f} hours")
    print(f"  Peak NO₂: {NO2.max():.4f} ppm at t = {t[NO2.argmax()]:.1f} hours")
    print(f"  Min NO: {NO.min():.4f} ppm at t = {t[NO.argmin()]:.1f} hours")
    
    # Verify photostationary state
    midday_idx = np.argmin(np.abs(t - 12.0))
    if NO2[midday_idx] > 0:
        phi_sim = (NO[midday_idx] * O3[midday_idx]) / NO2[midday_idx]
        phi_pred = params['k1_max'] / params['k3']
        print(f"\nPhotostationary State (noon, day 1):")
        print(f"  Predicted φ = k₁/k₃ = {phi_pred:.4f} ppm")
        print(f"  Simulated φ = [NO][O₃]/[NO₂] = {phi_sim:.4f} ppm")
        print(f"  Agreement: {(1 - abs(phi_sim - phi_pred)/phi_pred)*100:.1f}%")
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle(r'Model 1: Simple NO-NO$_2$-O$_3$ Photochemical Cycle', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: All concentrations (48 hours)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(t, NO, color=COLORS['NO'], linewidth=2.5, label='NO', alpha=0.8)
    ax1.plot(t, NO2, color=COLORS['NO2'], linewidth=2.5, label=r'NO$_2$', alpha=0.8)
    ax1.plot(t, O3, color=COLORS['O3'], linewidth=2.5, label=r'O$_3$', alpha=0.8)
    add_day_night_shading(ax1, 48)
    format_axis(ax1, 'Time (hours)', 'Concentration (ppm)', 
                'Concentration Profiles (48 hours)')
    ax1.legend(loc='upper right', framealpha=0.95)
    ax1.set_xlim(0, 48)
    ax1.set_ylim(0, max(NO2.max(), O3.max(), NO.max()) * 1.1)
    
    # Plot 2: First 24 hours detail
    ax2 = fig.add_subplot(gs[0, 2])
    mask_24h = t <= 24
    ax2.plot(t[mask_24h], NO[mask_24h], color=COLORS['NO'], linewidth=2.5, label='NO', alpha=0.8)
    ax2.plot(t[mask_24h], NO2[mask_24h], color=COLORS['NO2'], linewidth=2.5, label=r'NO$_2$', alpha=0.8)
    ax2.plot(t[mask_24h], O3[mask_24h], color=COLORS['O3'], linewidth=2.5, label=r'O$_3$', alpha=0.8)
    add_day_night_shading(ax2, 24)
    format_axis(ax2, 'Time (hours)', 'Concentration (ppm)', 
                'First 24 Hours (Detail)')
    ax2.legend(loc='best', framealpha=0.95)
    ax2.set_xlim(0, 24)
    
    # Plot 3: Phase portrait (O3 vs NO2)
    ax3 = fig.add_subplot(gs[1, 0])
    scatter = ax3.scatter(NO2, O3, c=t, cmap='viridis', s=20, alpha=0.6, edgecolors='none')
    format_axis(ax3, r'NO$_2$ Concentration (ppm)', r'O$_3$ Concentration (ppm)', 
                r'Phase Portrait: O$_3$ vs NO$_2$')
    cbar = plt.colorbar(scatter, ax=ax3, label='Time (hours)')
    
    # Add arrows to show direction
    step = len(t) // 10
    for i in range(0, len(t) - step, step):
        ax3.annotate('', xy=(NO2[i+step], O3[i+step]), xytext=(NO2[i], O3[i]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1, alpha=0.5))
    
    # Plot 4: Photolysis rate
    ax4 = fig.add_subplot(gs[1, 1])
    k1_values = np.array([solar_intensity(ti, params['k1_max']) for ti in t])
    ax4.fill_between(t, 0, k1_values, color='orange', alpha=0.3)
    ax4.plot(t, k1_values, color='orange', linewidth=2.5, label=r'k$_1$(t)')
    add_day_night_shading(ax4, 48)
    format_axis(ax4, 'Time (hours)', r'Rate Constant (min$^{-1}$)', 
                r'NO$_2$ Photolysis Rate (Solar Intensity)')
    ax4.legend(loc='upper right', framealpha=0.95)
    ax4.set_xlim(0, 48)
    ax4.set_ylim(0, params['k1_max'] * 1.1)
    
    # Plot 5: Diurnal averages
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Calculate hourly averages
    hours = np.arange(24)
    NO_avg = np.array([NO[t % 24 == h].mean() for h in hours])
    NO2_avg = np.array([NO2[t % 24 == h].mean() for h in hours])
    O3_avg = np.array([O3[t % 24 == h].mean() for h in hours])
    
    ax5.plot(hours, NO_avg, 'o-', color=COLORS['NO'], linewidth=2, 
             markersize=6, label='NO', alpha=0.8)
    ax5.plot(hours, NO2_avg, 's-', color=COLORS['NO2'], linewidth=2, 
             markersize=6, label=r'NO$_2$', alpha=0.8)
    ax5.plot(hours, O3_avg, '^-', color=COLORS['O3'], linewidth=2, 
             markersize=6, label=r'O$_3$', alpha=0.8)
    ax5.axvspan(0, 6, alpha=0.1, color='gray')
    ax5.axvspan(18, 24, alpha=0.1, color='gray')
    format_axis(ax5, 'Hour of Day', 'Avg Concentration (ppm)', 
                'Average Diurnal Pattern')
    ax5.legend(loc='best', framealpha=0.95)
    ax5.set_xlim(0, 24)
    ax5.set_xticks([0, 6, 12, 18, 24])
    
    plt.tight_layout()
    plt.savefig('model1_concentrations.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: model1_concentrations.png")
    plt.show()
    
    return t, solution, params

# ============================================================================
# MODEL 2: EXTENDED CHEMISTRY WITH VOCs
# ============================================================================

def model2_odes(y, t, params):
    """
    Model 2: Eight-species system with VOC chemistry.
    Species: NO, NO2, O3, HCHO, CO, OH, HO2, RO2
    """
    NO, NO2, O3, HCHO, CO, OH, HO2, RO2 = y
    
    # Time-dependent photolysis rates
    k1 = solar_intensity(t, params['k1_max'])
    k20 = solar_intensity(t, params['k20_max'])
    k21 = solar_intensity(t, params['k21_max'])
    
    # Extract parameters
    k3 = params['k3']
    k15 = params['k15']
    k16 = params['k16']
    k19 = params['k19']
    k23 = params['k23']
    k47 = params['k47']
    k50 = params['k50']
    E_NO = params['E_NO']
    E_NO2 = params['E_NO2']
    E_HCHO = params['E_HCHO']
    E_CO = params['E_CO']
    
    # Compute reaction rates
    R1 = k1 * NO2
    R3 = k3 * NO * O3
    R15 = k15 * NO * HO2
    R16 = k16 * NO * RO2
    R19 = k19 * CO * OH
    R20 = k20 * O3
    R21 = k21 * HCHO
    R23 = k23 * HCHO * OH
    R47 = k47 * O3 * OH
    R50 = k50 * HO2 * HO2
    
    # Differential equations
    dNO_dt = R1 - R3 - R15 - R16 + E_NO
    dNO2_dt = -R1 + R3 + R15 + R16 + E_NO2
    dO3_dt = R1 - R3 - R20 - R47
    dHCHO_dt = -R21 - R23 + E_HCHO
    dCO_dt = R21 + R23 - R19 + E_CO
    dOH_dt = 2*R20 - R19 - R23 + R15 - R47
    dHO2_dt = 2*R21 + R23 + R19 - R15 - 2*R50 + R47
    dRO2_dt = 0.5 * R23 - R16
    
    return [dNO_dt, dNO2_dt, dO3_dt, dHCHO_dt, dCO_dt, 
            dOH_dt, dHO2_dt, dRO2_dt]

def run_model2():
    """Run Model 2 simulation and create enhanced plots."""
    print("\n" + "="*70)
    print("MODEL 2: EXTENDED CHEMISTRY WITH VOCs")
    print("="*70)
    
    # Parameters
    params = {
        'k1_max': 0.508,
        'k20_max': 0.0328,
        'k21_max': 0.00284,
        'k3': 20.0,
        'k15': 12000.0,
        'k16': 12000.0,
        'k19': 440.0,
        'k23': 19200.0,
        'k47': 78.0,
        'k50': 3700.0,
        'E_NO': 0.02 / 60,
        'E_NO2': 0.01 / 60,
        'E_HCHO': 0.03 / 60,
        'E_CO': 0.02 / 60
    }
    
    # Initial conditions (ppm)
    y0 = [0.100, 0.050, 0.020, 0.010, 0.500, 1.0e-6, 1.0e-5, 1.0e-5]
    
    # Time array
    t = np.linspace(0, 48, 48 * 60 + 1)
    
    # Solve ODEs
    print("Solving ODEs using LSODA algorithm...")
    solution = odeint(model2_odes, y0, t, args=(params,), rtol=1e-6, atol=1e-9)
    
    NO, NO2, O3 = solution[:, 0], solution[:, 1], solution[:, 2]
    HCHO, CO = solution[:, 3], solution[:, 4]
    OH, HO2, RO2 = solution[:, 5], solution[:, 6], solution[:, 7]
    
    # Print statistics
    print(f"\nResults:")
    print(f"  Peak O3: {O3.max():.4f} ppm at t = {t[O3.argmax()]:.1f} hours")
    print(f"  Peak OH: {OH.max():.2e} ppm at t = {t[OH.argmax()]:.1f} hours")
    print(f"  Peak HO₂: {HO2.max():.2e} ppm at t = {t[HO2.argmax()]:.1f} hours")
    print(f"  Min NO: {NO.min():.4f} ppm")
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    fig.suptitle('Model 2: Extended Chemistry with VOCs (8 Species, 13 Reactions)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: NO
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, NO, color=COLORS['NO'], linewidth=2.5)
    ax1.fill_between(t, 0, NO, color=COLORS['NO'], alpha=0.2)
    add_day_night_shading(ax1, 48)
    format_axis(ax1, 'Time (hours)', 'Concentration (ppm)', 'NO (Nitric Oxide)')
    ax1.set_xlim(0, 48)
    
    # Plot 2: NO2
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, NO2, color=COLORS['NO2'], linewidth=2.5)
    ax2.fill_between(t, 0, NO2, color=COLORS['NO2'], alpha=0.2)
    add_day_night_shading(ax2, 48)
    format_axis(ax2, 'Time (hours)', 'Concentration (ppm)', r'NO$_2$ (Nitrogen Dioxide)')
    ax2.set_xlim(0, 48)
    
    # Plot 3: O3
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, O3, color=COLORS['O3'], linewidth=2.5)
    ax3.fill_between(t, 0, O3, color=COLORS['O3'], alpha=0.2)
    add_day_night_shading(ax3, 48)
    # Highlight peak
    peak_idx = O3.argmax()
    ax3.plot(t[peak_idx], O3[peak_idx], 'r*', markersize=15, 
             label=f'Peak: {O3.max():.3f} ppm')
    format_axis(ax3, 'Time (hours)', 'Concentration (ppm)', r'O$_3$ (Ozone)')
    ax3.legend(loc='upper left')
    ax3.set_xlim(0, 48)
    
    # Plot 4: HCHO
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, HCHO, color=COLORS['HCHO'], linewidth=2.5)
    ax4.fill_between(t, 0, HCHO, color=COLORS['HCHO'], alpha=0.2)
    add_day_night_shading(ax4, 48)
    format_axis(ax4, 'Time (hours)', 'Concentration (ppm)', 'HCHO (Formaldehyde)')
    ax4.set_xlim(0, 48)
    
    # Plot 5: CO
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t, CO, color=COLORS['CO'], linewidth=2.5)
    ax5.fill_between(t, 0, CO, color=COLORS['CO'], alpha=0.2)
    add_day_night_shading(ax5, 48)
    format_axis(ax5, 'Time (hours)', 'Concentration (ppm)', 'CO (Carbon Monoxide)')
    ax5.set_xlim(0, 48)
    
    # Plot 6: OH (log scale)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.semilogy(t, OH, color=COLORS['OH'], linewidth=2.5)
    add_day_night_shading(ax6, 48)
    format_axis(ax6, 'Time (hours)', 'Concentration (ppm, log)', 'OH (Hydroxyl Radical)')
    ax6.set_xlim(0, 48)
    ax6.set_ylim(1e-8, OH.max() * 2)
    
    # Plot 7: HO2 (log scale)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.semilogy(t, HO2, color=COLORS['HO2'], linewidth=2.5)
    add_day_night_shading(ax7, 48)
    format_axis(ax7, 'Time (hours)', 'Concentration (ppm, log)', r'HO$_2$ (Hydroperoxyl Radical)')
    ax7.set_xlim(0, 48)
    ax7.set_ylim(1e-7, HO2.max() * 2)
    
    # Plot 8: RO2 (log scale)
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.semilogy(t, RO2, color=COLORS['RO2'], linewidth=2.5)
    add_day_night_shading(ax8, 48)
    format_axis(ax8, 'Time (hours)', 'Concentration (ppm, log)', r'RO$_2$ (Organic Peroxy Radicals)')
    ax8.set_xlim(0, 48)
    ax8.set_ylim(1e-7, RO2.max() * 2)
    
    # Plot 9: All radicals together
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.semilogy(t, OH, color=COLORS['OH'], linewidth=2, label='OH', alpha=0.8)
    ax9.semilogy(t, HO2, color=COLORS['HO2'], linewidth=2, label=r'HO$_2$', alpha=0.8)
    ax9.semilogy(t, RO2, color=COLORS['RO2'], linewidth=2, label=r'RO$_2$', alpha=0.8)
    add_day_night_shading(ax9, 48)
    format_axis(ax9, 'Time (hours)', 'Concentration (ppm, log)', 
                'All Radicals (Catalysts)')
    ax9.legend(loc='upper right', framealpha=0.95)
    ax9.set_xlim(0, 48)
    ax9.set_ylim(1e-8, max(OH.max(), HO2.max(), RO2.max()) * 2)
    
    plt.tight_layout()
    plt.savefig('model2_concentrations.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: model2_concentrations.png")
    plt.show()
    
    return t, solution, params

# ============================================================================
# COMPARISON PLOTS
# ============================================================================

def create_comparison_plots(t1, sol1, params1, t2, sol2, params2):
    """Create comprehensive comparison between Model 1 and Model 2."""
    print("\n" + "="*70)
    print("CREATING MODEL COMPARISON")
    print("="*70)
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Model Comparison: Simple vs Extended Chemistry', 
                 fontsize=16, fontweight='bold', y=0.96)
    
    # Plot 1: O3 comparison
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t1, sol1[:, 2], color='#2ca02c', linewidth=3, 
             label='Model 1: Simple Cycle (No VOCs)', alpha=0.8, linestyle='--')
    ax1.plot(t2, sol2[:, 2], color='#d62728', linewidth=3, 
             label='Model 2: With VOC Chemistry', alpha=0.8)
    add_day_night_shading(ax1, 48)
    
    # Mark peaks
    peak1_idx = sol1[:, 2].argmax()
    peak2_idx = sol2[:, 2].argmax()
    ax1.plot(t1[peak1_idx], sol1[peak1_idx, 2], 'go', markersize=12, 
             markeredgecolor='darkgreen', markeredgewidth=2)
    ax1.plot(t2[peak2_idx], sol2[peak2_idx, 2], 'ro', markersize=12, 
             markeredgecolor='darkred', markeredgewidth=2)
    
    # Annotations
    ax1.annotate(f'Model 1 Peak\n{sol1[:, 2].max():.3f} ppm\n@{t1[peak1_idx]:.1f}h',
                xy=(t1[peak1_idx], sol1[peak1_idx, 2]),
                xytext=(t1[peak1_idx] + 5, sol1[peak1_idx, 2] + 0.02),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='darkgreen', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor='green', linewidth=2, alpha=0.9))
    
    ax1.annotate(f'Model 2 Peak\n{sol2[:, 2].max():.3f} ppm\n@{t2[peak2_idx]:.1f}h',
                xy=(t2[peak2_idx], sol2[peak2_idx, 2]),
                xytext=(t2[peak2_idx] + 5, sol2[peak2_idx, 2] - 0.03),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='darkred', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor='red', linewidth=2, alpha=0.9))
    
    format_axis(ax1, 'Time (hours)', r'O$_3$ Concentration (ppm)', 
                'Ozone Comparison: VOC Chemistry Doubles Peak Concentration')
    ax1.legend(loc='upper left', framealpha=0.95, fontsize=11)
    ax1.set_xlim(0, 48)
    ax1.set_ylim(0, sol2[:, 2].max() * 1.15)
    
    # Plot 2: NO comparison
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t1, sol1[:, 0], color='#1f77b4', linewidth=3, 
             label='Model 1', alpha=0.7, linestyle='--')
    ax2.plot(t2, sol2[:, 0], color='#ff7f0e', linewidth=3, 
             label='Model 2', alpha=0.8)
    add_day_night_shading(ax2, 48)
    format_axis(ax2, 'Time (hours)', 'NO Concentration (ppm)', 
                'NO: Faster Depletion with VOCs')
    ax2.legend(loc='upper right', framealpha=0.95)
    ax2.set_xlim(0, 48)
    
    # Plot 3: NO2 comparison
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t1, sol1[:, 1], color='#d62728', linewidth=3, 
             label='Model 1', alpha=0.7, linestyle='--')
    ax3.plot(t2, sol2[:, 1], color='#9467bd', linewidth=3, 
             label='Model 2', alpha=0.8)
    add_day_night_shading(ax3, 48)
    format_axis(ax3, 'Time (hours)', r'NO$_2$ Concentration (ppm)', 
                r'NO$_2$: Different Dynamics')
    ax3.legend(loc='upper right', framealpha=0.95)
    ax3.set_xlim(0, 48)
    
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: comparison_plot.png")
    plt.show()
    
    # Print quantitative comparison
    print("\n" + "-"*70)
    print("QUANTITATIVE COMPARISON")
    print("-"*70)
    print(f"{'Metric':<40} {'Model 1':<15} {'Model 2':<15}")
    print("-"*70)
    print(f"{'Peak O₃ (ppm)':<40} {sol1[:, 2].max():<15.4f} {sol2[:, 2].max():<15.4f}")
    print(f"{'Peak O₃ time (hours)':<40} {t1[sol1[:, 2].argmax()]:<15.1f} {t2[sol2[:, 2].argmax()]:<15.1f}")
    print(f"{'O₃ enhancement factor':<40} {'1.0×':<15} {sol2[:, 2].max()/sol1[:, 2].max():<15.2f}×")
    print(f"{'Min NO (ppm)':<40} {sol1[:, 0].min():<15.4f} {sol2[:, 0].min():<15.4f}")
    print(f"{'Peak NO₂ (ppm)':<40} {sol1[:, 1].max():<15.4f} {sol2[:, 1].max():<15.4f}")
    print(f"{'Final O₃ at 48h (ppm)':<40} {sol1[-1, 2]:<15.4f} {sol2[-1, 2]:<15.4f}")
    print("-"*70)

# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

def sensitivity_analysis_emissions():
    """Perform sensitivity analysis on emission rates for Model 2."""
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS: EMISSION RATES")
    print("="*70)
    
    # Base parameters
    base_params = {
        'k1_max': 0.508, 'k20_max': 0.0328, 'k21_max': 0.00284,
        'k3': 20.0, 'k15': 12000.0, 'k16': 12000.0,
        'k19': 440.0, 'k23': 19200.0, 'k47': 78.0, 'k50': 3700.0,
    }
    
    y0 = [0.100, 0.050, 0.020, 0.010, 0.500, 1.0e-6, 1.0e-5, 1.0e-5]
    t = np.linspace(0, 24, 24 * 60 + 1)
    
    # Vary NOx and VOC emissions
    nox_factors = np.linspace(0.2, 2.0, 15)
    voc_factors = np.linspace(0.2, 2.0, 15)
    peak_o3 = np.zeros((len(nox_factors), len(voc_factors)))
    
    base_E_NOx = 0.02 / 60
    base_E_VOC = 0.03 / 60
    
    print("Running simulations across emission space...")
    print(f"  NOx range: {nox_factors[0]:.1f}× to {nox_factors[-1]:.1f}× base")
    print(f"  VOC range: {voc_factors[0]:.1f}× to {voc_factors[-1]:.1f}× base")
    
    total_sims = len(nox_factors) * len(voc_factors)
    count = 0
    
    for i, nox_f in enumerate(nox_factors):
        for j, voc_f in enumerate(voc_factors):
            params = base_params.copy()
            params['E_NO'] = base_E_NOx * nox_f
            params['E_NO2'] = (0.01 / 60) * nox_f
            params['E_HCHO'] = base_E_VOC * voc_f
            params['E_CO'] = 0.02 / 60
            
            sol = odeint(model2_odes, y0, t, args=(params,), rtol=1e-6, atol=1e-9)
            peak_o3[i, j] = sol[:, 2].max()
            
            count += 1
            if count % 50 == 0:
                print(f"  Progress: {count}/{total_sims} ({100*count/total_sims:.0f}%)")
    
    print("✓ Simulations complete")
    
    # Create contour plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    VOC_grid, NOx_grid = np.meshgrid(voc_factors, nox_factors)
    
    # Filled contours
    levels = np.linspace(peak_o3.min(), peak_o3.max(), 20)
    contour = ax.contourf(VOC_grid, NOx_grid, peak_o3, levels=levels, 
                          cmap='RdYlGn_r', alpha=0.9)
    
    # Contour lines
    contour_lines = ax.contour(VOC_grid, NOx_grid, peak_o3, levels=10, 
                               colors='black', linewidths=1, alpha=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=9, fmt='%.3f')
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax, pad=0.02)
    cbar.set_label(r'Peak O$_3$ Concentration (ppm)', fontsize=13, fontweight='bold')
    
    # Mark base case
    ax.plot(1.0, 1.0, 'w*', markersize=25, markeredgecolor='black', 
            markeredgewidth=3, label='Base Case', zorder=10)
    
    # Add regime labels
    ax.text(0.4, 1.6, r'VOC-Limited (Low O$_3$)', fontsize=12, 
            ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='black', linewidth=2, alpha=0.8))

    ax.text(1.6, 0.4, r'NO$_x$-Limited (Moderate O$_3$)', fontsize=12, 
            ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='black', linewidth=2, alpha=0.8))
    
    ax.text(1.6, 1.6, r'Both High (Highest O$_3$)', fontsize=12, 
            ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='black', linewidth=2, alpha=0.8))
    
    format_axis(ax, 'VOC Emission Factor (relative to base)', 
                r'NO$_x$ Emission Factor (relative to base)', 
                r'Sensitivity of Peak O$_3$ to NO$_x$ and VOC Emissions')
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95)
    ax.set_xlim(voc_factors[0], voc_factors[-1])
    ax.set_ylim(nox_factors[0], nox_factors[-1])
    
    plt.tight_layout()
    plt.savefig('sensitivity_emissions.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: sensitivity_emissions.png")
    plt.show()
    
    # Analysis
    print("\n" + "-"*70)
    print("REGIME ANALYSIS")
    print("-"*70)
    voc_limited = peak_o3[-3, 2]   # High NOx, Low VOC
    nox_limited = peak_o3[2, -3]   # Low NOx, High VOC
    balanced = peak_o3[7, 7]       # Middle
    both_high = peak_o3[-2, -2]    # Both high
    
    print(f"VOC-limited (High NOₓ, Low VOC):  {voc_limited:.4f} ppm")
    print(f"NOₓ-limited (Low NOₓ, High VOC):  {nox_limited:.4f} ppm")
    print(f"Balanced (Moderate both):         {balanced:.4f} ppm")
    print(f"Both high:                        {both_high:.4f} ppm")
    print("-"*70)
    print("\nKey Insight:")
    print("  • Nonlinear response: Peak O₃ depends on NOₓ/VOC ratio")
    print("  • In VOC-limited areas, reducing NOₓ may INCREASE O₃!")
    print("  • In NOₓ-limited areas, reducing NOₓ DECREASES O₃")
    print("  • Policy must be location-specific")

def sensitivity_analysis_temperature():
    """Perform sensitivity analysis on temperature for Model 2."""
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS: TEMPERATURE")
    print("="*70)
    
    # Temperature range (Kelvin)
    temperatures = np.linspace(278, 308, 10)  # 5°C to 35°C
    peak_o3_values = []
    
    base_params = {
        'k1_max': 0.508, 'k20_max': 0.0328, 'k21_max': 0.00284,
        'k15': 12000.0, 'k16': 12000.0, 'k19': 440.0,
        'k23': 19200.0, 'k50': 3700.0,
        'E_NO': 0.02 / 60, 'E_NO2': 0.01 / 60,
        'E_HCHO': 0.03 / 60, 'E_CO': 0.02 / 60
    }
    
    y0 = [0.100, 0.050, 0.020, 0.010, 0.500, 1.0e-6, 1.0e-5, 1.0e-5]
    t = np.linspace(0, 24, 24 * 60 + 1)
    
    print("Running temperature sensitivity simulations...")
    for T in temperatures:
        params = base_params.copy()
        # Temperature-dependent rate constants
        params['k3'] = 3100 * np.exp(-1450 / T)
        params['k47'] = 2220 * np.exp(-1000 / T)
        
        sol = odeint(model2_odes, y0, t, args=(params,), rtol=1e-6, atol=1e-9)
        peak_o3_values.append(sol[:, 2].max())
    
    print("✓ Simulations complete")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    temps_celsius = temperatures - 273.15
    
    # Plot data points
    ax.plot(temps_celsius, peak_o3_values, 'ro-', linewidth=3, 
            markersize=10, label='Simulation Results', markeredgecolor='darkred',
            markeredgewidth=2)
    
    # Add linear fit
    z = np.polyfit(temps_celsius, peak_o3_values, 1)
    p = np.poly1d(z)
    fit_line = p(temps_celsius)
    ax.plot(temps_celsius, fit_line, 'b--', linewidth=2.5, alpha=0.7,
            label=fr'Linear Fit: slope = {z[0]:.5f} ppm/$^circ$C')
    
    # Shading between data and fit
    ax.fill_between(temps_celsius, peak_o3_values, fit_line, 
                    alpha=0.2, color='gray')
    
    # Highlight specific points
    ax.plot(temps_celsius[0], peak_o3_values[0], 'bs', markersize=12,
            label=fr'Cold: {peak_o3_values[0]:.4f} ppm @ {temps_celsius[0]:.0f}$^circ$C')
    ax.plot(temps_celsius[-1], peak_o3_values[-1], 'rs', markersize=12,
            label=fr'Hot: {peak_o3_values[-1]:.4f} ppm @ {temps_celsius[-1]:.0f}$^circ$C')

    format_axis(ax, r'Temperature ($^circ$C)', r'Peak O$_3$ Concentration (ppm)', 
                r'Temperature Sensitivity: Higher Temperature $\to$ More Ozone')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax.set_xlim(temps_celsius[0] - 1, temps_celsius[-1] + 1)
    ax.set_ylim(min(peak_o3_values) * 0.95, max(peak_o3_values) * 1.05)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('sensitivity_temperature.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: sensitivity_temperature.png")
    plt.show()
    
    # Statistics
    delta_o3 = peak_o3_values[-1] - peak_o3_values[0]
    delta_T = temperatures[-1] - temperatures[0]
    percent_per_10K = (delta_o3 / peak_o3_values[0]) * (10 / delta_T) * 100
    
    print("\n" + "-"*70)
    print("TEMPERATURE SENSITIVITY STATISTICS")
    print("-"*70)
    print(f"Temperature range:        {temps_celsius[0]:.1f}°C to {temps_celsius[-1]:.1f}°C")
    print(f"O₃ at lowest temp:        {peak_o3_values[0]:.4f} ppm")
    print(f"O₃ at highest temp:       {peak_o3_values[-1]:.4f} ppm")
    print(f"Absolute change:          {delta_o3:.4f} ppm")
    print(f"Relative change:          {(delta_o3/peak_o3_values[0])*100:.1f}%")
    print(f"Change per 10 K:          {percent_per_10K:.1f}%")
    print(f"Ratio (hot/cold):         {peak_o3_values[-1]/peak_o3_values[0]:.2f}")
    print("-"*70)
    print("\nImplication: Heat waves significantly worsen ozone pollution!")

# ============================================================================
# DETAILED PUBLICATION-QUALITY FIGURE
# ============================================================================

def create_publication_figure():
    """Create a single comprehensive figure for publication."""
    print("\n" + "="*70)
    print("CREATING PUBLICATION-QUALITY FIGURE")
    print("="*70)
    
    # Run both models
    print("Running Model 1...")
    params1 = {
        'k1_max': 0.508, 'k3': 20.0,
        'E_NO': 0.02 / 60, 'E_NO2': 0.01 / 60
    }
    y0_1 = [0.100, 0.050, 0.020]
    
    print("Running Model 2...")
    params2 = {
        'k1_max': 0.508, 'k20_max': 0.0328, 'k21_max': 0.00284,
        'k3': 20.0, 'k15': 12000.0, 'k16': 12000.0,
        'k19': 440.0, 'k23': 19200.0, 'k47': 78.0, 'k50': 3700.0,
        'E_NO': 0.02 / 60, 'E_NO2': 0.01 / 60,
        'E_HCHO': 0.03 / 60, 'E_CO': 0.02 / 60
    }
    y0_2 = [0.100, 0.050, 0.020, 0.010, 0.500, 1.0e-6, 1.0e-5, 1.0e-5]
    
    t = np.linspace(0, 48, 48 * 60 + 1)
    
    sol1 = odeint(model1_odes, y0_1, t, args=(params1,))
    sol2 = odeint(model2_odes, y0_2, t, args=(params2,), rtol=1e-6, atol=1e-9)
    
    print("✓ Simulations complete")
    print("Creating figure...")
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35,
                  left=0.08, right=0.95, top=0.93, bottom=0.07)
    
    fig.suptitle('Mathematical Modeling of Photochemical Smog Formation', 
                 fontsize=18, fontweight='bold')
    
    # Panel A: O3 comparison (large, top)
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.plot(t, sol1[:, 2], color='#2ca02c', linewidth=3.5, 
             label='Model 1: Simple Cycle', alpha=0.8, linestyle='--')
    ax_a.plot(t, sol2[:, 2], color='#d62728', linewidth=3.5, 
             label='Model 2: With VOC Chemistry', alpha=0.9)
    add_day_night_shading(ax_a, 48)
    
    # Annotate enhancement
    enhancement = sol2[:, 2].max() / sol1[:, 2].max()
    ax_a.text(0.98, 0.95, f'Enhancement Factor: {enhancement:.2f} x', 
             transform=ax_a.transAxes, fontsize=13, fontweight='bold',
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                      edgecolor='black', linewidth=2, alpha=0.8))
    
    format_axis(ax_a, 'Time (hours)', r'O$_3$ Concentration (ppm)', 
                '(a) Ozone: VOC Chemistry Enables Net Production')
    ax_a.legend(loc='upper left', fontsize=12, framealpha=0.95)
    ax_a.set_xlim(0, 48)
    ax_a.text(-0.08, 1.05, 'A', transform=ax_a.transAxes, 
             fontsize=20, fontweight='bold', va='top')
    
    # Panel B: NO comparison
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.plot(t, sol1[:, 0], color='#1f77b4', linewidth=2.5, 
             label='Model 1', alpha=0.7, linestyle='--')
    ax_b.plot(t, sol2[:, 0], color='#ff7f0e', linewidth=2.5, 
             label='Model 2', alpha=0.9)
    add_day_night_shading(ax_b, 48)
    format_axis(ax_b, 'Time (hours)', 'NO (ppm)', 
                '(b) NO: Faster Depletion')
    ax_b.legend(fontsize=10)
    ax_b.set_xlim(0, 48)
    ax_b.text(-0.15, 1.05, 'B', transform=ax_b.transAxes, 
             fontsize=20, fontweight='bold', va='top')
    
    # Panel C: NO2 comparison
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.plot(t, sol1[:, 1], color='#d62728', linewidth=2.5, 
             label='Model 1', alpha=0.7, linestyle='--')
    ax_c.plot(t, sol2[:, 1], color='#9467bd', linewidth=2.5, 
             label='Model 2', alpha=0.9)
    add_day_night_shading(ax_c, 48)
    format_axis(ax_c, 'Time (hours)', r'NO$_2$ (ppm)', 
                r'(c) NO$_2$: Modified Dynamics')
    ax_c.legend(fontsize=10)
    ax_c.set_xlim(0, 48)
    ax_c.text(-0.15, 1.05, 'C', transform=ax_c.transAxes, 
             fontsize=20, fontweight='bold', va='top')
    
    # Panel D: Radicals (Model 2 only)
    ax_d = fig.add_subplot(gs[1, 2])
    ax_d.semilogy(t, sol2[:, 5], color=COLORS['OH'], linewidth=2, 
                 label='OH', alpha=0.9)
    ax_d.semilogy(t, sol2[:, 6], color=COLORS['HO2'], linewidth=2, 
                 label=r'HO$_2$', alpha=0.9)
    ax_d.semilogy(t, sol2[:, 7], color=COLORS['RO2'], linewidth=2, 
                 label=r'RO$_2$', alpha=0.9)
    add_day_night_shading(ax_d, 48)
    format_axis(ax_d, 'Time (hours)', 'Concentration (ppm, log)', 
                '(d) Radicals (Catalysts)')
    ax_d.legend(fontsize=10)
    ax_d.set_xlim(0, 48)
    ax_d.text(-0.15, 1.05, 'D', transform=ax_d.transAxes, 
             fontsize=20, fontweight='bold', va='top')
    
    # Panel E: Key metrics comparison
    ax_e = fig.add_subplot(gs[2, :2])

    metrics = [r'Peak O$_3$ (ppm)', r'Peak Time (hours)', r'Min NO (ppm)', 
               r'Day 2 O$_3$ (ppm)']

    day2_idx_1 = np.argmin(np.abs(t - 36))  # Noon on day 2
    day2_idx_2 = np.argmin(np.abs(t - 36))
    
    model1_vals = [sol1[:, 2].max(), 
                   t[sol1[:, 2].argmax()], 
                   sol1[:, 0].min(),
                   sol1[day2_idx_1, 2]]
    model2_vals = [sol2[:, 2].max(), 
                   t[sol2[:, 2].argmax()], 
                   sol2[:, 0].min(),
                   sol2[day2_idx_2, 2]]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax_e.bar(x - width/2, model1_vals, width, label='Model 1', 
                    color='lightblue', edgecolor='#1f77b4', linewidth=2.5)
    bars2 = ax_e.bar(x + width/2, model2_vals, width, label='Model 2', 
                    color='lightcoral', edgecolor='#d62728', linewidth=2.5)
    
    format_axis(ax_e, '', 'Value', '(e) Quantitative Comparison', grid=False)
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(metrics, fontsize=11)
    ax_e.legend(fontsize=11, loc='upper left')
    ax_e.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_e.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
    
    ax_e.text(-0.08, 1.05, 'E', transform=ax_e.transAxes, 
             fontsize=20, fontweight='bold', va='top')
    
    # Panel F: Mechanism diagram (text-based)
    ax_f = fig.add_subplot(gs[2, 2])
    ax_f.axis('off')
    
    mechanism_text = r"""
    Key Mechanism:
    
    Model 1 (No net O$_3$):
    1. NO$_2$ + h$\nu$ $\to$ NO + O$_3$
    2. NO + O$_3$ $\to$ NO$_2$ + O$_2$
    Result: Cycling only
    
    Model 2 (Net O$_3$):
    1. CO + OH $\to$ HO$_2$ + CO$_2$
    2. HO$_2$ + NO $\to$ NO$_2$ + OH
    3. NO$_2$ + h$\nu$ $\to$ NO + O$_3$
    Result: Net O$_3$ production!
    
    Radicals convert NO $\to$ NO$_2$
    without consuming O$_3$
    """
    
    ax_f.text(0.05, 0.95, mechanism_text, transform=ax_f.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', 
                      edgecolor='black', linewidth=2, alpha=0.9))
    ax_f.text(-0.08, 1.05, 'F', transform=ax_f.transAxes, 
             fontsize=20, fontweight='bold', va='top')
    
    plt.savefig('publication_figure.png', dpi=300, bbox_inches='tight')
    print("\n✓ Publication figure saved: publication_figure.png")
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print(" " * 10 + "PHOTOCHEMICAL SMOG MODELING PROJECT")
    print(" " * 15 + "From Simple to Complex Systems")
    print("="*70)
    print("\nThis program simulates atmospheric photochemical reactions")
    print("to understand ozone formation in urban air pollution.")
    print("\nTwo models will be executed:")
    print("  1. Simple NO-NO₂-O₃ cycle (photostationary state)")
    print("  2. Extended model with VOC chemistry (net ozone production)")
    print("="*70)
    
    # Run Model 1
    t1, sol1, params1 = run_model1()
    
    # Run Model 2
    t2, sol2, params2 = run_model2()
    
    # Create comparison plots
    create_comparison_plots(t1, sol1, params1, t2, sol2, params2)
    
    # Create publication figure
    create_publication_figure()
    
    # Sensitivity analyses
    sensitivity_analysis_emissions()
    sensitivity_analysis_temperature()
    
    # Final summary
    print("\n" + "="*70)
    print(" " * 20 + "ALL SIMULATIONS COMPLETED!")
    print("="*70)
    print("\nGenerated Files:")
    print("  ✓ model1_concentrations.png      - Model 1 detailed results")
    print("  ✓ model2_concentrations.png      - Model 2 detailed results")
    print("  ✓ comparison_plot.png             - Direct model comparison")
    print("  ✓ publication_figure.png          - Comprehensive summary figure")
    print("  ✓ sensitivity_emissions.png       - NOx/VOC sensitivity analysis")
    print("  ✓ sensitivity_temperature.png     - Temperature sensitivity")
    print("\nKey Findings:")
    print(f"  • Model 1 peak O₃: {sol1[:, 2].max():.3f} ppm (no net production)")
    print(f"  • Model 2 peak O₃: {sol2[:, 2].max():.3f} ppm (net production!)")
    print(f"  • Enhancement factor: {sol2[:, 2].max()/sol1[:, 2].max():.2f}×")
    print(f"  • VOC chemistry is ESSENTIAL for realistic ozone levels")
    print("\nThank you for using this modeling framework!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()