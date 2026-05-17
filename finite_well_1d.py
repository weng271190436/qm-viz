"""
1D Finite Square Well — Griffiths §2.6

Solves for bound state energies and wavefunctions of the finite square well:

    V(x) = -V0   for |x| < a
    V(x) =  0    for |x| > a

Physics:
  - Even solutions: z tan(z) = sqrt(z0^2 - z^2)
  - Odd solutions:  -z cot(z) = sqrt(z0^2 - z^2)
  - where z = ka, z0 = a*sqrt(2mV0)/hbar
  - Wavefunctions: sinusoidal inside well, exponential decay outside
  - Number of bound states depends on z0 (well depth × width)

Usage:
  python finite_well_1d.py
  python finite_well_1d.py --V0 50 --a 1.5
  python finite_well_1d.py --V0 200 --a 1.0   # deeper well → more bound states
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import argparse


# --- Constants (natural units: hbar = m = 1) ---
HBAR = 1.0
M = 1.0


def find_bound_states(V0, a, num_points=1000):
    """Find all bound state energies using the transcendental equations.
    
    For the finite square well centered at origin with depth V0 and half-width a:
      Even states: z tan(z) = sqrt(z0^2 - z^2)
      Odd states:  -z cot(z) = sqrt(z0^2 - z^2)
    where z = ka, z0 = a*sqrt(2mV0)/hbar
    """
    z0 = a * np.sqrt(2 * M * V0) / HBAR
    
    energies = []
    parities = []  # 'even' or 'odd'
    
    # Even states: z*tan(z) = sqrt(z0^2 - z^2)
    # Rewrite as f(z) = z*tan(z) - sqrt(z0^2 - z^2) = 0
    def f_even(z):
        if z >= z0:
            return float('inf')
        return z * np.tan(z) - np.sqrt(z0**2 - z**2)
    
    # Odd states: -z*cot(z) = sqrt(z0^2 - z^2)
    def f_odd(z):
        if z >= z0:
            return float('inf')
        return -z / np.tan(z) - np.sqrt(z0**2 - z**2)
    
    # Search for roots in each interval between singularities of tan/cot
    # Even: tan(z) has singularities at z = (n+1/2)*pi
    # Odd: cot(z) has singularities at z = n*pi
    
    eps = 1e-10
    
    # Even states — search between 0 and z0, avoiding tan singularities
    n = 0
    while True:
        z_lo = n * np.pi + eps if n > 0 else eps
        z_hi = (n + 0.5) * np.pi - eps
        if z_lo >= z0:
            break
        z_hi = min(z_hi, z0 - eps)
        if z_lo >= z_hi:
            n += 1
            continue
        try:
            if f_even(z_lo) * f_even(z_hi) < 0:
                z_root = brentq(f_even, z_lo, z_hi)
                k = z_root / a
                E = (HBAR * k)**2 / (2 * M) - V0
                energies.append(E)
                parities.append('even')
        except (ValueError, ZeroDivisionError):
            pass
        n += 1
    
    # Odd states — search between pi/2 and z0, avoiding cot singularities
    n = 0
    while True:
        z_lo = (n + 0.5) * np.pi + eps
        z_hi = (n + 1) * np.pi - eps
        if z_lo >= z0:
            break
        z_hi = min(z_hi, z0 - eps)
        if z_lo >= z_hi:
            n += 1
            continue
        try:
            if f_odd(z_lo) * f_odd(z_hi) < 0:
                z_root = brentq(f_odd, z_lo, z_hi)
                k = z_root / a
                E = (HBAR * k)**2 / (2 * M) - V0
                energies.append(E)
                parities.append('odd')
        except (ValueError, ZeroDivisionError):
            pass
        n += 1
    
    # Sort by energy
    order = np.argsort(energies)
    energies = [energies[i] for i in order]
    parities = [parities[i] for i in order]
    
    return energies, parities, z0


def compute_wavefunction(E, V0, a, x, parity='even'):
    """Compute the normalized wavefunction for a given energy.
    
    Inside well:  psi = A*cos(kx) (even) or A*sin(kx) (odd)
    Outside well: psi = B*exp(-kappa*|x|)
    """
    k = np.sqrt(2 * M * (E + V0)) / HBAR
    kappa = np.sqrt(-2 * M * E) / HBAR  # E < 0 for bound states
    
    psi = np.zeros_like(x)
    
    inside = np.abs(x) <= a
    outside_right = x > a
    outside_left = x < -a
    
    if parity == 'even':
        psi[inside] = np.cos(k * x[inside])
        B = np.cos(k * a) * np.exp(kappa * a)
        psi[outside_right] = B * np.exp(-kappa * x[outside_right])
        psi[outside_left] = B * np.exp(kappa * x[outside_left])
    else:
        psi[inside] = np.sin(k * x[inside])
        B = np.sin(k * a) * np.exp(kappa * a)
        psi[outside_right] = B * np.exp(-kappa * x[outside_right])
        psi[outside_left] = -B * np.exp(kappa * x[outside_left])
    
    # Normalize
    dx = x[1] - x[0]
    norm = np.sqrt(np.trapezoid(psi**2, x))
    if norm > 0:
        psi /= norm
    
    return psi


def plot_well_and_states(V0, a, energies, parities, z0):
    """Plot the potential well, energy levels, and wavefunctions."""
    
    x = np.linspace(-3 * a, 3 * a, 2000)
    V = np.where(np.abs(x) <= a, -V0, 0)
    
    n_states = len(energies)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f'1D Finite Square Well  (V₀ = {V0:.1f}, a = {a:.1f}, '
                 f'z₀ = {z0:.2f}, {n_states} bound state{"s" if n_states != 1 else ""})',
                 fontsize=14, fontweight='bold')
    
    # --- Left panel: Energy levels in the well ---
    ax1.fill_between(x, V, -V0 * 1.2, alpha=0.15, color='blue', label='Well')
    ax1.plot(x, V, 'b-', linewidth=2)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, max(n_states, 1)))
    
    for i, (E, parity) in enumerate(zip(energies, parities)):
        label = f'n={i}  E={E:.3f}  ({parity})'
        ax1.axhline(y=E, color=colors[i], linestyle='--', alpha=0.7, linewidth=1.5)
        ax1.text(a * 2.5, E + V0 * 0.02, label, fontsize=9, color=colors[i],
                 va='bottom', ha='right')
    
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5, label='V = 0')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Energy', fontsize=12)
    ax1.set_title('Potential & Energy Levels')
    ax1.set_ylim(-V0 * 1.15, V0 * 0.3)
    ax1.set_xlim(-3 * a, 3 * a)
    
    # --- Right panel: Wavefunctions ---
    for i, (E, parity) in enumerate(zip(energies, parities)):
        psi = compute_wavefunction(E, V0, a, x, parity)
        
        # Offset wavefunction by energy level for visualization
        scale = V0 * 0.15  # scale factor for wavefunction display
        ax2.plot(x, psi * scale + E, color=colors[i], linewidth=1.5,
                 label=f'ψ_{i} ({parity})')
        ax2.fill_between(x, E, psi * scale + E, alpha=0.2, color=colors[i])
        ax2.axhline(y=E, color=colors[i], linestyle=':', alpha=0.3)
    
    # Draw the well outline
    ax2.fill_between(x, np.where(np.abs(x) <= a, -V0, 0), -V0 * 1.2,
                     alpha=0.08, color='blue')
    ax2.plot(x, V, 'b-', linewidth=2, alpha=0.5)
    
    # Mark well boundaries
    ax2.axvline(x=-a, color='blue', linestyle=':', alpha=0.3)
    ax2.axvline(x=a, color='blue', linestyle=':', alpha=0.3)
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Energy (with ψ overlay)', fontsize=12)
    ax2.set_title('Wavefunctions')
    ax2.set_ylim(-V0 * 1.15, V0 * 0.3)
    ax2.set_xlim(-3 * a, 3 * a)
    ax2.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('finite_well_1d.png', dpi=150, bbox_inches='tight')
    print(f"Saved: finite_well_1d.png")
    plt.show()


def plot_graphical_solution(V0, a, energies, z0):
    """Plot the graphical solution of the transcendental equations."""
    
    z = np.linspace(0.01, z0 * 1.1, 5000)
    
    # RHS for both: sqrt(z0^2 - z^2) — the circle
    rhs = np.where(z < z0, np.sqrt(np.maximum(z0**2 - z**2, 0)), np.nan)
    
    # Even LHS: z*tan(z)
    lhs_even = z * np.tan(z)
    # Mask out near singularities
    lhs_even[lhs_even < -50] = np.nan
    lhs_even[lhs_even > 50] = np.nan
    
    # Odd LHS: -z*cot(z)  
    lhs_odd = -z / np.tan(z)
    lhs_odd[lhs_odd < -50] = np.nan
    lhs_odd[lhs_odd > 50] = np.nan
    
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f'Graphical Solution: Finite Well (z₀ = {z0:.2f})', fontsize=14)
    
    ax.plot(z, rhs, 'b-', linewidth=2, label=r'$\sqrt{z_0^2 - z^2}$ (circle)')
    ax.plot(z, lhs_even, 'r-', linewidth=1.5, label=r'$z\tan(z)$ (even)')
    ax.plot(z, lhs_odd, 'g-', linewidth=1.5, label=r'$-z\cot(z)$ (odd)')
    
    # Mark intersection points
    for i, E in enumerate(energies):
        k = np.sqrt(2 * M * (E + V0)) / HBAR
        z_val = k * a
        rhs_val = np.sqrt(max(z0**2 - z_val**2, 0))
        ax.plot(z_val, rhs_val, 'ko', markersize=10, zorder=5)
        ax.annotate(f'n={i}\nE={E:.3f}', (z_val, rhs_val),
                    textcoords="offset points", xytext=(10, 10), fontsize=9)
    
    ax.set_xlim(0, z0 * 1.1)
    ax.set_ylim(0, z0 * 1.1)
    ax.set_xlabel('z = ka', fontsize=12)
    ax.set_ylabel('', fontsize=12)
    ax.legend(fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('finite_well_graphical.png', dpi=150, bbox_inches='tight')
    print(f"Saved: finite_well_graphical.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='1D Finite Square Well Solver')
    parser.add_argument('--V0', type=float, default=50.0,
                        help='Well depth (default: 50)')
    parser.add_argument('--a', type=float, default=1.0,
                        help='Half-width of well (default: 1.0)')
    parser.add_argument('--no-graphical', action='store_true',
                        help='Skip the graphical solution plot')
    args = parser.parse_args()
    
    V0 = args.V0
    a = args.a
    
    print(f"1D Finite Square Well")
    print(f"  V0 = {V0:.2f} (depth)")
    print(f"  a  = {a:.2f} (half-width)")
    print(f"  Natural units: ℏ = m = 1")
    print()
    
    energies, parities, z0 = find_bound_states(V0, a)
    
    print(f"  z0 = {z0:.4f}")
    print(f"  Number of bound states: {len(energies)}")
    print()
    
    for i, (E, p) in enumerate(zip(energies, parities)):
        print(f"  n={i}:  E = {E:+.6f}  ({p})")
    
    print()
    
    # Plots
    plot_well_and_states(V0, a, energies, parities, z0)
    
    if not args.no_graphical:
        plot_graphical_solution(V0, a, energies, z0)


if __name__ == '__main__':
    main()
