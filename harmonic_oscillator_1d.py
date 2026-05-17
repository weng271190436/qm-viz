"""
1D Quantum Harmonic Oscillator — Griffiths §2.3

Solves the quantum harmonic oscillator:

    V(x) = ½mω²x²

Physics:
  - Eₙ = ℏω(n + ½)  — evenly spaced energy levels
  - ψₙ(x) = (mω/πℏ)^(1/4) · 1/√(2ⁿn!) · Hₙ(ξ) · e^(-ξ²/2)
  - where ξ = √(mω/ℏ) · x and Hₙ are Hermite polynomials
  - Ladder operators: a± = (∓ℏ d/dx + mωx) / √(2ℏmω)

Usage:
  python harmonic_oscillator_1d.py
  python harmonic_oscillator_1d.py --omega 2.0 --nmax 10
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from scipy.stats import norm
import argparse


# --- Constants (natural units: hbar = m = 1) ---
HBAR = 1.0
M = 1.0


def psi_n(n, x, omega):
    """Compute the nth eigenstate of the harmonic oscillator.
    
    ψₙ(x) = (mω/πℏ)^(1/4) · 1/√(2ⁿn!) · Hₙ(ξ) · e^(-ξ²/2)
    where ξ = √(mω/ℏ) · x
    """
    xi = np.sqrt(M * omega / HBAR) * x
    prefactor = (M * omega / (np.pi * HBAR)) ** 0.25
    # Use log-space to avoid overflow for large n
    log_norm = -0.5 * (n * np.log(2) + sum(np.log(k) for k in range(1, n + 1)))
    normalization = np.exp(log_norm)
    Hn = hermite(n)  # Hermite polynomial
    return prefactor * normalization * Hn(xi) * np.exp(-xi**2 / 2)


def energy_n(n, omega):
    """Eₙ = ℏω(n + ½)"""
    return HBAR * omega * (n + 0.5)


def classical_turning_point(n, omega):
    """Classical turning point: V(x_tp) = Eₙ → x_tp = √(2Eₙ/(mω²))"""
    E = energy_n(n, omega)
    return np.sqrt(2 * E / (M * omega**2))


def plot_oscillator(omega, nmax):
    """Plot potential, energy levels, and wavefunctions."""
    
    # Spatial range based on highest state's classical turning point
    x_max = classical_turning_point(nmax - 1, omega) * 1.8
    x = np.linspace(-x_max, x_max, 2000)
    V = 0.5 * M * omega**2 * x**2
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(f'1D Quantum Harmonic Oscillator  (ω = {omega:.1f}, '
                 f'ℏ = m = 1, {nmax} states shown)',
                 fontsize=14, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, nmax))
    E_max = energy_n(nmax - 1, omega) * 1.3
    
    # --- Left panel: Energy levels in potential ---
    ax1.plot(x, V, 'b-', linewidth=2, label='V(x) = ½mω²x²')
    
    for n in range(nmax):
        E = energy_n(n, omega)
        x_tp = classical_turning_point(n, omega)
        ax1.hlines(E, -x_tp, x_tp, colors=colors[n], linewidth=2, alpha=0.8)
        ax1.text(x_max * 0.85, E, f'n={n}  E={E:.2f}', fontsize=8,
                 va='center', color=colors[n])
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Energy', fontsize=12)
    ax1.set_title('Potential & Energy Levels')
    ax1.set_ylim(-0.5, E_max)
    ax1.set_xlim(-x_max, x_max)
    ax1.legend(loc='upper left', fontsize=9)
    
    # Annotate equal spacing
    if nmax >= 2:
        E0 = energy_n(0, omega)
        E1 = energy_n(1, omega)
        ax1.annotate('', xy=(x_max * 0.6, E1), xytext=(x_max * 0.6, E0),
                     arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax1.text(x_max * 0.65, (E0 + E1) / 2, f'ℏω = {omega:.1f}',
                 fontsize=9, color='red', va='center')
    
    # --- Middle panel: ψₙ(x) wavefunctions ---
    ax2.plot(x, V, 'b-', linewidth=1.5, alpha=0.3)
    
    for n in range(nmax):
        E = energy_n(n, omega)
        psi = psi_n(n, x, omega)
        scale = 0.4 * HBAR * omega  # scale wavefunction for display
        ax2.plot(x, psi * scale + E, color=colors[n], linewidth=1.2)
        ax2.fill_between(x, E, psi * scale + E, alpha=0.15, color=colors[n])
        ax2.axhline(y=E, color=colors[n], linestyle=':', alpha=0.2)
        
        # Mark classical turning points
        x_tp = classical_turning_point(n, omega)
        ax2.plot([-x_tp, x_tp], [E, E], 'o', color=colors[n],
                 markersize=3, alpha=0.5)
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Energy (with ψ overlay)', fontsize=12)
    ax2.set_title('Wavefunctions ψₙ(x)')
    ax2.set_ylim(-0.5, E_max)
    ax2.set_xlim(-x_max, x_max)
    
    # --- Right panel: |ψₙ(x)|² probability density ---
    ax3.plot(x, V, 'b-', linewidth=1.5, alpha=0.3)
    
    for n in range(nmax):
        E = energy_n(n, omega)
        psi = psi_n(n, x, omega)
        prob = psi**2
        scale = 0.4 * HBAR * omega
        ax3.plot(x, prob * scale + E, color=colors[n], linewidth=1.2)
        ax3.fill_between(x, E, prob * scale + E, alpha=0.2, color=colors[n])
        ax3.axhline(y=E, color=colors[n], linestyle=':', alpha=0.2)
        
        # Mark classical turning points
        x_tp = classical_turning_point(n, omega)
        ax3.axvline(x=-x_tp, ymin=0, ymax=0.02, color=colors[n], alpha=0.3)
        ax3.axvline(x=x_tp, ymin=0, ymax=0.02, color=colors[n], alpha=0.3)
    
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('Energy (with |ψ|² overlay)', fontsize=12)
    ax3.set_title('Probability Density |ψₙ(x)|²')
    ax3.set_ylim(-0.5, E_max)
    ax3.set_xlim(-x_max, x_max)
    
    plt.tight_layout()
    plt.savefig('harmonic_oscillator_1d.png', dpi=150, bbox_inches='tight')
    print(f"Saved: harmonic_oscillator_1d.png")
    plt.show()


def plot_classical_comparison(omega, n_high):
    """Compare high-n quantum |ψ|² with classical probability distribution.
    
    Classical probability: P(x) ∝ 1/v(x) = 1/√(x_tp² - x²)
    At high n, quantum |ψ|² oscillates around the classical distribution.
    This is the correspondence principle in action.
    """
    x_tp = classical_turning_point(n_high, omega)
    x = np.linspace(-x_tp * 1.3, x_tp * 1.3, 5000)
    
    # Quantum
    psi = psi_n(n_high, x, omega)
    prob_quantum = psi**2
    
    # Classical: P(x) = 1/(π√(x_tp² - x²))
    x_cl = np.linspace(-x_tp * 0.999, x_tp * 0.999, 5000)
    prob_classical = 1.0 / (np.pi * np.sqrt(x_tp**2 - x_cl**2))
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, prob_quantum, 'b-', linewidth=0.8, alpha=0.8, label=f'Quantum |ψ_{n_high}|²')
    ax.plot(x_cl, prob_classical, 'r-', linewidth=2, label='Classical P(x)')
    ax.axvline(x=-x_tp, color='gray', linestyle='--', alpha=0.5, label=f'Classical turning points')
    ax.axvline(x=x_tp, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)
    ax.set_title(f'Correspondence Principle: n={n_high} (ω={omega:.1f})', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(-x_tp * 1.3, x_tp * 1.3)
    
    plt.tight_layout()
    plt.savefig('harmonic_oscillator_classical.png', dpi=150, bbox_inches='tight')
    print(f"Saved: harmonic_oscillator_classical.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='1D Quantum Harmonic Oscillator')
    parser.add_argument('--omega', type=float, default=1.0,
                        help='Angular frequency ω (default: 1.0)')
    parser.add_argument('--nmax', type=int, default=8,
                        help='Number of states to show (default: 8)')
    parser.add_argument('--no-classical', action='store_true',
                        help='Skip the classical comparison plot')
    args = parser.parse_args()
    
    omega = args.omega
    nmax = args.nmax
    
    print(f"1D Quantum Harmonic Oscillator")
    print(f"  ω = {omega:.2f}")
    print(f"  Natural units: ℏ = m = 1")
    print()
    
    print(f"  Energy levels: Eₙ = ℏω(n + ½)")
    print(f"  Spacing: ΔE = ℏω = {HBAR * omega:.2f}")
    print()
    
    for n in range(nmax):
        E = energy_n(n, omega)
        x_tp = classical_turning_point(n, omega)
        print(f"  n={n}:  E = {E:.4f}  x_tp = ±{x_tp:.4f}")
    
    print()
    
    plot_oscillator(omega, nmax)
    
    if not args.no_classical:
        n_classical = max(30, nmax * 3)
        plot_classical_comparison(omega, n_classical)


if __name__ == '__main__':
    main()
