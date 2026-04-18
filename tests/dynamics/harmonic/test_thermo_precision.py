import numpy as np
import pandas as pd
import pytest
import mpmath
from mbe_automation.dynamics.harmonic import thermodynamics
from phonopy.physical_units import get_physical_units

def get_ref_thermo(freqs_THz, temperatures_K, dps=50):
    """
    Compute reference thermodynamic properties using mpmath for high precision.
    Quantities are summed over all modes (per unit cell).
    """
    mpmath.mp.dps = dps
    
    units = get_physical_units()
    # Use mpmath versions of constants
    # THz -> eV
    thz_to_ev = mpmath.mpf(str(units.THzToEv))
    kb = mpmath.mpf(str(units.KB))
    ev_to_kj_mol = mpmath.mpf(str(units.EvTokJmol))
    ev_to_j_mol_k = ev_to_kj_mol * 1000
    
    results = []
    
    for T_val in temperatures_K:
        T = mpmath.mpf(str(T_val))
        
        e_vib_total = mpmath.mpf(0)
        s_vib_total = mpmath.mpf(0)
        cv_vib_total = mpmath.mpf(0)
        
        for f_val in freqs_THz:
            f = mpmath.mpf(str(f_val))
            if f <= 1e-12: # Skip very small frequencies like in the code
                continue
                
            hw = f * thz_to_ev
            
            if T < 1e-12:
                # Limit T -> 0
                e_vib = hw / 2
                s_vib = 0
                cv_vib = 0
            else:
                x = hw / (kb * T)
                # Bose-Einstein factor
                # expm1(x) = exp(x) - 1
                bose = 1 / mpmath.expm1(x)
                
                e_vib = hw * (mpmath.mpf(0.5) + bose)
                # s = kb * (x / (exp(x)-1) - ln(1 - exp(-x)))
                s_vib = kb * (x * bose - mpmath.log1p(-mpmath.exp(-x)))
                # cv = kb * x^2 * exp(x) / (exp(x)-1)^2
                cv_vib = kb * x**2 * mpmath.exp(x) / (mpmath.expm1(x)**2)
                
            e_vib_total += e_vib
            s_vib_total += s_vib
            cv_vib_total += cv_vib
            
        # Convert to chemical units (per unit cell — sum over all modes)
        e_vib_kj = e_vib_total * ev_to_kj_mol
        s_vib_j = s_vib_total * ev_to_j_mol_k
        cv_vib_j = cv_vib_total * ev_to_j_mol_k
        f_vib_kj = e_vib_kj - T_val * s_vib_j / 1000
        
        results.append({
            "T (K)": T_val,
            "E_vib_crystal (kJ∕mol∕unit cell)": float(e_vib_kj),
            "S_vib_crystal (J∕K∕mol∕unit cell)": float(s_vib_j),
            "C_V_vib_crystal (J∕K∕mol∕unit cell)": float(cv_vib_j),
            "F_vib_crystal (kJ∕mol∕unit cell)": float(f_vib_kj)
        })
        
    return pd.DataFrame(results)

def format_comparison_table(df_test, df_ref, label):
    """
    Format a comparison table for the report.
    """
    rows = []
    rows.append(f"### Stress Test: {label}")
    rows.append("| T (K) | Property | Computed | Reference | Rel. Error |")
    rows.append("|-------|----------|----------|-----------|------------|")
    
    props = [
        "E_vib_crystal (kJ∕mol∕unit cell)",
        "S_vib_crystal (J∕K∕mol∕unit cell)",
        "C_V_vib_crystal (J∕K∕mol∕unit cell)",
        "F_vib_crystal (kJ∕mol∕unit cell)"
    ]
    
    max_rel_err = 0
    
    for i in range(len(df_test)):
        T = df_test.iloc[i]["T (K)"]
        for p in props:
            val = df_test.iloc[i][p]
            ref = df_ref.iloc[i][p]
            
            if abs(ref) > 1e-20:
                rel_err = abs(val - ref) / abs(ref)
            else:
                rel_err = abs(val - ref)
                
            max_rel_err = max(max_rel_err, rel_err)
            
            rows.append(f"| {T:.1e} | {p.split()[0]} | {val:.8e} | {ref:.8e} | {rel_err:.2e} |")
            
    return "\n".join(rows), max_rel_err

def test_thermo_precision():
    """
    Stress test thermodynamics.run with synthetic data at extreme ranges.
    Uses 2D input with explicit weights.
    """
    print("\nStarting thermodynamic precision stress tests (synthetic data)...")
    
    # Synthetic data: a few representative frequencies
    freqs = np.array([[0.1, 1.0, 10.0, 50.0]]) # (1, 4) — single q-point
    weights = np.array([1.0]) # single q-point weight
    
    # Extreme temperature range
    temps = np.array([1e-12, 1e-9, 1e-6, 1e-3, 300.0, 1e6]) # K
    
    df_test = thermodynamics.run(freqs, temps, weights)
    df_ref = get_ref_thermo(freqs[0], temps)
    
    table, max_err = format_comparison_table(df_test, df_ref, "Synthetic Stress Test (Absolute Zero to 10^6 K)")
    print("\n" + table)
    
    # Save table to a temporary file for the user to see easily
    with open("/tmp/thermo_stress_test_report.md", "w") as f:
        f.write("# Thermodynamics Stress Test Report\n\n")
        f.write(table)
        f.write(f"\n\n**Maximum Relative Error: {max_err:.2e}**")

    assert max_err < 1e-10
    print(f"\nStress test PASSED (Max relative error: {max_err:.2e})")


def test_thermo_gamma_point_default_weights():
    """
    Test the gamma-point convenience path (weights=None, 1D freqs).
    Verifies that results match the explicit 2D path.
    """
    freqs_1d = np.array([0.1, 1.0, 10.0, 50.0])  # (4,)
    temps = np.array([0.0, 100.0, 300.0, 500.0])

    # Call with default weights=None (gamma-point path)
    df_gamma = thermodynamics.run(freqs_THz=freqs_1d, temperatures_K=temps)

    # Call with explicit 2D input
    df_explicit = thermodynamics.run(
        freqs_THz=freqs_1d.reshape(1, -1),
        weights=np.array([1.0]),
        temperatures_K=temps
    )

    pd.testing.assert_frame_equal(df_gamma, df_explicit)


def test_thermo_shape_assertions():
    """
    Verify that incorrect shapes raise AssertionError.
    """
    freqs_2d = np.array([[1.0, 2.0, 3.0]])
    temps = np.array([300.0])

    # 2D freqs without weights should fail
    with pytest.raises(AssertionError, match="gamma-point"):
        thermodynamics.run(freqs_THz=freqs_2d, temperatures_K=temps)

    # Mismatched q-point count should fail
    with pytest.raises(AssertionError, match="Shape mismatch"):
        thermodynamics.run(
            freqs_THz=freqs_2d,
            weights=np.array([1.0, 2.0]),  # 2 weights for 1 q-point
            temperatures_K=temps
        )


if __name__ == "__main__":
    test_thermo_precision()
    test_thermo_gamma_point_default_weights()
    test_thermo_shape_assertions()
