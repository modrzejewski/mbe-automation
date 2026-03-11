import re

with open("src/mbe_automation/dynamics/harmonic/core.py", "r") as f:
    content = f.read()

# I will find the lines right after `conditions = []` logic where we identify `good_points` and BEFORE `mbe_automation.storage.save_data_frame`
# Wait, currently `save_data_frame` is happening BEFORE we apply `conditions`. We need to move it AFTER we apply the empirical energy correction.

# First, locate the code chunk where `save_data_frame` and `to_csv` happen.
chunk_to_move = """    #
    # Store all harmonic properties of systems
    # used to sample the EOS curve. If EOS fit fails,
    # one can extract those data to see what went wrong.
    #
    df_eos = pd.concat(df_eos_points, ignore_index=True)
    mbe_automation.storage.save_data_frame(
        df=df_eos,
        dataset=dataset,
        key=f"{root_key}/eos_sampled"
    )
    df_eos.to_csv(os.path.join(work_dir, "eos_sampled.csv"))"""

if chunk_to_move in content:
    # Remove it
    content = content.replace(chunk_to_move, """    #
    # Construct the EOS dataframe.
    #
    df_eos = pd.concat(df_eos_points, ignore_index=True)""")

# Now locate where to put the correction and the saving logic.
# After `min_poly_points = mbe_automation.dynamics.harmonic.eos.get_minimum_points_for_eos("polynomial")` and before `if n_good_points >= min_points_needed:`

insert_target = """    n_total_points = len(df_eos[select_T[0]])
    n_good_points = len(df_eos[good_points & select_T[0]])
    min_points_needed = mbe_automation.dynamics.harmonic.eos.get_minimum_points_for_eos(equation_of_state)
    min_poly_points = mbe_automation.dynamics.harmonic.eos.get_minimum_points_for_eos("polynomial")"""

correction_logic = """

    if empirical_energy_correction is not None:
        import scipy.optimize
        import copy
        import ase.units

        T0 = empirical_energy_correction.T0
        V0_target = empirical_energy_correction.V0
        variant = empirical_energy_correction.variant
        alpha_min = empirical_energy_correction.alpha_min
        alpha_max = empirical_energy_correction.alpha_max

        # Find index for T0
        try:
            T0_idx = np.where(np.isclose(temperatures, T0, atol=1e-5))[0][0]
        except IndexError:
            raise ValueError(f"T0 ({T0}) not found in temperatures.")

        def evaluate_V_min(alpha):
            # Create a copy of df_eos to test this alpha
            df_test = df_eos.copy()

            V_array = df_test["V_crystal (Å³∕unit cell)"].to_numpy()

            # Correction term in eV (alpha in appropriate units depending on variant)
            # variant 1: alpha * (V - V0)
            # variant 2: alpha / V
            # Note: G_tot_crystal is in kJ/mol/unit cell. We need to convert eV to kJ/mol.
            eV_to_kJ_mol = ase.units.mol / (1000.0 / ase.units.kJ)
            eV_to_kJ_mol = ase.units.eV / (ase.units.kJ / ase.units.mol)
            # Wait, let's just make the correction directly in kJ/mol, assuming alpha is scaled to give energy in eV.
            # No, user didn't specify units of alpha, so alpha is just an adjustable parameter. Let's assume the formula gives energy in eV.
            # Then we convert eV to kJ/mol/unit cell.
            if variant == "alpha*(V-V0)":
                correction_eV = alpha * (V_array - V0_target)
            elif variant == "alpha/V":
                correction_eV = alpha / V_array
            else:
                raise ValueError(f"Unknown empirical energy correction variant: {variant}")

            correction_kJ_mol = correction_eV * ase.units.eV / (ase.units.kJ / ase.units.mol)

            # Add correction to G
            G_test = df_test["G_tot_crystal (kJ∕mol∕unit cell)"].to_numpy() + correction_kJ_mol

            # Perform fit for T0
            mask = good_points & select_T[T0_idx]
            V_fit = V_array[mask]
            G_fit = G_test[mask]

            fit_res = mbe_automation.dynamics.harmonic.eos.fit(
                V=V_fit,
                G=G_fit,
                equation_of_state=equation_of_state
            )

            if not fit_res.min_found:
                # If fitting fails, return a large dummy difference to guide optimizer away
                # or raise error
                return np.nan

            return fit_res.V_min - V0_target

        print(f"Applying empirical energy correction: variant={variant}, T0={T0}, V0={V0_target}")

        # We need a robust root finding. Brentq is good but needs a bracket.
        # Let's search for a bracket if f(alpha_min) and f(alpha_max) have the same sign.
        try:
            val_min = evaluate_V_min(alpha_min)
            val_max = evaluate_V_min(alpha_max)

            if np.isnan(val_min) or np.isnan(val_max):
                 raise RuntimeError("EOS fitting failed at boundaries of alpha.")

            if val_min * val_max > 0:
                 raise RuntimeError(f"Could not bracket root for alpha in range [{alpha_min}, {alpha_max}]. "
                                    f"f({alpha_min})={val_min}, f({alpha_max})={val_max}")

            optimal_alpha = scipy.optimize.brentq(evaluate_V_min, alpha_min, alpha_max)
            print(f"Found optimal alpha = {optimal_alpha}")

            # Apply optimal alpha to df_eos permanently
            V_array = df_eos["V_crystal (Å³∕unit cell)"].to_numpy()
            if variant == "alpha*(V-V0)":
                correction_eV = optimal_alpha * (V_array - V0_target)
            elif variant == "alpha/V":
                correction_eV = optimal_alpha / V_array

            correction_kJ_mol = correction_eV * ase.units.eV / (ase.units.kJ / ase.units.mol)

            # Update energies in df_eos
            df_eos["E_el_crystal (kJ∕mol∕unit cell)"] += correction_kJ_mol
            df_eos["E_tot_crystal (kJ∕mol∕unit cell)"] += correction_kJ_mol
            df_eos["F_tot_crystal (kJ∕mol∕unit cell)"] += correction_kJ_mol
            df_eos["G_tot_crystal (kJ∕mol∕unit cell)"] += correction_kJ_mol
            df_eos["H_tot_crystal (kJ∕mol∕unit cell)"] += correction_kJ_mol

        except Exception as e:
            print(f"Failed to find empirical energy correction parameter alpha: {e}")
            raise

    #
    # Store all harmonic properties of systems
    # used to sample the EOS curve.
    #
    mbe_automation.storage.save_data_frame(
        df=df_eos,
        dataset=dataset,
        key=f"{root_key}/eos_sampled"
    )
    df_eos.to_csv(os.path.join(work_dir, "eos_sampled.csv"))

"""

if "empirical_energy_correction is not None" not in content:
    content = content.replace(insert_target, insert_target + correction_logic)

with open("src/mbe_automation/dynamics/harmonic/core.py", "w") as f:
    f.write(content)
