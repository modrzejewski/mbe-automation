import numpy as np

# We want to trace where the energy should be added.
# In `src/mbe_automation/dynamics/harmonic/core.py`, in `equilibrium_curve`, we fit the EOS:
#         fit = mbe_automation.dynamics.harmonic.eos.fit(
#            V=df_eos[good_points & select_T[i]]["V_crystal (Å³∕unit cell)"].to_numpy(),
#            G=df_eos[good_points & select_T[i]]["G_tot_crystal (kJ∕mol∕unit cell)"].to_numpy(),
#            equation_of_state=equation_of_state
#        )
