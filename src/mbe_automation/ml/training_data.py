from mbe_automation.structure.dynamics import sample_supercells

def md_supercells(unit_cell,
                  calculator,
                  supercell_radius,
                  temperature_K=298.15,
                  training_dir,
                  total_time_fs=50000,
                  time_step_fs=0.5,
                  sampling_interval_fs=50,
                  averaging_window_fs=5000
                  ):

    sample_supercells(unit_cell,
                      calculator,
                      supercell_radius,
                      temperature_K,
                      total_time_fs,
                      time_step_fs,
                      sampling_interval_fs,
                      averaging_window_fs,
                      training_dir
                      )
