import mbe_automation.structure.dynamics
import mbe_automation.display
import mbe_automation.kpoints
import ase.build
import os.path
import sys
import numpy as np

def supercell_md(unit_cell,
                 calculator,
                 supercell_radius,
                 training_dir,
                 temperature_K=298.15,
                 time_total_fs=50000,
                 time_step_fs=0.5,
                 sampling_interval_fs=50,
                 averaging_window_fs=5000
                 ):
    #
    # Redirect all print instructions to the log file. Apply buffer
    # flush after every print.
    #
    summary_file = os.path.join(training_dir, "supercell_md.txt")
    sys.stdout = mbe_automation.display.ReplicatedOutput(summary_file)
    
    dims = np.array(mbe_automation.kpoints.RminSupercell(unit_cell, supercell_radius))
    super_cell = ase.build.make_supercell(unit_cell, np.diag(dims))
    mbe_automation.display.framed("Molecular dynamics (crystal)")
    print(f"Requested supercell radius R={supercell_radius:.1f} Å")
    print(f"{len(super_cell)} atoms in the {dims[0]}×{dims[1]}×{dims[2]} supercell")
    
    mbe_automation.structure.dynamics.sample_NVT(super_cell,
                                                 calculator,
                                                 training_dir,
                                                 temperature_K,
                                                 time_total_fs,
                                                 time_step_fs,
                                                 sampling_interval_fs,
                                                 averaging_window_fs,
                                                 trajectory_file=os.path.join(training_dir, "supercell_md.traj"),
                                                 plot_file=os.path.join(training_dir, "supercell_md.png")
                                                 )

    print(f"Summary written to {summary_file}")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout



def molecule_md(molecule,
                calculator,
                training_dir,
                temperature_K=298.15,
                time_total_fs=50000,
                time_step_fs=0.5,
                sampling_interval_fs=50,
                averaging_window_fs=5000
                ):
    #
    # Redirect all print instructions to the log file. Apply buffer
    # flush after every print.
    #
    summary_file = os.path.join(training_dir, "molecule_md.txt")
    sys.stdout = mbe_automation.display.ReplicatedOutput(summary_file)

    mbe_automation.display.framed("Molecular dynamics (single molecule)")
    mbe_automation.structure.dynamics.sample_NVT(molecule,
                                                 calculator,
                                                 training_dir,
                                                 temperature_K,
                                                 time_total_fs,
                                                 time_step_fs,
                                                 sampling_interval_fs,
                                                 averaging_window_fs,
                                                 trajectory_file=os.path.join(training_dir, "molecule_md.traj"),
                                                 plot_file=os.path.join(training_dir, "molecule_md.png")
                                                 )

    print(f"Summary written to {summary_file}")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout

    
