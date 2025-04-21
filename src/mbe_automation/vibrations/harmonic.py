from ase.io import read
from ase.atoms import Atoms
from mace.calculators import MACECalculator
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import matplotlib.pyplot as plt
import time
import mbe_automation.kpoints

def phonopy(
        UnitCell,
        Calculator,
        SupercellRadius=20.0,
        Mesh=[20,20,20],
        SupercellDisplacement=0.01,
        MinTemperature=0,
        MaxTemperature=1000,
        TemperatureStep=10,
        Sigma=0.1,
        DOSPlot="dos.png"):

    SupercellDims = mbe_automation.kpoints.RminSupercell(UnitCell, SupercellRadius)
    print(f"Supercell radius: {SupercellRadius}")
    print(f"Supercell dimensions: {SupercellDims[0]}×{SupercellDims[1]}×{SupercellDims[2]}")
    
    phonopy_struct = PhonopyAtoms(
        symbols=UnitCell.get_chemical_symbols(),
        cell=UnitCell.cell,
        masses=UnitCell.get_masses(),
        scaled_positions=UnitCell.get_scaled_positions()
    )

    phonons = Phonopy(
        phonopy_struct,
        supercell_matrix=np.diag(SupercellDims))
        
    phonons.generate_displacements(distance=SupercellDisplacement)

    Supercells = phonon.get_supercells_with_displacements()
    Forces = []
    NSupercells = len(supercells)

    start_time = time.time()
    last_time = start_time
    next_print = 10

    for i, s in enumerate(Supercells, 1):
        s_ase = Atoms(
            symbols=s.symbols,
            scaled_positions=s.scaled_positions,
            cell=s.cell,
            pbc=True)
        s_ase.calc = Calculator
        Forces.append(s_ase.get_forces())
        progress = i * 100 // NSupercells
        if progress >= next_print:
            now = time.time()
            print(f"Computed {progress}% of supercells ({now - last_time:.1f} s)")
            last_time = now
            next_print += 10

    phonon.set_forces(forces)
    phonon.produce_force_constants()
    phonon.set_mesh(mesh, is_gamma_center=True)

    # Thermal properties
    phonon.set_thermal_properties(t_min=MinTemperature,
                                  t_max=MaxTemperature,
                                  t_step=TemperatureStep)
    thermal_props = phonon.get_thermal_properties()
    temperatures = thermal_props[:, 0]
    free_energies = thermal_props[:, 1]

    # DOS
    phonon.set_total_DOS(sigma=Sigma)
    freqs, dos = phonon.get_total_DOS()

    # Plot DOS
    plt.plot(freqs, dos, label='Phonon DOS')
    plt.xlabel('Frequency (THz)')
    plt.ylabel('DOS (states/THz/unit cell)')
    plt.title('Phonon Density of States')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(DOSPlot, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        "temperatures": temperatures,
        "free_energies": free_energies,
        "frequencies": freqs,
        "dos": dos
    }


