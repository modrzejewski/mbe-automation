from ase.io import read
from ase.atoms import Atoms
from mace.calculators import MACECalculator
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import time
import numpy as np
import mbe_automation.kpoints

def phonopy(
        UnitCell,
        Calculator,
        Temperatures=np.arange(0, 1001, 10),
        SupercellRadius=30.0,
        SupercellDisplacement=0.01,
        MeshRadius=100.0):

    SupercellDims = mbe_automation.kpoints.RminSupercell(UnitCell, SupercellRadius)

    print("")
    print(f"Vibrations and thermal properties in the harmonic approximation")
    print(f"Supercells for the numerical computation the dynamic matrix")
    print(f"Minimum point-image distance: {SupercellRadius:.1f} Å")
    print(f"Dimensions: {SupercellDims[0]}×{SupercellDims[1]}×{SupercellDims[2]}")
    print(f"Displacement: {SupercellDisplacement:.3f} Å")
    
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

    Supercells = phonons.get_supercells_with_displacements()
    Forces = []
    NSupercells = len(Supercells)
    print(f"Number of supercells: {NSupercells}")

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
            print(f"Processed {progress}% of supercells (Δt={now - last_time:.1f} s)")
            last_time = now
            next_print += 10

    phonons.set_forces(Forces)
    #
    # Compute second-order dynamic matrix (Hessian)
    # by numerical differentiation. The force vectors
    # used to assemble the second derivatives are obtained
    # from the list of the displaced supercells.
    #
    phonons.produce_force_constants()
    phonons.run_mesh(mesh=MeshRadius)
    #    
    # Heat capacity, free energy, entropy due to harmonic vibrations
    #
    phonons.run_thermal_properties(temperatures=Temperatures)
    thermodynamic_functions = phonons.get_thermal_properties_dict()
    #
    # Phonon density of states
    #
    phonons.run_total_dos()
    phonon_dos = phonons.get_total_dos_dict()
    
    return thermodynamic_functions, phonon_dos
    



