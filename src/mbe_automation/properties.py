import mbe_automation.vibrations.harmonic
import mbe_automation.structure.relax
from ase.units import kJ, mol
import sys
import matplotlib.pyplot as plt
import phonopy.units
import os.path

def thermodynamic(
        UnitCell,
        Molecule,
        Calculator,
        Temperatures,
        ConstantVolume=True,
        CSV_Dir="",
        Plots_Dir="",
        SupercellRadius=30.0,
        SupercellDisplacement=0.01,
        PreserveSpaceGroup=True):

    print("")
    print("Thermodynamic properties")
    NAtoms = len(Molecule)
    if len(UnitCell) % NAtoms != 0:
        print("Invalid number of atoms in the unit cell: cannot determine the number of molecules")
        sys.exit(1)
    NMoleculesPerCell = len(UnitCell) // NAtoms
    print(f"Molecules per cell: {NMoleculesPerCell}")

    Molecule.calc = Calculator
    UnitCell.calc = Calculator
    MoleculeEnergy = Molecule.get_potential_energy()
    CellEnergy = UnitCell.get_potential_energy()
    LatticeEnergy_NoOpt = (CellEnergy/NMoleculesPerCell - MoleculeEnergy) / (kJ/mol)

    RelaxedMolecule = mbe_automation.structure.relax.isolated_molecule(Molecule, Calculator)
    RelaxedUnitCell = mbe_automation.structure.relax.atoms_and_cell(
        UnitCell,
        Calculator,
        preserve_space_group=PreserveSpaceGroup,
        constant_volume=ConstantVolume)

    RelaxedMoleculeEnergy = RelaxedMolecule.get_potential_energy()
    RelaxedCellEnergy = RelaxedUnitCell.get_potential_energy()
    LatticeEnergy_Opt = (RelaxedCellEnergy/NMoleculesPerCell - RelaxedMoleculeEnergy) / (kJ/mol)

    MeshRadius = 100.0
    thermodynamic_functions, dos = mbe_automation.vibrations.harmonic.phonopy(
        RelaxedUnitCell,
        Calculator,
        Temperatures,
        SupercellRadius,
        SupercellDisplacement,
        MeshRadius)
    # 
    # Plot density of states
    #
    plt.plot(dos["frequency_points"], dos["total_dos"])
    plt.xlabel('Frequency (THz)')
    plt.ylabel('DOS (states/THz/unit cell)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Plots_Dir, "phonon_density_of_states.png"), dpi=300, bbox_inches='tight')
    plt.close()

    T = thermodynamic_functions["temperatures"]
    Cv = thermodynamic_functions["heat_capacity"]
    F = thermodynamic_functions["free_energy"]
    S = thermodynamic_functions["entropy"]
    #
    # Normalize Cv to Cv/CvInf where
    # CvInf = lim(T->Inf) Cv(T) = 3 * N * kb 
    # is the classical limit of heat capacity.
    #
    # Cv/CvInf is heat capacity per single
    # atom in the unit cell. Cv/CvInf approaches
    # 1.0 at high temperatures.
    # 
    #
    CvInf = 3 * len(UnitCell) * phonopy.units.Avogadro * phonopy.units.kb_J
    Cv = Cv / CvInf
    plt.plot(T, Cv)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Cv/(3 N kb)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Plots_Dir, "heat_capacity.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Calculations completed")
    print(f"Energies (kJ/mol):")
    print(f"{'static lattice energy (input coords)':40} {LatticeEnergy_NoOpt:.6f}")
    print(f"{'static lattice energy (relaxed coords)':40} {LatticeEnergy_Opt:.6f}")


