import mbe_automation.vibrations.harmonic
import mbe_automation.structure.relax
import mbe_automation.kpoints
from ase.units import kJ, mol
import numpy as np
import ase.build
import sys
import matplotlib.pyplot as plt
import phonopy.units
import os.path

def thermodynamic(
        UnitCell,
        Molecule,
        Calc_Molecule,
        Calc_UnitCell,
        Calc_SuperCell,
        Temperatures,
        ConstantVolume=True,
        CSV_Dir="",
        Plots_Dir="",
        SupercellRadius=30.0,
        SupercellDisplacement=0.01,
        PreserveSpaceGroup=True):

    print("Thermodynamic properties")
    
    LatticeEnergy_NoOpt = StaticLatticeEnergy(UnitCell, Molecule,
                                              Calc_Molecule, Calc_UnitCell, SupercellRadius)
    RelaxedMolecule = mbe_automation.structure.relax.isolated_molecule(Molecule, Calc_Molecule)
    RelaxedUnitCell = mbe_automation.structure.relax.atoms_and_cell(
        UnitCell,
        Calc_UnitCell,
        preserve_space_group=PreserveSpaceGroup,
        constant_volume=ConstantVolume)
    LatticeEnergy_Opt = StaticLatticeEnergy(RelaxedUnitCell, RelaxedMolecule,
                                            Calc_Molecule, Calc_UnitCell, SupercellRadius)

    MeshRadius = 100.0
    thermodynamic_functions, dos = mbe_automation.vibrations.harmonic.phonopy(
        RelaxedUnitCell,
        Calc_SuperCell,
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


def StaticLatticeEnergy(UnitCell, Molecule, Calc_Molecule, Calc_UnitCell, SupercellRadius=None):
    if SupercellRadius:
        Dims = np.array(mbe_automation.kpoints.RminSupercell(UnitCell, SupercellRadius))
        Cell = ase.build.make_supercell(UnitCell, np.diag(Dims))
    else:
        Cell = UniCell.copy()
    Cell.calc = Calc_UnitCell
    Molecule.calc = Calc_Molecule    
    NAtoms = len(Molecule)
    if len(Cell) % NAtoms != 0:
        print("Invalid number of atoms in the simulation cell: cannot determine the number of molecules")
        sys.exit(1)
    NMoleculesPerCell = len(Cell) // NAtoms
    if SupercellRadius:
        print(f"Static lattice energy: Γ-point, {Dims[0]}×{Dims[1]}×{Dims[2]} supercell with {NMoleculesPerCell} molecules")
    else:
        print(f"Static lattice energy: Γ-point, unit cell with {NMoleculesPerCell} molecules")

    MoleculeEnergy = Molecule.get_potential_energy()
    CellEnergy = Cell.get_potential_energy()
    LatticeEnergy = (CellEnergy/NMoleculesPerCell - MoleculeEnergy) / (kJ/mol)
    return LatticeEnergy
