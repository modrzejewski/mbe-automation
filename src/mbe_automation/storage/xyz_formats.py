from __future__ import annotations
from collections import defaultdict
from typing import Any, Literal
from typing import TYPE_CHECKING
import pymatgen
from pymatgen.io.cif import CifParser
import pymatgen.core
import ase.io
import ase
import numpy as np
import numpy.typing as npt
import warnings
import gemmi

if TYPE_CHECKING:
    import mbe_automation.dynamics.harmonic.modes

import mbe_automation.structure
import mbe_automation.storage.core
import mbe_automation.storage.views
import mbe_automation.common.display
from mbe_automation.configs.structure import SYMMETRY_TOLERANCE_STRICT, SYMMETRY_TOLERANCE_LOOSE
from mbe_automation.dynamics.harmonic.modes import symmetrize_adps

def _read_cif_pymatgen(filepath: str) -> pymatgen.core.Structure:
    """Read a structure from a file."""
    parser = CifParser(
        filepath,
        site_tolerance=0.1,
        occupancy_tolerance=np.inf
    )
    structures = parser.parse_structures()
    return structures[0]

def _read_cif_gemmi(
        filepath: str
    ) -> pymatgen.core.Structure:
    """
    Read periodic structure and expand asymmetric unit to P1 unit cell.
    
    Args:
        filepath: Path to the structure file.
        
    Returns:
        Periodic structure with expanded atomic positions.
    """
    document = gemmi.cif.read_file(filepath)
    block = document.sole_block()
    asymmetric_unit = gemmi.make_small_structure_from_block(block)
    
    unit_cell_sites = asymmetric_unit.get_all_unit_cell_sites()
    unit_cell_sites = sorted(unit_cell_sites, key=lambda site: site.occ, reverse=True)
    
    crystal_lattice = pymatgen.core.Lattice.from_parameters(
        a=asymmetric_unit.cell.a,
        b=asymmetric_unit.cell.b,
        c=asymmetric_unit.cell.c,
        alpha=asymmetric_unit.cell.alpha,
        beta=asymmetric_unit.cell.beta,
        gamma=asymmetric_unit.cell.gamma
    )
    
    atomic_symbols = [site.element.name for site in unit_cell_sites]
    
    fractional_positions = [
        [site.fract.x, site.fract.y, site.fract.z]
        for site in unit_cell_sites
    ]
    
    structure = pymatgen.core.Structure(
        lattice=crystal_lattice,
        species=atomic_symbols,
        coords=fractional_positions,
        coords_are_cartesian=False
    )
    
    structure.merge_sites(tol=0.1, mode="delete")
    
    return structure

def _read_cif(
    filepath: str,
    backend: Literal["gemmi", "pymatgen"] = "gemmi"
):
    if backend == "gemmi":
        return _read_cif_gemmi(filepath)
    elif backend == "pymatgen":
        return _read_cif_pymatgen(filepath)
    else:
        raise ValueError("Invalid backend requested in _read_cif.")        

def _cif_with_adps(
        save_path: str,
        struct: pymatgen.core.Structure,
        adps_cif: npt.NDArray[np.floating] | None = None,
        symprec: float = SYMMETRY_TOLERANCE_STRICT,
        significant_figures: int = 8,
    ) -> None:
    """
    Code from pymatgen.io.cif.CifWriter modified to handle anisotropic displacement
    parameters.
    """
    if adps_cif is not None:
        if len(struct) != adps_cif.shape[0]:
            raise ValueError("Number of atoms in structure does not match the number of ADPs.")

    blocks: dict[str, Any] = {}
    lattice = struct.lattice
    comp = struct.composition
    no_oxi_comp = comp.element_composition
    format_str: str = f"{{:.{significant_figures}f}}"
    blocks["_chemical_formula_sum"] = no_oxi_comp.formula
    if symprec is not None:
        spg_analyzer = pymatgen.symmetry.analyzer.SpacegroupAnalyzer(struct, symprec)
        blocks["_space_group_crystal_system"] = spg_analyzer.get_crystal_system()
        blocks["_space_group_name_H-M_alt"] = spg_analyzer.get_space_group_symbol()
        blocks["_space_group_IT_number"] = spg_analyzer.get_space_group_number()
        blocks["_space_group_name_Hall"] = spg_analyzer.get_hall()
    else:
        blocks["_space_group_name_H-M_alt"] = "P 1"
        blocks["_space_group_IT_number"] = 1
        
    for cell_attr in ("a", "b", "c"):
        blocks[f"_cell_length_{cell_attr}"] = format_str.format(getattr(lattice, cell_attr))
    for cell_attr in ("alpha", "beta", "gamma"):
        blocks[f"_cell_angle_{cell_attr}"] = format_str.format(getattr(lattice, cell_attr))
    blocks["_cell_volume"] = format_str.format(lattice.volume)

    if symprec is None:
        blocks["_symmetry_equiv_pos_site_id"] = ["1"]
        blocks["_symmetry_equiv_pos_as_xyz"] = ["x, y, z"]
    else:
        symm_ops: list[pymatgen.core.operations.SymmOp] = []
        for op in spg_analyzer.get_symmetry_operations():
            v = op.translation_vector
            symm_ops.append(pymatgen.core.operations.SymmOp.from_rotation_and_translation(op.rotation_matrix, v))
            
        ops = [op.as_xyz_str() for op in symm_ops]
        blocks["_symmetry_equiv_pos_site_id"] = [f"{i}" for i in range(1, len(ops) + 1)]
        blocks["_symmetry_equiv_pos_as_xyz"] = ops

    loops: list[list[str]] = [
        ["_symmetry_equiv_pos_site_id", "_symmetry_equiv_pos_as_xyz"],
    ]

    atom_site_type_symbol = []
    atom_site_symmetry_multiplicity = []
    atom_site_fract_x = []
    atom_site_fract_y = []
    atom_site_fract_z = []
    atom_site_label = []
    atom_site_occupancy = []
    atom_site_properties: dict[str, list] = defaultdict(list)
    count = 0

    aniso_site_labels = []
    original_indices = []

    if symprec is None:
        for site in struct:

            if not site.is_ordered:
                raise TypeError(
                    f"The CIF writer was told to assume an ordered crystal, but found a disordered site: {site.species}"
                )

            original_indices.append(struct.index(site))

            element = site.species.elements[0]
            site_label = f"{element.symbol}{count}"
            atom_site_type_symbol.append(str(element))
            atom_site_symmetry_multiplicity.append("1")
            atom_site_fract_x.append(format_str.format(site.a))
            atom_site_fract_y.append(format_str.format(site.b))
            atom_site_fract_z.append(format_str.format(site.c))
            atom_site_label.append(site_label)
            atom_site_occupancy.append("1")
            count += 1
            
    else:

        unique_sites = [
            (
                min(sites, key=lambda site: tuple(abs(x) for x in site.frac_coords)),
                len(sites),
            )
            for sites in spg_analyzer.get_symmetrized_structure().equivalent_sites 
        ]
        for site, mult in sorted(
            unique_sites,
            key=lambda t: (
                t[0].species.average_electroneg,
                -t[1],
                t[0].a,
                t[0].b,
                t[0].c,
            ),
        ):

            if not site.is_ordered:
                raise TypeError(
                    f"The CIF writer was told to assume an ordered crystal, but found a disordered site: {site.species}"
                )

            original_indices.append(struct.index(site))

            element = site.species.elements[0]
            site_label = f"{element.symbol}{count}"
            atom_site_type_symbol.append(str(element))
            atom_site_symmetry_multiplicity.append(f"{mult}")
            atom_site_fract_x.append(format_str.format(site.a))
            atom_site_fract_y.append(format_str.format(site.b))
            atom_site_fract_z.append(format_str.format(site.c))
            atom_site_label.append(site_label)
            atom_site_occupancy.append("1")
            count += 1

    blocks["_atom_site_type_symbol"] = atom_site_type_symbol
    blocks["_atom_site_label"] = atom_site_label
    blocks["_atom_site_symmetry_multiplicity"] = atom_site_symmetry_multiplicity
    blocks["_atom_site_fract_x"] = atom_site_fract_x
    blocks["_atom_site_fract_y"] = atom_site_fract_y
    blocks["_atom_site_fract_z"] = atom_site_fract_z
    blocks["_atom_site_occupancy"] = atom_site_occupancy
    loop_labels = [
        "_atom_site_type_symbol",
        "_atom_site_label",
        "_atom_site_symmetry_multiplicity",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]    
    loops.append(loop_labels)

    if adps_cif is not None:
        if symprec is not None:
            #
            # Compute averaged ADPs for symmetry-equivalent atoms
            #        
            adps_to_use = symmetrize_adps(struct, adps_cif, symprec=symprec)
        else:
            adps_to_use = adps_cif

        adps_asymmetric_unit = adps_to_use[original_indices]
        blocks["_atom_site_aniso_label"] = atom_site_label
        blocks["_atom_site_aniso_U_11"] = [format_str.format(adp[0, 0]) for adp in adps_asymmetric_unit]
        blocks["_atom_site_aniso_U_22"] = [format_str.format(adp[1, 1]) for adp in adps_asymmetric_unit]
        blocks["_atom_site_aniso_U_33"] = [format_str.format(adp[2, 2]) for adp in adps_asymmetric_unit]
        blocks["_atom_site_aniso_U_23"] = [format_str.format(adp[1, 2]) for adp in adps_asymmetric_unit]
        blocks["_atom_site_aniso_U_13"] = [format_str.format(adp[0, 2]) for adp in adps_asymmetric_unit]
        blocks["_atom_site_aniso_U_12"] = [format_str.format(adp[0, 1]) for adp in adps_asymmetric_unit]
        aniso_loop = [
            "_atom_site_aniso_label", "_atom_site_aniso_U_11", "_atom_site_aniso_U_22",
            "_atom_site_aniso_U_33", "_atom_site_aniso_U_23", "_atom_site_aniso_U_13",
            "_atom_site_aniso_U_12"
        ]
        loops.append(aniso_loop)

    cif_block = pymatgen.io.cif.CifBlock(blocks, loops, comp.reduced_formula)
    cif_block.max_len = 160
    dct = {comp.reduced_formula: cif_block}
    with open(save_path, mode="w", encoding="utf-8") as cif_file:
        cif_file.write(str(pymatgen.io.cif.CifFile(dct)))

    return None


def _print_cell_summary(system: ase.Atoms, label: str) -> None:
    """Print lattice parameters and atom count for a periodic system."""
    space_group, hmsymbol = mbe_automation.structure.crystal.check_symmetry(
        unit_cell=system,
        symmetry_thresh=SYMMETRY_TOLERANCE_STRICT,
    )
    lengths = system.cell.lengths()
    angles = system.cell.angles()
    print(label)
    print(f"  Symmetry          [{hmsymbol}][{space_group}]")
    print(f"  Lattice lengths   a={lengths[0]:.4f}, b={lengths[1]:.4f}, c={lengths[2]:.4f} Å")
    print(f"  Lattice angles    alpha={angles[0]:.4f}, beta={angles[1]:.4f}, gamma={angles[2]:.4f} °")
    print(f"  Cell volume       {system.get_volume():.4f} Å³")
    print(f"  Number of atoms   {len(system)}")


def from_xyz_file(
        read_path: str,
        transform: Literal[
            "to_symmetrized_conventional_cell",
            "to_symmetrized_primitive_cell",
            "no_transformation"
        ] = "to_symmetrized_primitive_cell",
        symprec: float = SYMMETRY_TOLERANCE_LOOSE,
        cif_backend: Literal["gemmi", "pymatgen"] = "gemmi"
) -> ase.Atoms:

    mbe_automation.common.display.framed([
        "Reading structure from file",
        read_path
    ])

    if read_path.lower().endswith(".cif"):
        structure = _read_cif(read_path, backend=cif_backend)
        system = pymatgen.io.ase.AseAtomsAdaptor.get_atoms(structure)
    else:
        system = ase.io.read(read_path)

    if np.all(system.pbc):
        do_transform = transform != "no_transformation"
        
        if do_transform:
            print(f"Input cell will be symmetrized with tolerance {symprec:.6f} Å")
            system_initial = system
            if transform == "to_symmetrized_primitive_cell":
                cell_vectors, atomic_numbers, scaled_positions = (
                    mbe_automation.structure.crystal.to_symmetrized_primitive_cell(
                        cell_vectors=system.cell.array,
                        atomic_numbers=system.get_atomic_numbers(),
                        scaled_positions=system.get_scaled_positions(),
                        symprec=symprec,
                    )
                )
            elif transform == "to_symmetrized_conventional_cell":
                cell_vectors, atomic_numbers, scaled_positions = (
                    mbe_automation.structure.crystal.to_symmetrized_conventional_cell(
                        cell_vectors=system.cell.array,
                        atomic_numbers=system.get_atomic_numbers(),
                        scaled_positions=system.get_scaled_positions(),
                        symprec=symprec,
                    )
                )
            else:
                raise ValueError(f"Unknown transformation: {transform}")
                
            system = ase.Atoms(
                numbers=atomic_numbers,
                scaled_positions=scaled_positions,
                cell=cell_vectors,
                pbc=True,
            )

        mbe_automation.structure.crystal.display_conventional_cell(
            structure=system,
            label="symmetrized input structure" if do_transform else "input structure",
            symprec=symprec,
        )

    return system

def to_xyz_file(
        save_path: str,
        system: ase.Atoms | mbe_automation.storage.core.Structure,
        frame_index: int = 0,
        thermal_displacements: mbe_automation.dynamics.harmonic.modes.ThermalDisplacements | None = None,
        temperature_idx: int = 0,
        symprec: float = SYMMETRY_TOLERANCE_STRICT
):
    if isinstance(system, mbe_automation.storage.core.Structure):
        system_ase = mbe_automation.storage.views.to_ase(system, frame_index=frame_index)
    else:
        system_ase = system
        
    if save_path.lower().endswith(".cif"):
        pmg_structure = pymatgen.io.ase.AseAtomsAdaptor.get_structure(system_ase)
        _cif_with_adps(
            save_path=save_path,
            struct=pmg_structure,
            adps_cif=(
                thermal_displacements.mean_square_displacements_matrix_diagonal_cif[temperature_idx]
                if thermal_displacements is not None else None
            ),
            symprec=symprec
        )
    else:
        ase.io.write(save_path, system_ase)


def to_cif_file(
        save_path: str,
        system: ase.Atoms | mbe_automation.storage.core.Structure,
        frame_index: int = 0,
        thermal_displacements: mbe_automation.dynamics.harmonic.modes.ThermalDisplacements | None = None,
        temperature_idx: int = 0,
        symprec: float = SYMMETRY_TOLERANCE_STRICT
) -> None:
    """
    Save the system to a CIF file.
    
    This is a wrapper around to_xyz_file that enforces the .cif extension.
    """
    if not save_path.lower().endswith(".cif"):
        raise ValueError(f"The save_path must end with .cif, got: {save_path}")

    to_xyz_file(
        save_path=save_path,
        system=system,
        frame_index=frame_index,
        thermal_displacements=thermal_displacements,
        temperature_idx=temperature_idx,
        symprec=symprec
    )
