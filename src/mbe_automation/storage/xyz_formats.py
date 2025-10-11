from __future__ import annotations
from collections import defaultdict
from typing import Any
from typing import TYPE_CHECKING
import pymatgen
import ase.io
import ase
import numpy as np
import numpy.typing as npt
import warnings

if TYPE_CHECKING:
    import mbe_automation.dynamics.harmonic.modes

import mbe_automation.structure
import mbe_automation.storage.core
import mbe_automation.storage.views

def _cif_with_adps(
        save_path: str,
        struct: pymatgen.core.Structure,
        adps_cif: npt.NDArray[np.floating] | None = None,
        symprec: float = 1.0E-5,
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
            for sites in spg_analyzer.get_symmetrized_structure().equivalent_sites  # type: ignore[reportPossiblyUnboundVariable]
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
        adps_asymmetric_unit = adps_cif[original_indices]
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


def from_xyz_file(
        read_path: str,
        convert_to_symmetrized_primitive: bool = True,
        symprec: float = 1.0E-2
) -> ase.Atoms:
    
    if read_path.lower().endswith(".cif"):
        structure = pymatgen.core.Structure.from_file(read_path)
        system = pymatgen.io.ase.AseAtomsAdaptor.get_atoms(structure)
    else:
        system = ase.io.read(read_path)

    if convert_to_symmetrized_primitive and np.all(system.pbc):
        system = mbe_automation.structure.crystal.to_symmetrized_primitive(
            unit_cell=system,
            symprec=symprec
        )
        
    return system


def to_xyz_file(
        save_path: str,
        system: ase.Atoms | mbe_automation.storage.core.Structure,
        frame_index: int = 0,
        thermal_displacements: mbe_automation.dynamics.harmonic.modes.ThermalDisplacements | None = None,
        temperature_idx: int = 0,
        symprec: float = 1.0E-5
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
