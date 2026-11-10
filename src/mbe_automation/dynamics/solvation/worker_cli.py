from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def _read_xyz(xyz_path: Path) -> tuple[list[str], list[list[float]]]:
    """Parse elemental symbols and Cartesian coordinates (Angstrom) from an XYZ file."""
    lines = [
        line.strip()
        for line in xyz_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n_atoms = int(lines[0])
    # Second line in XYZ can be an empty comment line, so re-read raw lines preserving line 2
    raw_lines = xyz_path.read_text(encoding="utf-8").splitlines()
    atom_lines = [line.strip() for line in raw_lines[2 : 2 + n_atoms]]
    symbols: list[str] = []
    coords: list[list[float]] = []
    for row in atom_lines:
        parts = row.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, coords


def _format_xyz_block(symbols: list[str], coords: list[list[float]]) -> str:
    """Format atomic symbols and Angstrom coordinates into an XYZ text block."""
    body = "\n".join(
        f"{sym:<4s} {xyz[0]:16.8f} {xyz[1]:16.8f} {xyz[2]:16.8f}"
        for sym, xyz in zip(symbols, coords)
    )
    return f"{len(symbols)}\nconformer\n{body}\n"


def _kabsch_rmsd(
    coords_a,
    coords_b,
) -> float:
    """Compute optimal rigid-body superposed RMSD between equal-length coordinate arrays."""
    import numpy as np

    p = np.asarray(coords_a, dtype=np.float64)
    q = np.asarray(coords_b, dtype=np.float64)
    p_centered = p - np.mean(p, axis=0, keepdims=True)
    q_centered = q - np.mean(q, axis=0, keepdims=True)
    cov = p_centered.T @ q_centered
    u_mat, _, vt_mat = np.linalg.svd(cov)
    det_sign = np.linalg.det(vt_mat.T @ u_mat.T)
    corr = np.diag([1.0, 1.0, 1.0 if det_sign >= 0.0 else -1.0])
    rot = vt_mat.T @ corr @ u_mat.T
    p_aligned = p_centered @ rot.T
    diff = p_aligned - q_centered
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def _perceive_openff_solute(
    xyz_path: Path,
    solute_smiles: str | None,
    solute_charge: int,
):
    """Construct an OpenFF Molecule preserving the exact atom ordering of xyz_path."""
    from openff.toolkit.topology import Molecule
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds

    symbols, coords = _read_xyz(xyz_path)
    xyz_block = _format_xyz_block(symbols, coords)
    rd_mol = Chem.MolFromXYZBlock(xyz_block)
    if rd_mol is None:
        raise RuntimeError(f"RDKit failed to parse XYZ file: {xyz_path}.")

    assigned = False
    if solute_smiles is not None:
        template = Chem.MolFromSmiles(solute_smiles)
        if template is None:
            raise ValueError(f"Invalid solute_smiles: {solute_smiles}.")
        template_with_h = Chem.AddHs(template)
        try:
            mol_conn = Chem.Mol(rd_mol)
            rdDetermineBonds.DetermineConnectivity(mol_conn)
            rd_mol = AllChem.AssignBondOrdersFromTemplate(
                template_with_h, mol_conn
            )
            Chem.SanitizeMol(rd_mol)
            Chem.AssignStereochemistryFrom3D(rd_mol)
            assigned = True
        except Exception:
            assigned = False

    if not assigned:
        rd_mol_fallback = Chem.MolFromXYZBlock(xyz_block)
        rdDetermineBonds.DetermineBonds(rd_mol_fallback, charge=int(solute_charge))
        Chem.SanitizeMol(rd_mol_fallback)
        Chem.AssignStereochemistryFrom3D(rd_mol_fallback)
        rd_mol = rd_mol_fallback

    off_mol = Molecule.from_rdkit(
        rd_mol,
        allow_undefined_stereo=True,
        hydrogens_are_explicit=True,
    )
    off_mol.name = "LIG"
    for atom in off_mol.atoms:
        atom.metadata["residue_name"] = "LIG"
    return off_mol


def _openmm_topology_from_molecules(
    solute_mol,
    solvent_mol,
    n_solvent: int,
    box_length_nm: float,
):
    """Build an OpenMM Topology with residue 0 named LIG followed by solvent residues."""
    from openmm import Vec3, app, unit

    topology = app.Topology()
    topology.setPeriodicBoxVectors(
        (
            Vec3(box_length_nm, 0.0, 0.0) * unit.nanometers,
            Vec3(0.0, box_length_nm, 0.0) * unit.nanometers,
            Vec3(0.0, 0.0, box_length_nm) * unit.nanometers,
        )
    )
    chain_solute = topology.addChain("A")
    res_lig = topology.addResidue("LIG", chain_solute)
    lig_atoms = []
    element_counts: dict[str, int] = {}
    for off_atom in solute_mol.atoms:
        elem = app.Element.getByAtomicNumber(off_atom.atomic_number)
        sym = elem.symbol
        element_counts[sym] = element_counts.get(sym, 0) + 1
        omm_atom = topology.addAtom(f"{sym}{element_counts[sym]}", elem, res_lig)
        lig_atoms.append(omm_atom)
    for bond in solute_mol.bonds:
        topology.addBond(lig_atoms[bond.atom1_index], lig_atoms[bond.atom2_index])

    chain_solvent = topology.addChain("B")
    is_water = (
        solvent_mol.n_atoms == 3
        and sorted(a.atomic_number for a in solvent_mol.atoms) == [1, 1, 8]
    )
    res_name = "HOH" if is_water else "SLV"

    for _ in range(n_solvent):
        res_slv = topology.addResidue(res_name, chain_solvent)
        slv_atoms = []
        slv_counts: dict[str, int] = {}
        for off_atom in solvent_mol.atoms:
            elem = app.Element.getByAtomicNumber(off_atom.atomic_number)
            sym = elem.symbol
            slv_counts[sym] = slv_counts.get(sym, 0) + 1
            if is_water and elem.atomic_number == 8:
                atom_name = "O"
            elif is_water and elem.atomic_number == 1:
                atom_name = f"H{slv_counts[sym]}"
            else:
                atom_name = f"{sym}{slv_counts[sym]}"
            slv_atoms.append(topology.addAtom(atom_name, elem, res_slv))
        for bond in solvent_mol.bonds:
            topology.addBond(slv_atoms[bond.atom1_index], slv_atoms[bond.atom2_index])

    return topology


def _pack_box_with_packmol(
    solute_xyz_path: Path,
    solvent_mol,
    n_solvent: int,
    density_g_cm3: float,
    solute_mass_amu: float,
) -> tuple[list[list[float]], float]:
    """Pack 1 solute and n_solvent solvent molecules into a cubic box via Packmol."""
    from openmm import unit

    packmol_bin = shutil.which("packmol")
    if packmol_bin is None:
        raise RuntimeError(
            "packmol executable not found on PATH inside the mace-openmm environment. "
            "Install it with: mamba install -n mace-openmm -c conda-forge packmol"
        )

    slv_mass_amu = sum(
        atom.mass.m_as("dalton")
        if hasattr(atom.mass, "m_as")
        else atom.mass.value_in_unit(unit.dalton)
        for atom in solvent_mol.atoms
    )
    total_mass_g = (solute_mass_amu + n_solvent * slv_mass_amu) / 6.02214076e23
    target_vol_A3 = (total_mass_g / density_g_cm3) * 1.0e24
    target_edge_A = target_vol_A3 ** (1.0 / 3.0)
    pack_edge_A = max(target_edge_A * (105.0 / 100.0), 18.0)

    rd_slv = solvent_mol.to_rdkit()
    conf = rd_slv.GetConformer(0)
    slv_symbols = [atom.GetSymbol() for atom in rd_slv.GetAtoms()]
    slv_coords = [
        [
            float(conf.GetAtomPosition(i).x),
            float(conf.GetAtomPosition(i).y),
            float(conf.GetAtomPosition(i).z),
        ]
        for i in range(rd_slv.GetNumAtoms())
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        sol_file = tmp / "solute.xyz"
        slv_file = tmp / "solvent.xyz"
        out_file = tmp / "packed.xyz"
        inp_file = tmp / "pack.inp"

        shutil.copyfile(solute_xyz_path, sol_file)
        slv_file.write_text(_format_xyz_block(slv_symbols, slv_coords), encoding="utf-8")

        box_max = pack_edge_A - 1.5
        inp_text = (
            "tolerance 2.0\n"
            "filetype xyz\n"
            f"output {out_file.name}\n\n"
            f"structure {sol_file.name}\n"
            "  number 1\n"
            f"  fixed {pack_edge_A / 2.0:.4f} {pack_edge_A / 2.0:.4f} {pack_edge_A / 2.0:.4f} 0. 0. 0.\n"
            "  centerofmass\n"
            "end structure\n\n"
            f"structure {slv_file.name}\n"
            f"  number {n_solvent}\n"
            f"  inside box 0.0 0.0 0.0 {box_max:.4f} {box_max:.4f} {box_max:.4f}\n"
            "end structure\n"
        )
        inp_file.write_text(inp_text, encoding="utf-8")

        with inp_file.open("r", encoding="utf-8") as stdin_f:
            proc = subprocess.run(
                [packmol_bin],
                stdin=stdin_f,
                cwd=tmp,
                capture_output=True,
                text=True,
                check=False,
            )
        if proc.returncode != 0 or not out_file.is_file():
            raise RuntimeError(
                f"Packmol failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
            )

        _, packed_coords_A = _read_xyz(out_file)

    return packed_coords_A, pack_edge_A / 10.0


def cmd_check(spec: dict) -> None:
    """Validate that required packages and softcore forks exist in mace-openmm."""
    import mace  # noqa: F401
    import mace_md  # noqa: F401
    import mdtraj  # noqa: F401
    import openff.toolkit  # noqa: F401
    import openmm  # noqa: F401
    import openmmforcefields  # noqa: F401
    from openmmml import MLPotential
    import openmmtools  # noqa: F401
    import openmmtorch  # noqa: F401
    import rdkit  # noqa: F401

    if spec["alchemical"]["system_type"] == "pure":
        if not hasattr(MLPotential, "createAlchemicalSystem"):
            raise RuntimeError(
                "Installed openmmml lacks MLPotential.createAlchemicalSystem. "
                "Install the ml_alchemy fork via: "
                "pip install git+https://github.com/jharrymoore/openmm-ml.git@ml_alchemy"
            )

    model_path = Path(spec["alchemical"]["model_path"])
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Alchemical MACE model file does not exist: {model_path}."
        )

    solvent = spec["solvent"]["solvent"].strip().lower()
    if solvent != "water" and shutil.which("packmol") is None:
        raise RuntimeError(
            "Non-aqueous solvent requested, but packmol is missing on PATH. "
            "Install via: mamba install -n mace-openmm -c conda-forge packmol"
        )


def cmd_prepare(spec: dict) -> None:
    """Build topology, solvate, run classical NPT pre-equilibration, and write PDB/SDF."""
    from mace_md.nnp_repex.repex import (
        assert_no_residue_constraints,
        get_atoms_from_resname,
    )
    import numpy as np
    from openff.toolkit.topology import Molecule
    import openmm
    from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Vec3, app, unit
    from openmm.app.internal.unitcell import reducePeriodicBoxVectors
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    solute_xyz = Path(spec["paths"]["solute_xyz"])
    lig_sdf = Path(spec["paths"]["lig_sdf"])
    solvated_pdb = Path(spec["paths"]["solvated_pdb"])

    solute_mol = _perceive_openff_solute(
        xyz_path=solute_xyz,
        solute_smiles=spec["solute_smiles"],
        solute_charge=int(spec["solute_charge"]),
    )
    solute_mol.to_file(str(lig_sdf), file_format="SDF")

    solute_mass_amu = sum(
        atom.mass.m_as("dalton")
        if hasattr(atom.mass, "m_as")
        else atom.mass.value_in_unit(unit.dalton)
        for atom in solute_mol.atoms
    )

    solvent_cfg = spec["solvent"]
    solvent_str = solvent_cfg["solvent"].strip()
    is_water = solvent_str.lower() == "water"
    density_g_cm3 = float(solvent_cfg["solvent_density_g_cm3"])
    n_solvent = int(solvent_cfg["n_solvent_molecules"])
    padding_nm = solvent_cfg["padding_nm"]

    _, solute_coords_A = _read_xyz(solute_xyz)

    if is_water and (padding_nm is not None or shutil.which("packmol") is None):
        topology_lig = app.Topology()
        chain = topology_lig.addChain("A")
        res = topology_lig.addResidue("LIG", chain)
        omm_atoms = []
        counts: dict[str, int] = {}
        for off_atom in solute_mol.atoms:
            elem = app.Element.getByAtomicNumber(off_atom.atomic_number)
            counts[elem.symbol] = counts.get(elem.symbol, 0) + 1
            omm_atoms.append(
                topology_lig.addAtom(
                    f"{elem.symbol}{counts[elem.symbol]}", elem, res
                )
            )
        for bond in solute_mol.bonds:
            topology_lig.addBond(
                omm_atoms[bond.atom1_index], omm_atoms[bond.atom2_index]
            )
        pos_nm = [Vec3(c[0] / 10.0, c[1] / 10.0, c[2] / 10.0) for c in solute_coords_A]
        modeller = app.Modeller(topology_lig, pos_nm * unit.nanometers)

        ff_water = app.ForceField("amber/tip3p_standard.xml")
        smirnoff_gen = SMIRNOFFTemplateGenerator(
            molecules=[solute_mol],
            forcefield="openff_unconstrained-2.1.0.offxml",
        )
        ff_water.registerTemplateGenerator(smirnoff_gen.generator)

        if padding_nm is not None:
            modeller.addSolvent(
                ff_water,
                model="tip3p",
                padding=float(padding_nm) * unit.nanometers,
                neutralize=False,
            )
        else:
            total_mass_g = (solute_mass_amu + n_solvent * 18.01528) / 6.02214076e23
            edge_nm = ((total_mass_g / density_g_cm3) * 1.0e24) ** (1.0 / 3.0) / 10.0
            edge_nm = max(edge_nm, 2.2)
            modeller.addSolvent(
                ff_water,
                model="tip3p",
                boxSize=Vec3(edge_nm, edge_nm, edge_nm) * unit.nanometers,
                neutralize=False,
            )
        forcefield = ff_water
    else:
        if is_water:
            solvent_mol = Molecule.from_smiles("O")
        else:
            solvent_mol = Molecule.from_smiles(
                solvent_str, allow_undefined_stereo=True
            )
        solvent_mol.generate_conformers(n_conformers=1)
        packed_coords_A, box_edge_nm = _pack_box_with_packmol(
            solute_xyz_path=solute_xyz,
            solvent_mol=solvent_mol,
            n_solvent=n_solvent,
            density_g_cm3=density_g_cm3,
            solute_mass_amu=solute_mass_amu,
        )
        topology = _openmm_topology_from_molecules(
            solute_mol=solute_mol,
            solvent_mol=solvent_mol,
            n_solvent=n_solvent,
            box_length_nm=box_edge_nm,
        )
        pos_nm = [
            Vec3(c[0] / 10.0, c[1] / 10.0, c[2] / 10.0) for c in packed_coords_A
        ]
        modeller = app.Modeller(topology, pos_nm * unit.nanometers)

        if is_water:
            forcefield = app.ForceField("amber/tip3p_standard.xml")
            template_mols = [solute_mol]
        else:
            forcefield = app.ForceField()
            template_mols = [solute_mol, solvent_mol]
        smirnoff_gen = SMIRNOFFTemplateGenerator(
            molecules=template_mols,
            forcefield="openff_unconstrained-2.1.0.offxml",
        )
        forcefield.registerTemplateGenerator(smirnoff_gen.generator)

    cutoff_nm = float(solvent_cfg["nonbonded_cutoff_nm"])
    box_vecs = modeller.topology.getPeriodicBoxVectors()
    min_box_nm = min(
        box_vecs[0][0].value_in_unit(unit.nanometer),
        box_vecs[1][1].value_in_unit(unit.nanometer),
        box_vecs[2][2].value_in_unit(unit.nanometer),
    )
    effective_cutoff_nm = min(cutoff_nm, (min_box_nm / 2.0) - 0.05)

    mm_system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=effective_cutoff_nm * unit.nanometers,
        constraints=None,
        rigidWater=is_water,
    )
    lig_indices = get_atoms_from_resname(modeller.topology, "LIG", "resname")
    assert_no_residue_constraints(mm_system, lig_indices)

    temperature_K = float(spec["temperature_K"])
    pressure_bar = float(spec["pressure_GPa"]) * 1.0e4
    npt_ps = float(solvent_cfg["npt_equilibration_ps"])

    if npt_ps > 0.0:
        mm_system.addForce(
            MonteCarloBarostat(
                pressure_bar * unit.bar, temperature_K * unit.kelvin
            )
        )

    integrator = LangevinMiddleIntegrator(
        temperature_K * unit.kelvin,
        1.0 / unit.picosecond,
        1.0 * unit.femtoseconds,
    )
    simulation = app.Simulation(modeller.topology, mm_system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=1000)

    if npt_ps > 0.0:
        simulation.context.setVelocitiesToTemperature(temperature_K * unit.kelvin)
        n_steps = int(round(npt_ps * 1000.0))
        simulation.step(n_steps)

    final_state = simulation.context.getState(
        getPositions=True, enforcePeriodicBox=True
    )
    eq_vecs = reducePeriodicBoxVectors(final_state.getPeriodicBoxVectors())
    modeller.topology.setPeriodicBoxVectors(eq_vecs)

    expected_indices = list(range(solute_mol.n_atoms))
    if list(lig_indices) != expected_indices:
        raise RuntimeError(
            f"Unexpected solute atom indices {lig_indices}, expected {expected_indices}."
        )

    solvated_pdb.parent.mkdir(parents=True, exist_ok=True)
    with solvated_pdb.open("w", encoding="utf-8") as pdb_f:
        app.PDBFile.writeFile(
            modeller.topology, final_state.getPositions(), file=pdb_f
        )

    prep_meta = {
        "n_total_atoms": int(modeller.topology.getNumAtoms()),
        "n_solute_atoms": int(solute_mol.n_atoms),
        "box_vectors_nm": [
            [float(v.value_in_unit(unit.nanometer)) for v in row]
            for row in eq_vecs
        ],
    }
    Path(spec["paths"]["prep_metadata"]).write_text(
        json.dumps(prep_meta, indent=2), encoding="utf-8"
    )


def _detect_mpi_ranks() -> tuple[int, int | None]:
    """Return (global_rank, local_rank) from standard MPI or Slurm environment variables."""
    global_vars = (
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "PMIX_RANK",
        "SLURM_PROCID",
        "MV2_COMM_WORLD_RANK",
    )
    local_vars = (
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "MV2_COMM_WORLD_LOCAL_RANK",
    )
    global_rank = 0
    for key in global_vars:
        val = os.environ.get(key)
        if val is not None and val.isdigit():
            global_rank = int(val)
            break

    local_rank: int | None = None
    for key in local_vars:
        val = os.environ.get(key)
        if val is not None and val.isdigit():
            local_rank = int(val)
            break
    return global_rank, local_rank


def _bind_local_rank_gpu(local_rank: int | None) -> None:
    """Pin CUDA_VISIBLE_DEVICES to a single GPU per local MPI rank before importing torch."""
    if local_rank is None:
        return
    raw_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_cvd is not None:
        devices = [d.strip() for d in raw_cvd.split(",") if d.strip()]
        if len(devices) > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank % len(devices)]
        return

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return
    proc = subprocess.run(
        [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        gpu_ids = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if len(gpu_ids) > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[local_rank % len(gpu_ids)]


def cmd_repex(spec: dict) -> None:
    """Execute Hamiltonian replica exchange using mace-md."""
    mpi_rank, local_rank = _detect_mpi_ranks()
    _bind_local_rank_gpu(local_rank)

    from mace_md.hybrid_md import MixedSystem, PureSystem
    from mace_md.nnp_repex.repex import (
        MixedSystemConstructor,
        RepexConstructor,
        get_atoms_from_resname,
    )
    from openff.toolkit.topology import Molecule
    from openmm import MonteCarloBarostat, app, unit
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    import torch

    if hasattr(torch._C, "_jit_set_nvfuser_enabled"):
        torch._C._jit_set_nvfuser_enabled(False)

    alch = spec["alchemical"]
    dtype = torch.float32 if alch["dtype"] == "float32" else torch.float64
    torch.set_default_dtype(dtype)

    solvated_pdb = str(spec["paths"]["solvated_pdb"])
    lig_sdf = str(spec["paths"]["lig_sdf"])
    repex_dir = str(spec["paths"]["repex_dir"])
    rank_scratch_dir = os.path.join(repex_dir, f"rank_{mpi_rank}")
    os.makedirs(rank_scratch_dir, exist_ok=True)

    temperature_K = float(spec["temperature_K"])
    pressure_bar = float(spec["pressure_GPa"]) * 1.0e4
    model_path = str(alch["model_path"])
    is_water = spec["solvent"]["solvent"].strip().lower() == "water"
    time_step_fs = float(alch["time_step_fs"])

    if alch["system_type"] == "pure":
        mace_sys = PureSystem(
            file=solvated_pdb,
            ml_mol=solvated_pdb,
            model_path=model_path,
            potential=alch["potential"],
            output_dir=rank_scratch_dir,
            temperature=temperature_K,
            pressure=pressure_bar,
            dtype=dtype,
            decouple=True,
            nl=alch["neighbor_list"],
            max_n_pairs=int(alch["max_n_pairs"]),
            timestep=time_step_fs,
            minimiser="openmm",
            resname="LIG",
            nnpify_type="resname",
            optimized_model=bool(alch["optimized_model"]),
        )
        target_omm_system = mace_sys.system
        target_topology = mace_sys.modeller.topology
        target_positions = mace_sys.modeller.getPositions()
    elif alch["system_type"] == "hybrid" and is_water:
        mace_sys = MixedSystem(
            file=solvated_pdb,
            ml_mol=lig_sdf,
            model_path=model_path,
            forcefields=["amber/tip3p_standard.xml"],
            resname="LIG",
            nnpify_type="resname",
            nonbondedCutoff=float(spec["solvent"]["nonbonded_cutoff_nm"]),
            potential=alch["potential"],
            padding=0.0,
            temperature=temperature_K,
            pressure=pressure_bar,
            dtype=dtype,
            nl=alch["neighbor_list"],
            max_n_pairs=int(alch["max_n_pairs"]),
            output_dir=rank_scratch_dir,
            smff="2.0",
            decouple=False,
            interpolate=True,
            minimiser="openmm",
            mm_only=False,
            water_model="tip3p",
        )
        target_omm_system = mace_sys.system
        target_topology = mace_sys.modeller.topology
        target_positions = mace_sys.modeller.getPositions()
    else:
        pdb_in = app.PDBFile(solvated_pdb)
        solute_mol = Molecule.from_file(lig_sdf)
        solvent_mol = Molecule.from_smiles(
            spec["solvent"]["solvent"].strip(), allow_undefined_stereo=True
        )
        ff = app.ForceField()
        smirnoff_gen = SMIRNOFFTemplateGenerator(
            molecules=[solute_mol, solvent_mol],
            forcefield="openff_unconstrained-2.1.0.offxml",
        )
        ff.registerTemplateGenerator(smirnoff_gen.generator)
        mm_sys = ff.createSystem(
            pdb_in.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=float(spec["solvent"]["nonbonded_cutoff_nm"])
            * unit.nanometers,
            constraints=None,
        )
        mm_sys.addForce(
            MonteCarloBarostat(
                pressure_bar * unit.bar, temperature_K * unit.kelvin
            )
        )
        target_omm_system = MixedSystemConstructor(
            system=mm_sys,
            topology=pdb_in.topology,
            nnpify_id="LIG",
            nnpify_type="resname",
            nnp_potential=alch["potential"],
            model_path=model_path,
            interpolate=True,
            filename=model_path,
            modelPath=model_path,
            dtype=dtype,
            nl=alch["neighbor_list"],
            max_n_pairs=int(alch["max_n_pairs"]),
        ).mixed_system
        target_topology = pdb_in.topology
        target_positions = pdb_in.positions

    repex_nc = os.path.join(repex_dir, "repex.nc")
    restart = bool(alch["restart"]) and os.path.isfile(repex_nc)
    n_replicas = int(alch["n_replicas"])
    equil_proto = alch["equilibration_protocol"]

    sampler = RepexConstructor(
        mixed_system=target_omm_system,
        initial_positions=target_positions,
        intervals_per_lambda_window=2 * n_replicas,
        steps_per_equilibration_interval=int(
            alch["steps_per_equilibration_interval"]
        ),
        equilibration_protocol=equil_proto,
        temperature=temperature_K * unit.kelvin,
        lambda_schedule=[float(x) for x in alch["lambda_schedule"]],
        n_states=n_replicas,
        restart=restart,
        mcmc_moves_kwargs={
            "timestep": time_step_fs * unit.femtoseconds,
            "collision_rate": (10.0 / 1.0) / unit.picoseconds,
            "n_steps": int(alch["steps_per_iteration"]),
            "reassign_velocities": False,
            "n_restart_attempts": 20,
        },
        replica_exchange_sampler_kwargs={
            "number_of_iterations": int(alch["n_iterations"]),
            "online_analysis_interval": int(alch["checkpoint_interval"]),
            "online_analysis_minimum_iterations": min(
                10, int(alch["n_iterations"])
            ),
            "replica_mixing_scheme": alch["replica_mixing_scheme"],
        },
        storage_kwargs={
            "storage": repex_nc,
            "checkpoint_interval": int(alch["checkpoint_interval"]),
            "analysis_particle_indices": get_atoms_from_resname(
                topology=target_topology,
                nnpify_id="LIG",
                nnpify_type="resname",
            ),
        },
    ).sampler

    if not restart and equil_proto == "minimise":
        sampler.minimize()

    sampler.run()


def cmd_analyze(spec: dict) -> None:
    """Extract MBAR solvation free energy, state overlap, and lambda=0 conformer RMSD."""
    import numpy as np
    from openmm import unit
    from openmmtools.multistate import MultiStateReporter, MultiStateSamplerAnalyzer

    repex_nc = str(Path(spec["paths"]["repex_dir"]) / "repex.nc")
    if not os.path.isfile(repex_nc):
        raise FileNotFoundError(f"Replica-exchange output not found: {repex_nc}.")

    temperature_K = float(spec["temperature_K"])
    kt_kJ_mol = float(
        (unit.MOLAR_GAS_CONSTANT_R * (temperature_K * unit.kelvin)).value_in_unit(
            unit.kilojoule_per_mole
        )
    )
    reporter = MultiStateReporter(repex_nc, open_mode="r")
    try:
        try:
            thermo_states, _ = reporter.read_thermodynamic_states()
            lambdas = np.asarray(
                [float(ts.lambda_interpolate) for ts in thermo_states],
                dtype=np.float64,
            )
        except Exception:
            lambdas = np.asarray(
                spec["alchemical"]["lambda_schedule"], dtype=np.float64
            )
        idx_coupled = int(np.argmin(np.abs(lambdas - 1.0)))
        idx_decoupled = int(np.argmin(np.abs(lambdas - 0.0)))

        analyzer = MultiStateSamplerAnalyzer(reporter)
        delta_f_ij, ddelta_f_ij = analyzer.get_free_energy()
        delta_f_arr = np.asarray(delta_f_ij, dtype=np.float64)
        ddelta_f_arr = np.asarray(ddelta_f_ij, dtype=np.float64)

        # Delta_f_ij[i, j] = (G_j - G_i) / kT -> transfer from gas (lambda=0) to solution (lambda=1)
        dg_solv_kJ_mol = float(delta_f_arr[idx_decoupled, idx_coupled]) * kt_kJ_mol
        ddg_solv_kJ_mol = (
            float(ddelta_f_arr[idx_decoupled, idx_coupled]) * kt_kJ_mol
        )

        dg_profile_kJ_mol = [
            float(delta_f_arr[idx_decoupled, k]) * kt_kJ_mol
            for k in range(len(lambdas))
        ]
        ddg_profile_kJ_mol = [
            float(ddelta_f_arr[idx_decoupled, k]) * kt_kJ_mol
            for k in range(len(lambdas))
        ]

        mbar_obj = analyzer.mbar
        if hasattr(mbar_obj, "compute_overlap"):
            overlap_res = mbar_obj.compute_overlap()
        else:
            overlap_res = mbar_obj.computeOverlap()

        if isinstance(overlap_res, dict):
            overlap_mat = np.asarray(overlap_res["matrix"], dtype=np.float64)
        else:
            overlap_mat = np.asarray(overlap_res[0], dtype=np.float64)

        adjacent_overlaps = [
            float(overlap_mat[k, k + 1]) for k in range(len(lambdas) - 1)
        ]
        min_adjacent_overlap = (
            float(min(adjacent_overlaps)) if adjacent_overlaps else 1.0
        )

        # Evaluate solute heavy-atom RMSD in the decoupled (lambda=0) state
        symbols, ref_coords_A = _read_xyz(Path(spec["paths"]["solute_xyz"]))
        heavy_mask = np.array([sym != "H" for sym in symbols], dtype=bool)
        if not np.any(heavy_mask):
            heavy_mask = np.ones(len(symbols), dtype=bool)
        ref_heavy_A = np.asarray(ref_coords_A, dtype=np.float64)[heavy_mask]

        rmsd_values_A: list[float] = []
        nc_storage = reporter._storage_analysis
        if (
            nc_storage is not None
            and "positions" in nc_storage.variables
            and "states" in reporter._storage_checkpoint.variables
        ):
            # analysis positions shape: (n_iterations, n_replicas, n_solute_atoms, 3) in nm
            pos_var = nc_storage.variables["positions"]
            states_var = reporter._storage_checkpoint.variables["states"]
            n_frames = min(pos_var.shape[0], states_var.shape[0])
            start_frame = max(0, n_frames // 2)
            for frame_idx in range(start_frame, n_frames):
                replica_states = np.asarray(states_var[frame_idx])
                matches = np.where(replica_states == idx_decoupled)[0]
                if len(matches) > 0:
                    rep_idx = int(matches[0])
                    frame_A = np.asarray(pos_var[frame_idx, rep_idx], dtype=np.float64) * 10.0
                    if frame_A.shape[0] == len(symbols):
                        rmsd_values_A.append(
                            _kabsch_rmsd(frame_A[heavy_mask], ref_heavy_A)
                        )

        mean_lambda0_rmsd_A = (
            float(np.mean(rmsd_values_A)) if rmsd_values_A else None
        )
    finally:
        reporter.close()

    results = {
        "solvation_free_energy_ben_naim_kJ_mol": dg_solv_kJ_mol,
        "solvation_free_energy_error_kJ_mol": ddg_solv_kJ_mol,
        "lambda_schedule": [float(x) for x in lambdas],
        "dg_profile_kJ_mol": dg_profile_kJ_mol,
        "ddg_profile_kJ_mol": ddg_profile_kJ_mol,
        "overlap_matrix": overlap_mat.tolist(),
        "min_adjacent_overlap": min_adjacent_overlap,
        "rmsd_lambda0_to_gas_A": mean_lambda0_rmsd_A,
    }
    Path(spec["paths"]["results_json"]).write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="External worker entrypoint for mace-md alchemical solvation."
    )
    parser.add_argument(
        "command",
        choices=("check", "prepare", "repex", "analyze"),
        help="Worker phase to execute.",
    )
    parser.add_argument(
        "--spec",
        required=True,
        type=Path,
        help="Path to JSON job specification.",
    )
    args = parser.parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))

    if args.command == "check":
        cmd_check(spec)
    elif args.command == "prepare":
        cmd_prepare(spec)
    elif args.command == "repex":
        cmd_repex(spec)
    elif args.command == "analyze":
        cmd_analyze(spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
