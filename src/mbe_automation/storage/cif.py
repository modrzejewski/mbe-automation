from __future__ import annotations
import iotbx.cif
from cctbx import adptbx

import numpy as np
import numpy.typing as npt
from typing import Tuple
from pathlib import Path

from mbe_automation.storage.core import Structure

def read_cif_with_apds(path: str) -> Tuple[Structure, npt.NDArray[np.float64] | None]:
    """
    Read periodic structure from CIF.

    Extract structure expanded to P1. Retrieve Cartesian ADPs if available.
    
    When ADPs are unspecified for certain atoms, the corresponding entries 
    in `u_cart` are populated with NaNs, preserving the (n_atoms, 3, 3) dimensionality.

    Args:
        path: Path to CIF file.

    Returns:
        tuple: (structure, u_cart)
            u_cart: (n_atoms, 3, 3) Cartesian ADPs in Å², or None.
    """
    try:
        reader = iotbx.cif.reader(file_path=path)
        structures = reader.build_crystal_structures()
    except Exception as e:
        raise ValueError(f"Failed to read CIF: {e}")

    if not structures:
        raise ValueError(f"No structures in {path}")
    
    xray_structure = list(structures.values())[0]
    xs_p1 = xray_structure.expand_to_p1(sites_mod_positive=True)
    
    sites_cart = np.array(xs_p1.sites_cart().as_numpy_array(), dtype=np.float64)
    
    # Masses and atomic numbers using pymatgen
    import pymatgen.core
    
    atomic_numbers_int = []
    masses = []
    
    for sc in xs_p1.scatterers():
        el = pymatgen.core.Element(sc.element_symbol())
        atomic_numbers_int.append(el.Z)
        masses.append(el.atomic_mass)
        
    atomic_numbers_int = np.array(atomic_numbers_int, dtype=np.int64)
    masses = np.array(masses, dtype=np.float64)
    
    unit_cell = xs_p1.unit_cell()
    # cctbx orthogonalization matrix columns are lattice vectors
    cell_vectors = np.array(unit_cell.orthogonalization_matrix(), dtype=np.float64).reshape(3, 3).T
    
    structure = Structure(
        positions=sites_cart,
        atomic_numbers=atomic_numbers_int,
        masses=masses,
        cell_vectors=cell_vectors,
        n_frames=1,
        n_atoms=len(xs_p1.scatterers()),
        periodic=True
    )
    
    u_cart_list = []
    has_adps = False
    
    for sc in xs_p1.scatterers():
        if sc.u_iso != -1.0 and sc.u_star == (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0):
            u_cart_val = adptbx.u_iso_as_u_cart(sc.u_iso)
            has_adps = True
        elif sc.u_star != (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0):
            u_cart_val = adptbx.u_star_as_u_cart(unit_cell, sc.u_star)
            has_adps = True
        else:
            u_cart_val = (np.nan,) * 6
            
        u_tensor = np.array([
            [u_cart_val[0], u_cart_val[3], u_cart_val[4]],
            [u_cart_val[3], u_cart_val[1], u_cart_val[5]],
            [u_cart_val[4], u_cart_val[5], u_cart_val[2]]
        ], dtype=np.float64)
        u_cart_list.append(u_tensor)
        
    if has_adps:
        return structure, np.array(u_cart_list, dtype=np.float64)
    
    return structure, None
