"""Tests for molecular cluster extraction."""

import sys
import tempfile
from pathlib import Path

# Ensure the project root is in sys.path for standalone execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import numpy.testing as npt
import pytest

import mbe_automation
from mbe_automation import (
    MACE,
    Structure,
    UniqueClustersFilter,
)
from mbe_automation.configs.many_body_expansion import Clusters
from mbe_automation.common.display import dotted_separator

DIMER_CUTOFF = 30.0
TRIMER_CUTOFF = 10.0

from tests.reference_data.test_cases import TEST_CASES
TEST_CASES_WITH_MODELS = [c for c in TEST_CASES if c.get("general_model_path") is not None]

def _find_safe_cutoff(distances: np.ndarray, t: float, tolerance: float = 1e-3) -> float:
    """Find a safe cutoff near `t` by locating a gap in the distances of at least `tolerance`.
    Places the cutoff in the middle of the closest valid gap.
    
    Args:
        distances: Array of cluster distances to analyze for gaps.
        t: The target cutoff distance we want to evaluate.
        tolerance: Minimum gap size required to safely place a cutoff (default: 1e-3).
    """
    assert len(distances) >= 2, "Need at least 2 distances to find gaps."
    
    distances = np.sort(distances)
    gaps = np.diff(distances)
    centers = distances[:-1][gaps >= tolerance] + gaps[gaps >= tolerance] / 2.0
    
    # Prefer a gap slightly above t to include clusters exactly on the boundary
    return centers[np.argmin(np.abs(centers - (t + tolerance)))] if len(centers) else t

def _recompute_weights_from_descriptors(
    df_reference: pd.DataFrame, 
    descriptors: dict, 
    desc_tol: float = 1.0e-2,
    dist_tol: float = 1.0e-2
) -> pd.DataFrame:
    """Recompute true symmetry weights by merging legacy clusters with similar physical distances and Coulomb Matrix descriptors."""
    
    descriptors_int = {int(k): v for k, v in descriptors.items()}
    
    valid_indices = []
    desc_list = []
    
    for idx, row in df_reference.iterrows():
        sys_id = int(row['System'])
        if sys_id in descriptors_int:
            valid_indices.append(idx)
            desc_list.append(descriptors_int[sys_id].flatten())
            
    df_merged = df_reference.loc[valid_indices].copy()
    
    if len(df_merged) == 0:
        return pd.DataFrame()
        
    df_merged['Descriptor'] = desc_list
    df_merged = df_merged.sort_values(by='MaxMinRij').reset_index(drop=True)
    
    group_ids = []
    if len(df_merged) > 0:
        active_groups = [(1, df_merged.iloc[0])]
        group_ids.append(1)
        next_group_id = 2
        
        for i in range(1, len(df_merged)):
            row_curr = df_merged.iloc[i]
            matched_group = None
            
            for group_id, group_repr in active_groups:
                phys_dist = max(
                    abs(row_curr['SumAvRij'] - group_repr['SumAvRij']),
                    abs(row_curr['MaxMinRij'] - group_repr['MaxMinRij']),
                    abs(row_curr['MaxCOMRij'] - group_repr['MaxCOMRij'])
                )
                
                if phys_dist <= dist_tol:
                    desc_repr = group_repr['Descriptor']
                    desc_curr = row_curr['Descriptor']
                    desc_dist = np.linalg.norm(desc_curr - desc_repr)
                    
                    if desc_dist <= desc_tol:
                        matched_group = group_id
                        break
                        
            if matched_group is not None:
                group_ids.append(matched_group)
            else:
                group_ids.append(next_group_id)
                active_groups.append((next_group_id, row_curr))
                next_group_id += 1
            
    df_merged['Desc_Group'] = group_ids
    df_merged = df_merged.drop(columns=['Descriptor'])
        
    df_desc_based = df_merged.groupby('Desc_Group').agg({
        'Weight': 'sum',
        'SumAvRij': 'mean',
        'MaxMinRij': 'mean',
        'MaxCOMRij': 'mean'
    }).reset_index(drop=True)
    
    df_desc_based = df_desc_based.sort_values(by='MaxMinRij').reset_index(drop=True)
    return df_desc_based


def _assert_cumulative_weights_comparison(
    df_ref1: pd.DataFrame, 
    df_ref2: pd.DataFrame, 
    df_this: pd.DataFrame, 
    max_cutoff: float, 
    cluster_type: str, 
    system_name: str,
    distance_column: str
) -> None:
    """Assert physical sums match and unique clusters satisfy the new passing criteria."""
    
    thresholds = np.arange(0.0, max_cutoff + 0.5, 0.5)
    
    print(f"\nSystem: {system_name} | Cluster: {cluster_type}")
    dotted_separator(120)
    print(f"{'cutoff (Å)':<12} | {'n_unique(ref 1)':<17} {'n_unique(ref 2)':<17} {'n_unique(this)':<17} {'Status':<8}")
    dotted_separator(120)
    
    ref_distances = df_ref1['MaxMinRij'].to_numpy()
    
    for t in thresholds:
        safe_t = _find_safe_cutoff(ref_distances, t, tolerance=1e-3)
        
        ref1_subset = df_ref1[df_ref1['MaxMinRij'] <= safe_t]
        this_subset = df_this[df_this[distance_column] <= safe_t]
        
        ref1_phys = int(ref1_subset['Weight'].sum())
        this_phys = int(this_subset['cluster_count'].sum())
        
        ref1_uniq = len(ref1_subset)
        this_uniq = len(this_subset)
        
        ref2_subset = df_ref2[df_ref2['MaxMinRij'] <= safe_t]
        ref2_uniq = len(ref2_subset)
            
        phys_match = (ref1_phys == this_phys)
        
        passed = True
        if not phys_match:
            passed = False
        if not (ref2_uniq <= this_uniq <= ref1_uniq):
            passed = False
            
        status = "PASS" if passed else "FAIL"
        
        print(f"{t:<12.1f} | {ref1_uniq:<17} {ref2_uniq:<17} {this_uniq:<17} {status:<8}")
        
        assert phys_match, f"Physical count mismatch at {t} A: {ref1_phys} vs {this_phys}"
        assert ref2_uniq <= this_uniq <= ref1_uniq, f"Current code unique clusters ({this_uniq}) not in expected bounds [{ref2_uniq}, {ref1_uniq}] at {t} A"
            
    dotted_separator(120)


def _verify_symmetry_weights(case: dict, work_dir: Path) -> None:
    """Verify generated symmetry weights against reference data.
    
    Args:
        case: Dictionary containing system paths and reference paths.
        work_dir: Directory containing the generated output CSVs.
    """
    csv_dir = work_dir / "csv"
    assert csv_dir.exists(), f"CSV directory missing: {csv_dir}"
    
    cluster_configs = [
        ("dimers", DIMER_CUTOFF, "dimers[AA].csv", "min_r (Å)"),
        ("trimers", TRIMER_CUTOFF, "trimers[AAA].csv", "max_min_r (Å)")
    ]
    
    for cluster_type, cutoff, csv_name, dist_col in cluster_configs:
        generated_csv_path = csv_dir / csv_name
        legacy_csv_path = case["symmetry_weights"].get(cluster_type)
        
        if generated_csv_path.exists() and legacy_csv_path and Path(legacy_csv_path).exists():
            df_generated = pd.read_csv(generated_csv_path)
            df_reference = pd.read_csv(legacy_csv_path, skipinitialspace=True)
            
            desc_npz_path = case["descriptors"][cluster_type]
            descriptors_data = np.load(desc_npz_path)
            df_desc_based = _recompute_weights_from_descriptors(df_reference, descriptors_data)
                
            _assert_cumulative_weights_comparison(
                df_ref1=df_reference,
                df_ref2=df_desc_based,
                df_this=df_generated,
                max_cutoff=cutoff,
                cluster_type=cluster_type,
                system_name=case["name"],
                distance_column=dist_col
            )

def _run_cluster_extraction(case: dict, work_dir: Path) -> None:
    """Execute MBE cluster extraction pipeline for a given test case.
    
    Args:
        case: Dictionary containing system paths and reference paths.
        work_dir: Output directory for the generated clusters.
    """
    crystal = Structure.from_xyz_file(case["crystal_path"])
    calc = MACE(model_path=str(case["general_model_path"])) 

    unique_cluster_filter = UniqueClustersFilter(
        cluster_types=["monomers", "dimers", "trimers"],
        cutoffs={"dimers": DIMER_CUTOFF, "trimers": TRIMER_CUTOFF},
    )

    config = Clusters(
        crystal=crystal,
        calculator=calc,
        filter=unique_cluster_filter,
        work_dir=work_dir,
        dataset=work_dir / "properties.hdf5",
        save_xyz=True,
        save_csv=True,
    )

    mbe_automation.run(config)
    _verify_symmetry_weights(case, work_dir)

@pytest.mark.parametrize("case", TEST_CASES_WITH_MODELS, ids=[c["name"] for c in TEST_CASES_WITH_MODELS])
def test_extract_unique_clusters(case: dict, tmp_path: Path):
    """Test full unique cluster extraction pipeline against golden references."""
    _run_cluster_extraction(case, tmp_path)

if __name__ == "__main__":
    for case in TEST_CASES_WITH_MODELS:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                _run_cluster_extraction(case, Path(temp_dir))
        except AssertionError as e:
            print(f"\n{e}")
            sys.exit(1)
