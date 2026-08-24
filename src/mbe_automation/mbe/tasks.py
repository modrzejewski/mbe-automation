from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from collections import UserList, defaultdict
from pathlib import Path
import numpy as np
import numpy.typing as npt

import mbe_automation.calculators.electronic.core
import mbe_automation.calculators.electronic.mrcc
import mbe_automation.calculators.electronic.slurm

if TYPE_CHECKING:
    from mbe_automation.mbe.core import MBE
    from mbe_automation.structure.clusters import UniqueClusters

#
# Directory path relative to the project's working directory where input files
# for quantum-chemical software are stored and calculations are performed.
#
_TASKS_DIR = Path("tasks")


@dataclass
class ScheduledTask:
    """
    A single quantum chemistry computation for one (sub)system of a cluster.

    Attributes:
        cluster_label: Unique label of the parent cluster,
            e.g. "02-dimer[AA]-a1b2c3d4e5f6g7h8".
        method: Quantum-chemical model,
            e.g. "lno-ccsd(t)_tight_avqz".
        input_string: Complete input for the quantum chemistry program.
        cluster_type: Cluster type identifier,
            e.g. "dimers[AA]".
        characteristic_distance: Characteristic intermolecular distance
            in Å. NaN for monomers.
        subsystem_label: Bitmask label for the subsystem,
            e.g. "11", "10", "01". None when the calculator
            produces a single input per cluster.
    """
    cluster_label: str
    method: str
    input_string: str
    cluster_type: str
    characteristic_distance: np.float64
    subsystem_label: str | None

    @property
    def quantum_chemical_code(self) -> str:
        return mbe_automation.calculators.electronic.core.SOFTWARE_MAP[self.method]

    @property
    def needs_individual_dir(self) -> bool:
        """
        True if the method requires a dedicated subdirectory for each task,
        even when a subsystem_label is not present.
        """
        return self.quantum_chemical_code == "mrcc"

    @property
    def directory(self) -> Path:
        """
        Return the directory path for the task, relative to the
        project's work dir (e.g. `tasks/lno-ccsd(t)_vtight_avqz/dimers[AA]/...`).
        """
        dir = _TASKS_DIR / self.method / self.cluster_type
        if self.needs_individual_dir:
            dir = dir / self.cluster_label
            if self.subsystem_label is not None:
                dir = dir / self.subsystem_label
        return dir

    @property
    def input_file(self) -> Path:
        """
        Return the exact file path for the quantum chemistry input,
        relative to the project's work dir.
        """
        if self.quantum_chemical_code == "mrcc":
            return self.directory / "MINP"
        else:
            return self.directory / f"{self.cluster_label}.inp"


class Tasks(UserList[ScheduledTask]):
    """
    Collection of ScheduledTask objects.
    """
    
    @property
    def methods(self) -> list[str]:
        """List of unique methods present in the tasks, preserving order."""
        return list(dict.fromkeys(task.method for task in self.data))
        
    @property
    def cluster_types(self) -> list[str]:
        """List of unique cluster types present in the tasks, preserving order."""
        return list(dict.fromkeys(task.cluster_type for task in self.data))
        
    def filter_by(
        self,
        method: str | None = None,
        cluster_type: str | None = None,
        min_distance: float | None = None,
        max_distance: float | None = None,
    ) -> Tasks:
        """
        Return a sub-collection of Tasks filtered by the specified criteria.

        Args:
            method: Quantum-chemical model (e.g., 'lno-ccsd(t)_tight_avqz').
            cluster_type: Cluster identifier (e.g., 'monomers[A]', 'dimers[AA]').
            min_distance: Minimum characteristic distance in Å. Excludes monomers if provided.
            max_distance: Maximum characteristic distance in Å. Excludes monomers if provided.

        Returns:
            Tasks: Filtered collection.
        """
        filtered = self.data
        if method is not None:
            filtered = [t for t in filtered if t.method == method]
        if cluster_type is not None:
            filtered = [t for t in filtered if t.cluster_type == cluster_type]
            
        if min_distance is not None:
            filtered = [
                t for t in filtered 
                if not np.isnan(t.characteristic_distance) and t.characteristic_distance >= min_distance
            ]
        if max_distance is not None:
            filtered = [
                t for t in filtered 
                if not np.isnan(t.characteristic_distance) and t.characteristic_distance < max_distance
            ]
            
        return Tasks(filtered)
    
    def _split_by_distance(self, distance_threshold: float) -> tuple[Tasks, Tasks]:
        """
        Split tasks into two collections based on a characteristic distance threshold.
        Monomers (NaN distance) are included in the first collection.

        Args:
            distance_threshold: Distance in Å. Tasks strictly below this value go 
                to the first collection, others go to the second.
                
        Returns:
            tuple[Tasks, Tasks]: (Tasks below threshold, Tasks at or above threshold)
        """
        below = []
        above = []
        for t in self.data:
            if np.isnan(t.characteristic_distance) or t.characteristic_distance < distance_threshold:
                below.append(t)
            else:
                above.append(t)
                
        return Tasks(below), Tasks(above)


    def _distance_bins(
        self,
        cluster_type: str,
        distance_increment: float = 2.0,
        initial_distance: float = 4.0,
    ) -> dict[tuple[float, float], Tasks]:
        """
        Group tasks of a specific cluster type by characteristic distance.
        
        Args:
            cluster_type: Exact cluster topology identifier (e.g., 'dimers[AA]').
            distance_increment: The size of the distance bins for characteristic distances
                greater than the initial_distance.
            initial_distance: The upper bound for the first distance bin. 

        Returns: 
            dict[tuple[float, float], Tasks]: Mapping of bin boundaries to task subsets.
                The upper bound of the last bin is represented as np.inf.
        """
        if cluster_type.startswith("monomers"):
            raise ValueError("Distance binning cannot be applied to monomers.")
            
        grouped_tasks = {}
        
        remaining = self.filter_by(cluster_type=cluster_type)
        
        lower = 0.0
        upper = initial_distance
        while len(remaining) > 0:
            bin, remaining = remaining._split_by_distance(upper)
            
            if len(bin) > 0:
                upper_bound = np.inf if len(remaining) == 0 else upper
                grouped_tasks[(lower, upper_bound)] = bin
            
            lower = upper
            upper = lower + distance_increment
            
        return grouped_tasks

    def to_input_files(
        self,
        work_dir: str | Path,
        queue: str | None = None,
    ) -> None:
        """
        Export scheduled tasks to input files and generate SLURM array scripts.
        """
        work_dir = Path(work_dir).expanduser()
        
        self._to_quantum_chemical_inputs(work_dir)
        self._to_slurm_scripts(work_dir, queue)

    def _to_quantum_chemical_inputs(
        self,
        work_dir: Path,
    ) -> None:
        """
        Export scheduled tasks to input files.
        
        For MRCC methods, files are saved as MINP within a subsystem subdirectory
        or directly in the cluster directory if no subsystems are present.
        For other methods, files are saved as {cluster_label}.inp.
        """
        for task in self.data:
            file_path = work_dir / task.input_file
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(task.input_string, encoding="utf-8")

    def _to_slurm_scripts(
        self,
        work_dir: Path,
        queue: str | None = None,
    ) -> None:
        """
        Export native SLURM job array scripts grouped by method.
        """
        for method in self.methods:
            method_dir = work_dir / _TASKS_DIR / method
            method_tasks = self.filter_by(method=method)
            
            grouped_paths = {}
            for cluster_type in method_tasks.cluster_types:
                if cluster_type.startswith("monomers"):
                    subset = method_tasks.filter_by(cluster_type=cluster_type)
                    grouped_paths[cluster_type] = [
                        str(task.input_file.relative_to(_TASKS_DIR / method))
                        for task in subset
                    ]
                else:
                    bins = method_tasks._distance_bins(
                        cluster_type=cluster_type,
                        distance_increment=2.0,
                        initial_distance=4.0,
                    )
                    for (lower, upper), subset in bins.items():
                        upper_str = "max cutoff" if np.isinf(upper) else f"{upper:.2f}"
                        group_name = f"{cluster_type} ({lower:.2f} ≤ r < {upper_str} Å)"
                        grouped_paths[group_name] = [
                            str(task.input_file.relative_to(_TASKS_DIR / method))
                            for task in subset
                        ]
            
            files = mbe_automation.calculators.electronic.slurm.to_input_string(
                method=method,
                grouped_tasks=grouped_paths,
                queue=queue,
            )
            
            for filename, content in files.items():
                file_path = method_dir / filename
                file_path.write_text(content, encoding="utf-8")


@dataclass
class ClusterSelection:
    """
    Fluent builder for selecting clusters and generating scheduled tasks.

    Created by ``MBE.select()``. Use ``.below()`` to apply
    an optional distance cutoff, then ``.schedule()`` to generate tasks.
    """
    _mbe: MBE
    _cluster_type: str
    _max_distance: float | None = field(default=None, repr=False)

    def below(self, distance: float) -> ClusterSelection:
        """
        Restrict to clusters with characteristic distance < ``distance`` (Å).
        """
        if self._cluster_type == "monomers":
            raise ValueError("Distance cutoff cannot be applied to monomers.")
        return ClusterSelection(
            _mbe=self._mbe,
            _cluster_type=self._cluster_type,
            _max_distance=distance,
        )

    def _assert_distance_below_cutoff(
        self,
        clusters: UniqueClusters,
    ) -> None:
        if self._max_distance is None or clusters.type_string == "monomers":
            return
        if self._max_distance > self._mbe.filter.cutoffs[clusters.type_string]:
            raise ValueError(
                f"Requested distance cutoff {self._max_distance} Å is larger "
                f"than the generation cutoff ({self._mbe.filter.cutoffs[clusters.type_string]} Å) for \"{clusters.type_string}\"."
            )

    def schedule(self, method: str) -> Tasks:
        """
        Generate scheduled tasks for the selected clusters.

        Args:
            method: Quantum-chemical model identifier.

        Returns:
            Collection of ``ScheduledTask`` objects.
        """
        if method not in mbe_automation.calculators.electronic.core.METHODS:
            raise ValueError(
                f"Invalid electronic method: '{method}'. "
                f"Supported methods are: "
                f"{', '.join(mbe_automation.calculators.electronic.core.METHODS)}"
            )

        matching_types = [
            ct for ct in self._mbe.cluster_types
            if ct.startswith(self._cluster_type)
        ]
        if not matching_types:
            raise ValueError(
                f"No cluster types matching '{self._cluster_type}' "
                f"in MBE metadata. "
                f"Available types: {self._mbe.cluster_types}"
            )

        tasks = Tasks()
        for cluster_type in matching_types:
            clusters = self._mbe.read_clusters(cluster_type=cluster_type)
            self._assert_distance_below_cutoff(clusters)
            mol_sizes = clusters.molecule_sizes
            char_distances = clusters.characteristic_distances
            is_monomer = (clusters.n_molecules == 1)

            for i in range(clusters.n_clusters_unique):
                if is_monomer:
                    distance = np.nan
                else:
                    distance = np.float64(char_distances[i])
                    if (self._max_distance is not None
                            and distance >= self._max_distance):
                        continue

                cluster_label = clusters.labels(frame_index=i)[0]
                inputs = mbe_automation.calculators.electronic.core.to_input_string(
                    method=method,
                    structure=clusters.structures,
                    subsystem_sizes=mol_sizes,
                    frame_index=i,
                )
                for subsystem_label, input_string in inputs.items():
                    tasks.append(ScheduledTask(
                        cluster_label=cluster_label,
                        method=method,
                        input_string=input_string,
                        cluster_type=cluster_type,
                        characteristic_distance=distance,
                        subsystem_label=subsystem_label,
                    ))
        return tasks
