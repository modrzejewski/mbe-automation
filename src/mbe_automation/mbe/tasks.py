from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np
import numpy.typing as npt

import mbe_automation.calculators.electronic.core

if TYPE_CHECKING:
    from mbe_automation.mbe.core import MBEMetadata


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
            in Å. NaN for monomers.
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


@dataclass
class ClusterSelection:
    """
    Fluent builder for selecting clusters and generating scheduled tasks.

    Created by ``MBEMetadata.select()``. Use ``.below()`` to apply
    an optional distance cutoff, then ``.schedule()`` to generate tasks.
    """
    _mbe: MBEMetadata
    _cluster_type: str
    _max_distance: float | None = field(default=None, repr=False)

    def below(self, distance: float) -> ClusterSelection:
        """
        Restrict to clusters with characteristic distance < ``distance`` (Å).
        """
        return ClusterSelection(
            _mbe=self._mbe,
            _cluster_type=self._cluster_type,
            _max_distance=distance,
        )

    def schedule(self, method: str) -> list[ScheduledTask]:
        """
        Generate scheduled tasks for the selected clusters.

        Args:
            method: Quantum-chemical model identifier.

        Returns:
            List of ``ScheduledTask`` objects.
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

        tasks: list[ScheduledTask] = []
        for cluster_type in matching_types:
            clusters = self._mbe.read_clusters(cluster_type=cluster_type)
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
