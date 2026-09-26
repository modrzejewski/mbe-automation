with open("report.md", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "## Implementation Review" in line:
        insert_idx = i + 1
        break

addition = """
### Note on Coordinate Systems and Cell Axes Orthogonality

A critical aspect of the new logic is its handling of non-orthogonal cell axes. The calculation of the phase shift evaluates `mapped_pos @ G`.
* `mapped_pos`: These are the atomic positions in **fractional** coordinates (obtained from `positions[perm]` where `positions = ph.primitive.scaled_positions`).
* `G`: This is the reciprocal translation vector, which is calculated as `q_vec @ np.linalg.inv(r) - q_vec`. Since `q_vec` is provided in fractional reciprocal coordinates, $G$ is also a vector of integers representing fractional reciprocal shifts.

Because both `mapped_pos` and `G` are in their respective fractional bases (real space and reciprocal space), their dot product `mapped_pos @ G` correctly yields a dimensionless scalar proportional to the phase, entirely independent of the real-space orthogonality or metric tensor of the unit cell. Therefore, the logic is robust and mathematically valid for arbitrary, non-orthogonal unit cells (e.g., triclinic, monoclinic, hexagonal).
"""

lines.insert(insert_idx, addition)

with open("report.md", "w") as f:
    f.writelines(lines)
