# Transformation of the Dynamical Matrix under Spatial Symmetry

This document derives the transformation properties of the dynamical matrix $D(\mathbf{q})$ under a spatial point group symmetry operation $S$ belonging to the space group of a crystal.

## 1. Transformation of Atomic Displacements

Consider a crystal with equilibrium atomic positions defined by $\mathbf{x}(\kappa_j, \mathbf{R}) = \mathbf{R} + \mathbf{r}_{\kappa_j}$, where $\mathbf{R}$ is the Bravais lattice vector and $\kappa_j$ is the basis index.

A spatial symmetry operation $S$ maps the equilibrium position to a symmetrically equivalent site, where the lattice vector $\mathbf{R}$ is transformed to $S\mathbf{R}$ and the basis index $\kappa_j$ is permuted to $S(\kappa_j)$:

$$S\mathbf{x}(\kappa_j, \mathbf{R}) = \mathbf{x}(S(\kappa_j), S\mathbf{R})$$

Let $\mathbf{u}(\kappa_j, \mathbf{R})$ be the displacement of the atom from its equilibrium position. The instantaneous position of the atom in the original (unprimed) coordinate system is:

$$\mathbf{r}(\kappa_j, \mathbf{R}) = \mathbf{x}(\kappa_j, \mathbf{R}) + \mathbf{u}(\kappa_j, \mathbf{R})$$

Applying the linear operator $S$ to the entire system yields the instantaneous position in the transformed (primed) coordinate system. We define the displacements in this new system as $\mathbf{u}'$:

$$S\mathbf{r}(\kappa_j, \mathbf{R}) = \mathbf{x}(S(\kappa_j), S\mathbf{R}) + \mathbf{u}'(S(\kappa_j), S\mathbf{R})$$

Equating the expansion of $S\mathbf{r}(\kappa_j, \mathbf{R})$ with the primed definition isolates the relation between the displacement vectors:

$$S\mathbf{x}(\kappa_j, \mathbf{R}) + S\mathbf{u}(\kappa_j, \mathbf{R}) = \mathbf{x}(S(\kappa_j), S\mathbf{R}) + \mathbf{u}'(S(\kappa_j), S\mathbf{R})$$

$$S\mathbf{u}(\kappa_j, \mathbf{R}) = \mathbf{u}'(S(\kappa_j), S\mathbf{R})$$

Multiplying by the inverse operator $S^{-1} = S^T$ provides the unprimed displacements in terms of the primed displacements:

$$\mathbf{u}(\kappa_j, \mathbf{R}) = S^{-1}\mathbf{u}'(S(\kappa_j), S\mathbf{R})$$

To avoid index collision with the basis atoms $\kappa_i, \kappa_j$, we use Greek letters for Cartesian index notation ($\alpha, \beta \in \{x, y, z\}$):

$$u_\alpha(\kappa_j, \mathbf{R}) = \sum_\beta S_{\beta\alpha} u'_\beta(S(\kappa_j), S\mathbf{R})$$

The partial derivative mapping the two coordinate systems is therefore:

$$\frac{\partial u_\alpha(\kappa_j, \mathbf{R})}{\partial u'_\beta(S(\kappa_j), S\mathbf{R})} = S_{\beta\alpha}$$

## 2. Transformation of the Force Constant Matrix

The harmonic force constant matrix $\Phi$ is defined as the second derivative of the invariant potential energy $V$ with respect to the atomic displacements. In the unprimed system, for two basis atoms $\kappa_i$ and $\kappa_j$:

$$\Phi_{\alpha\beta}(\kappa_i, \kappa_j | \mathbf{R}) = \frac{\partial^2 V}{\partial u_\alpha(\kappa_i, \mathbf{0}) \partial u_\beta(\kappa_j, \mathbf{R})}$$

We evaluate the force constant matrix in the transformed (primed) coordinate system, denoted $\Phi'$, using the chain rule and dummy Cartesian indices $\gamma, \lambda$:

$$\Phi'_{\gamma\lambda}(S(\kappa_i), S(\kappa_j) | S\mathbf{R}) = \frac{\partial^2 V}{\partial u'_\gamma(S(\kappa_i), \mathbf{0}) \partial u'_\lambda(S(\kappa_j), S\mathbf{R})}$$

$$\Phi'_{\gamma\lambda}(S(\kappa_i), S(\kappa_j) | S\mathbf{R}) = \sum_{\alpha,\beta} \frac{\partial u_\alpha(\kappa_i, \mathbf{0})}{\partial u'_\gamma(S(\kappa_i), \mathbf{0})} \frac{\partial^2 V}{\partial u_\alpha(\kappa_i, \mathbf{0}) \partial u_\beta(\kappa_j, \mathbf{R})} \frac{\partial u_\beta(\kappa_j, \mathbf{R})}{\partial u'_\lambda(S(\kappa_j), S\mathbf{R})}$$

Substitute the partial derivatives from Section 1 and recognize the definition of the unprimed force constant matrix:

$$\Phi'_{\gamma\lambda}(S(\kappa_i), S(\kappa_j) | S\mathbf{R}) = \sum_{\alpha,\beta} S_{\gamma\alpha} \Phi_{\alpha\beta}(\kappa_i, \kappa_j | \mathbf{R}) S_{\lambda\beta}$$

In tensor notation, this is $\Phi' = S \Phi S^T$. Because $S$ is a symmetry operation of the crystal, the force constants of the transformed crystal ($\Phi'$) must be identical to the exact force constants of the original crystal evaluated at the permuted sites ($\Phi$). Therefore:

$$\Phi(S(\kappa_i), S(\kappa_j) | S\mathbf{R}) = S \Phi(\kappa_i, \kappa_j | \mathbf{R}) S^{-1}$$

## 3. Transformation of the Dynamical Matrix

Following the Phonopy convention, the dynamical matrix block connecting basis atoms $\kappa_i$ and $\kappa_j$ at wavevector $\mathbf{q}$ is:

$$D(\kappa_i, \kappa_j | \mathbf{q}) = \frac{1}{\sqrt{m_{\kappa_i} m_{\kappa_j}}} \sum_{\mathbf{R}} \Phi(\kappa_i, \kappa_j | \mathbf{R}) e^{i \mathbf{q} \cdot (\mathbf{x}(\kappa_j, \mathbf{R}) - \mathbf{x}(\kappa_i, \mathbf{0}))}$$

Apply the similarity transformation $S$ to the dynamical matrix. Given that basis atom masses are invariant under crystal symmetry operations ($m_{S(\kappa)} = m_\kappa$), $S$ operates solely on the force constant tensor:

$$S D(\kappa_i, \kappa_j | \mathbf{q}) S^{-1} = \frac{1}{\sqrt{m_{S(\kappa_i)} m_{S(\kappa_j)}}} \sum_{\mathbf{R}} \left[ S \Phi(\kappa_i, \kappa_j | \mathbf{R}) S^{-1} \right] e^{i \mathbf{q} \cdot (\mathbf{x}(\kappa_j, \mathbf{R}) - \mathbf{x}(\kappa_i, \mathbf{0}))}$$

Substitute the relation derived in Section 2:

$$S D(\kappa_i, \kappa_j | \mathbf{q}) S^{-1} = \frac{1}{\sqrt{m_{S(\kappa_i)} m_{S(\kappa_j)}}} \sum_{\mathbf{R}} \Phi(S(\kappa_i), S(\kappa_j) | S\mathbf{R}) e^{i \mathbf{q} \cdot (\mathbf{x}(\kappa_j, \mathbf{R}) - \mathbf{x}(\kappa_i, \mathbf{0}))}$$

Rewrite the argument of the phase factor using the invariance of the scalar product under orthogonal transformations, $(S\mathbf{q}) \cdot (S\mathbf{b}) = \mathbf{a} \cdot \mathbf{b}$, and the coordinate mapping definitions:

$$\mathbf{q} \cdot (\mathbf{x}(\kappa_j, \mathbf{R}) - \mathbf{x}(\kappa_i, \mathbf{0})) = (S\mathbf{q}) \cdot (S\mathbf{x}(\kappa_j, \mathbf{R}) - S\mathbf{x}(\kappa_i, \mathbf{0}))$$

$$= (S\mathbf{q}) \cdot (\mathbf{x}(S(\kappa_j), S\mathbf{R}) - \mathbf{x}(S(\kappa_i), \mathbf{0}))$$

Substitute this back into the summation:

$$S D(\kappa_i, \kappa_j | \mathbf{q}) S^{-1} = \frac{1}{\sqrt{m_{S(\kappa_i)} m_{S(\kappa_j)}}} \sum_{\mathbf{R}} \Phi(S(\kappa_i), S(\kappa_j) | S\mathbf{R}) e^{i (S\mathbf{q}) \cdot (\mathbf{x}(S(\kappa_j), S\mathbf{R}) - \mathbf{x}(S(\kappa_i), \mathbf{0}))}$$

Perform a change of variables in the summation. Define the transformed lattice vector $\mathbf{R}' = S\mathbf{R}$. Summing over all $\mathbf{R}$ is identical to summing over all $\mathbf{R}'$:

$$S D(\kappa_i, \kappa_j | \mathbf{q}) S^{-1} = \frac{1}{\sqrt{m_{S(\kappa_i)} m_{S(\kappa_j)}}} \sum_{\mathbf{R}'} \Phi(S(\kappa_i), S(\kappa_j) | \mathbf{R}') e^{i (S\mathbf{q}) \cdot (\mathbf{x}(S(\kappa_j), \mathbf{R}') - \mathbf{x}(S(\kappa_i), \mathbf{0}))}$$

The right-hand side is exactly the definition of the dynamical matrix block evaluated for the sites $S(\kappa_i)$ and $S(\kappa_j)$ at the transformed wavevector $S\mathbf{q}$:

$$S D(\kappa_i, \kappa_j | \mathbf{q}) S^{-1} = D(S(\kappa_i), S(\kappa_j) | S\mathbf{q})$$

## 4. Symmetrization using the Little Group

The little group $G_{\mathbf{q}}$ is the subgroup of the crystal's point group containing operations $S$ that leave the wavevector $\mathbf{q}$ invariant up to a reciprocal lattice vector $\mathbf{G}$:

$$S\mathbf{q} = \mathbf{q} + \mathbf{G}$$

Substitute this transformation into the definition of the dynamical matrix evaluated at $S\mathbf{q}$:

$$D(S(\kappa_i), S(\kappa_j) | \mathbf{q} + \mathbf{G}) = \frac{1}{\sqrt{m_{S(\kappa_i)} m_{S(\kappa_j)}}} \sum_{\mathbf{R}'} \Phi(S(\kappa_i), S(\kappa_j) | \mathbf{R}') e^{i (\mathbf{q} + \mathbf{G}) \cdot (\mathbf{R}' + \mathbf{r}_{S(\kappa_j)} - \mathbf{r}_{S(\kappa_i)})}$$

Expand the dot product in the exponential:

$$e^{i (\mathbf{q} + \mathbf{G}) \cdot (\mathbf{R}' + \mathbf{r}_{S(\kappa_j)} - \mathbf{r}_{S(\kappa_i)})} = e^{i \mathbf{q} \cdot (\mathbf{R}' + \mathbf{r}_{S(\kappa_j)} - \mathbf{r}_{S(\kappa_i)})} e^{i \mathbf{G} \cdot \mathbf{R}'} e^{i \mathbf{G} \cdot (\mathbf{r}_{S(\kappa_j)} - \mathbf{r}_{S(\kappa_i)})}$$

The first term is exactly the phase factor required to reconstruct $D(S(\kappa_i), S(\kappa_j) | \mathbf{q})$. Because $\mathbf{G}$ is a reciprocal lattice vector and $\mathbf{R}'$ is a Bravais lattice vector, their dot product is an integer multiple of $2\pi$, meaning $e^{i \mathbf{G} \cdot \mathbf{R}'} = 1$. The expression simplifies to:

$$D(S(\kappa_i), S(\kappa_j) | S\mathbf{q}) = D(S(\kappa_i), S(\kappa_j) | \mathbf{q}) e^{i\mathbf{G} \cdot (\mathbf{r}_{S(\kappa_j)} - \mathbf{r}_{S(\kappa_i)})}$$

Equating this with the transformation derived in Section 3 yields a relation between $3 \times 3$ blocks evaluated at the exact same wavevector $\mathbf{q}$:

$$S D(\kappa_i, \kappa_j | \mathbf{q}) S^{-1} = D(S(\kappa_i), S(\kappa_j) | \mathbf{q}) e^{i\mathbf{G} \cdot (\mathbf{r}_{S(\kappa_j)} - \mathbf{r}_{S(\kappa_i)})}$$

To correct numerical deviations in a computed dynamical matrix, this exact symmetry is enforced by averaging over all $N_q$ operations in the little group. Isolate the target block $D(\kappa_i, \kappa_j | \mathbf{q})$ by applying $S^{-1}$ from the left and $S$ from the right:

$$D(\kappa_i, \kappa_j | \mathbf{q}) = S^{-1} \left[ D(S(\kappa_i), S(\kappa_j) | \mathbf{q}) e^{i\mathbf{G} \cdot (\mathbf{r}_{S(\kappa_j)} - \mathbf{r}_{S(\kappa_i)})} \right] S$$

The fully symmetrized block $D^{\text{sym}}(\kappa_i, \kappa_j | \mathbf{q})$ is the group average of the rotationally transformed blocks connecting the permuted atoms:

$$D^{\text{sym}}(\kappa_i, \kappa_j | \mathbf{q}) = \frac{1}{N_q} \sum_{S \in G_{\mathbf{q}}} S^{-1} \left[ D(S(\kappa_i), S(\kappa_j) | \mathbf{q}) e^{i\mathbf{G} \cdot (\mathbf{r}_{S(\kappa_j)} - \mathbf{r}_{S(\kappa_i)})} \right] S$$

Because $S$ is an orthogonal transformation in Cartesian coordinates, $S^{-1} = S^T$. The equation evaluates a purely $3 \times 3$ matrix multiplication for each symmetry operation.

Here, the basis indices $\kappa_i$ and $\kappa_j$ strictly label the $n_{\text{primitive}}$ atoms within the same primitive unit cell. The symmetry mapping $S(\kappa)$ permutes each atom to its equivalent counterpart within this primitive basis, ensuring the full symmetrized dynamical matrix $\mathbf{D}^{\text{sym}}(\mathbf{q})$ is a $3n_{\text{primitive}} \times 3n_{\text{primitive}}$ matrix.

For wavevectors $\mathbf{q}$ strictly inside the first Brillouin zone, operations in the little group map the wavevector onto itself without requiring a reciprocal lattice shift ($\mathbf{G} = \mathbf{0}$). In this case, the $\mathbf{G}$-dependent phase factor reduces exactly to $1$. The symmetrization simplifies to a pure rotation and block permutation:

$$D^{\text{sym}}(\kappa_i, \kappa_j | \mathbf{q}) = \frac{1}{N_q} \sum_{S \in G_{\mathbf{q}}} S^T D(S(\kappa_i), S(\kappa_j) | \mathbf{q}) S$$

The $\mathbf{G}$-dependent phase correction is only required for points on the Brillouin zone boundary, where $S\mathbf{q}$ maps to an equivalent point on the opposite face of the zone.

## 5. Alternative Phase Expression in Fractional Coordinates

When using the **crystallographic (fractional) coordinate system** (for example, as implemented internally by software like Phonopy), positions and reciprocal space vectors are scaled differently relative to Cartesian space:

*   **Fractional Real Space**: Atomic positions are given as fractional coordinate arrays $\mathbf{r}_{\text{frac}}$.
*   **Fractional Reciprocal Space**: Reciprocal lattice translations $\mathbf{G}_{\text{frac}}$ are exactly integers, and wavevectors $\mathbf{q}_{\text{frac}}$ are expressed in the reciprocal fractional basis.

In solid state physics Cartesian derivations (such as above), $\mathbf{G} \cdot \mathbf{R} = 2\pi n$, allowing the exponent to simply read $e^{i\mathbf{G} \cdot \Delta\mathbf{r}}$.

However, because $\mathbf{G}_{\text{frac}} \cdot \mathbf{R}_{\text{frac}} = n$ exactly without a $2\pi$ factor, implementing the analytical similarity mapping in software requires re-inserting $2\pi$ to conserve the physical phase array magnitude. Therefore, the same formula in a purely fractional basis becomes:

$$D^{\text{sym}}(\kappa_i, \kappa_j | \mathbf{q}_{\text{frac}}) = \frac{1}{N_q} \sum_{S \in G_{\mathbf{q}}} S^{-1} \left[ D(S(\kappa_i), S(\kappa_j) | \mathbf{q}_{\text{frac}}) e^{i 2\pi \mathbf{G}_{\text{frac}} \cdot (\mathbf{r}_{S(\kappa_j), \text{frac}} - \mathbf{r}_{S(\kappa_i), \text{frac}})} \right] S$$
