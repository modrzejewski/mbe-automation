# Comparison of External Pressure Implementation: `machine-learning` vs. `feature/qha-pressure`

This document outlines the key differences in the implementation of external pressure within the quasi-harmonic approximation workflow between the `machine-learning` and `feature/qha-pressure` branches.

## 1. `machine-learning` Branch Implementation

The `machine-learning` branch introduces a robust and physically direct method for handling external pressure by explicitly calculating and minimizing the Gibbs free energy.

### Configuration (`configs/quasi_harmonic.py`)

- **`pressure_GPa`**: A `float` parameter that defines the constant external pressure applied to the system.
- **`thermal_pressures_GPa`**: An `npt.NDArray[np.floating]` that provides a range of pressures for sampling the equation of state (EOS). This allows the system to explore different volumes to accurately map the `G(V)` curve.

### Workflow Logic (`workflows/quasi_harmonic.py`)

The workflow is structured to first calculate the Gibbs free energy `G` for each volume sampled, where `G = F + pV`. The `pressure_GPa` value is passed directly to the data collection and EOS fitting functions.

### EOS Fitting (`dynamics/harmonic/eos.py`)

- The core of this implementation is that the `fit` function in `eos.py` directly receives and operates on the Gibbs free energy (`G`).
- It fits the `G(V)` curve using the chosen equation of state (e.g., polynomial, Birch-Murnaghan).
- The equilibrium volume `V_min` is found by directly minimizing the fitted `G(V)` curve, which is the correct thermodynamic procedure for a system under constant pressure.

## 2. `feature/qha-pressure` Branch Implementation

The `feature/qha-pressure` branch uses a less direct, two-step approach that can be less accurate. It fits the Helmholtz free energy and applies the pressure term afterward.

### Configuration (`configs/quasi_harmonic.py`)

- **`pressure_GPa`**: A `float` parameter for the constant external pressure.
- **`pressure_range`**: An `npt.NDArray[np.floating]` for EOS sampling, functionally identical to `thermal_pressures_GPa` in the other branch.

### Workflow Logic (`workflows/quasi_harmonic.py`)

The workflow calculates the Helmholtz free energy `F` at each sampled volume. The `pressure_GPa` is passed to the final data analysis stage but is not used during the primary EOS fit.

### EOS Fitting (`dynamics/harmonic/eos.py`)

- The `fit` function in this branch operates on the Helmholtz free energy (`F`), fitting an `F(V)` curve.
- To find the equilibrium volume under pressure, it implicitly minimizes `F(V) + pV`, but the `pV` term is added *after* the `F(V)` curve has been fitted. This is less direct and can introduce inaccuracies because the shape of the `F(V)` curve is determined without considering the pressure term that influences the equilibrium state.

## 3. Key Differences and Analysis

| Feature                  | `machine-learning` Branch                                   | `feature/qha-pressure` Branch                               | Analysis                                                                                                                               |
| ------------------------ | ----------------------------------------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Thermodynamic Quantity** | Fits **Gibbs Free Energy (G)** directly.                    | Fits **Helmholtz Free Energy (F)**.                         | Fitting `G` is more physically sound and direct for systems under external pressure.                                                   |
| **Pressure Parameters**  | `pressure_GPa` and `thermal_pressures_GPa`.                 | `pressure_GPa` and `pressure_range`.                        | Naming is slightly different, but functionality for EOS sampling is the same. The key is how `pressure_GPa` is used.               |
| **Accuracy**             | Higher, as it minimizes the correct thermodynamic potential. | Potentially lower, as it relies on a two-step fitting process. | The `machine-learning` branch avoids assumptions about the shape of the `F(V)` curve being independent of the final pressure analysis. |
| **Coding Style**         | The coding style and structure are largely similar.         | The coding style and structure are largely similar.         | No significant differences in style; the divergence is purely logical and physical.                                                  |

## 4. Conclusion and Recommendation

The implementation in the **`machine-learning` branch is superior**. It is more robust, physically accurate, and aligns better with standard thermodynamic principles for simulating systems under constant pressure. By directly fitting the Gibbs free energy, it ensures that the equilibrium volume is determined from the correct potential energy surface.

The approach in the `feature/qha-pressure` branch, while functional, is less direct and could lead to less accurate results, especially at higher pressures where the `pV` term significantly alters the shape of the potential energy surface.

It is recommended to adopt the logic from the `machine-learning` branch for all future work involving external pressure in the quasi-harmonic approximation.
