from mbe_automation.configs.quasi_harmonic import EmpiricalEnergyCorrection

def test_empirical_energy_correction():
    eec = EmpiricalEnergyCorrection(
        variant="alpha*(V-V0)",
        T0=300.0,
        V0=150.0
    )

    assert eec.variant == "alpha*(V-V0)"
    assert eec.T0 == 300.0
    assert eec.V0 == 150.0

print("Running test_qha.py...")
test_empirical_energy_correction()
print("test_qha.py passed.")
