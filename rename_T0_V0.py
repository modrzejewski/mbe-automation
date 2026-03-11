with open("src/mbe_automation/configs/quasi_harmonic.py", "r") as f:
    content = f.read()

content = content.replace("T0: float", "T_ref: float")
content = content.replace("V0: float", "V_ref: float")
content = content.replace("self.empirical_energy_correction.T0", "self.empirical_energy_correction.T_ref")

with open("src/mbe_automation/configs/quasi_harmonic.py", "w") as f:
    f.write(content)

with open("src/mbe_automation/dynamics/harmonic/core.py", "r") as f:
    content = f.read()

content = content.replace("empirical_energy_correction.T0", "empirical_energy_correction.T_ref")
content = content.replace("empirical_energy_correction.V0", "empirical_energy_correction.V_ref")
content = content.replace("T0 = empirical", "T_ref = empirical")
content = content.replace("V0_target = empirical", "V_ref_target = empirical")
content = content.replace("T0_idx", "T_ref_idx")
content = content.replace("T0,", "T_ref,")
content = content.replace("T0)", "T_ref)")
content = content.replace("T0}", "T_ref}")
content = content.replace("V0=", "V_ref=")
content = content.replace("V0}", "V_ref}")
content = content.replace("V0_target", "V_ref_target")

with open("src/mbe_automation/dynamics/harmonic/core.py", "w") as f:
    f.write(content)
