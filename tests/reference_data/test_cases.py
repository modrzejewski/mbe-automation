from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent.parent
REFERENCE_DATA_DIR = Path(__file__).resolve().parent
X23_DIR = REFERENCE_DATA_DIR / "xyz" / "X23"
MACE_MODELS_DIR = REFERENCE_DATA_DIR / "mace_models" / "Pia_et_al_Chem_Sci_2025"

TEST_CASES = [
    {
        "name": "1,4-cyclohexanedione",
        "crystal_path": X23_DIR / "01_1,4-cyclohexanedione" / "solid.xyz",
        "molecule_path": X23_DIR / "01_1,4-cyclohexanedione" / "molecule.xyz",
        "model_path": MACE_MODELS_DIR / "01_cyclohexanedione" / "MACE_model_swa.model",
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "01_1,4-cyclohexanedione" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "01_1,4-cyclohexanedione" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "acetic_acid",
        "crystal_path": X23_DIR / "02_acetic_acid" / "solid.xyz",
        "molecule_path": X23_DIR / "02_acetic_acid" / "molecule.xyz",
        "model_path": MACE_MODELS_DIR / "02_acetic_acid" / "MACE_model_swa.model",
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "02_acetic_acid" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "02_acetic_acid" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "adamantane",
        "crystal_path": X23_DIR / "03_adamantane" / "solid.xyz",
        "molecule_path": X23_DIR / "03_adamantane" / "molecule.xyz",
        "model_path": MACE_MODELS_DIR / "03_adamantane" / "MACE_model_swa.model",
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "03_adamantane" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "03_adamantane" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "ammonia",
        "crystal_path": X23_DIR / "04_ammonia" / "solid.xyz",
        "molecule_path": X23_DIR / "04_ammonia" / "molecule.xyz",
        "model_path": MACE_MODELS_DIR / "04_ammonia" / "MACE_model_swa.model",
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "04_ammonia" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "04_ammonia" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "anthracene",
        "crystal_path": X23_DIR / "05_anthracene" / "solid.xyz",
        "molecule_path": X23_DIR / "05_anthracene" / "molecule.xyz",
        "model_path": MACE_MODELS_DIR / "05_anthracene" / "MACE_model_swa.model",
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "05_anthracene" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "05_anthracene" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "benzene",
        "crystal_path": X23_DIR / "06_benzene" / "solid.xyz",
        "molecule_path": X23_DIR / "06_benzene" / "molecule.xyz",
        "model_path": MACE_MODELS_DIR / "06_benzene" / "MACE_model_swa.model",
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "06_benzene" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "06_benzene" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "CO2",
        "crystal_path": X23_DIR / "07_CO2" / "solid.xyz",
        "molecule_path": X23_DIR / "07_CO2" / "molecule.xyz",
        "model_path": None,
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "07_CO2" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "07_CO2" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "cyanamide",
        "crystal_path": X23_DIR / "08_cyanamide" / "solid.xyz",
        "molecule_path": X23_DIR / "08_cyanamide" / "molecule.xyz",
        "model_path": None,
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "08_cyanamide" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "08_cyanamide" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "cytosine",
        "crystal_path": X23_DIR / "09_cytosine" / "solid.xyz",
        "molecule_path": X23_DIR / "09_cytosine" / "molecule.xyz",
        "model_path": None,
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "09_cytosine" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "09_cytosine" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "ethyl_carbamate",
        "crystal_path": X23_DIR / "10_ethyl_carbamate" / "solid.xyz",
        "molecule_path": X23_DIR / "10_ethyl_carbamate" / "molecule.xyz",
        "model_path": None,
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "10_ethyl_carbamate" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "10_ethyl_carbamate" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "formamide",
        "crystal_path": X23_DIR / "11_formamide" / "solid.xyz",
        "molecule_path": X23_DIR / "11_formamide" / "molecule.xyz",
        "model_path": None,
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "11_formamide" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "11_formamide" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "imidazole",
        "crystal_path": X23_DIR / "12_imidazole" / "solid.xyz",
        "molecule_path": X23_DIR / "12_imidazole" / "molecule.xyz",
        "model_path": None,
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "12_imidazole" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "12_imidazole" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "naphthalene",
        "crystal_path": X23_DIR / "13_naphthalene" / "solid.xyz",
        "molecule_path": X23_DIR / "13_naphthalene" / "molecule.xyz",
        "model_path": None,
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "13_naphthalene" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "13_naphthalene" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "oxalic_acid_alpha",
        "crystal_path": X23_DIR / "14_oxalic_acid_alpha" / "solid.xyz",
        "molecule_path": X23_DIR / "14_oxalic_acid_alpha" / "molecule.xyz",
        "model_path": None,
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "14_oxalic_acid_alpha" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "14_oxalic_acid_alpha" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "oxalic_acid_beta",
        "crystal_path": X23_DIR / "15_oxalic_acid_beta" / "solid.xyz",
        "molecule_path": X23_DIR / "15_oxalic_acid_beta" / "molecule.xyz",
        "model_path": None,
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "15_oxalic_acid_beta" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "15_oxalic_acid_beta" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "pyrazine",
        "crystal_path": X23_DIR / "16_pyrazine" / "solid.xyz",
        "molecule_path": X23_DIR / "16_pyrazine" / "molecule.xyz",
        "model_path": None,
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "16_pyrazine" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "16_pyrazine" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "pyrazole",
        "crystal_path": X23_DIR / "17_pyrazole" / "solid.xyz",
        "molecule_path": X23_DIR / "17_pyrazole" / "molecule.xyz",
        "model_path": None,
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "17_pyrazole" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "17_pyrazole" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "triazine",
        "crystal_path": X23_DIR / "18_triazine" / "solid.xyz",
        "molecule_path": X23_DIR / "18_triazine" / "molecule.xyz",
        "model_path": None,
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "18_triazine" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "18_triazine" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "trioxane",
        "crystal_path": X23_DIR / "19_trioxane" / "solid.xyz",
        "molecule_path": X23_DIR / "19_trioxane" / "molecule.xyz",
        "model_path": None,
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "19_trioxane" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "19_trioxane" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "uracil",
        "crystal_path": X23_DIR / "20_uracil" / "solid.xyz",
        "molecule_path": X23_DIR / "20_uracil" / "molecule.xyz",
        "model_path": None,
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "20_uracil" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "20_uracil" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "urea",
        "crystal_path": X23_DIR / "21_urea" / "solid.xyz",
        "molecule_path": X23_DIR / "21_urea" / "molecule.xyz",
        "model_path": None,
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "21_urea" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "21_urea" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "hexamethylenetetramine",
        "crystal_path": X23_DIR / "22_hexamethylenetetramine" / "solid.xyz",
        "molecule_path": X23_DIR / "22_hexamethylenetetramine" / "molecule.xyz",
        "model_path": None,
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "22_hexamethylenetetramine" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "22_hexamethylenetetramine" / "symmetry_weights" / "trimers.csv",
        }
    },
    {
        "name": "succinic_acid",
        "crystal_path": X23_DIR / "23_succinic_acid" / "solid.xyz",
        "molecule_path": X23_DIR / "23_succinic_acid" / "molecule.xyz",
        "model_path": None,
        "general_model_path": MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model",
        "symmetry_weights": {
            "dimers": REFERENCE_DATA_DIR / "X23" / "23_succinic_acid" / "symmetry_weights" / "dimers.csv",
            "trimers": REFERENCE_DATA_DIR / "X23" / "23_succinic_acid" / "symmetry_weights" / "trimers.csv",
        }
    },
]
