#
# Models for which recommended parameters can be automatically
# filled in into configuration data classes
#
SEMIEMPIRICAL = [
    "gfn2-xtb",
    "gfn1-xtb",
    "dftb3-d4",
    "dftb+mbd",    
]

SEMIEMPIRICAL_DFTB = SEMIEMPIRICAL

NEURAL_NETWORKS = [
    "mace",
    "uma",
]

COUPLED_CLUSTERS = [
    "rpa",
    "lno-ccsd(t)"
]

KNOWN_MODELS = SEMIEMPIRICAL + NEURAL_NETWORKS + COUPLED_CLUSTERS
