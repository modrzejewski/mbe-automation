#
# Models for which recommended parameters can be automatically
# filled in into configuration data classes
#
SEMIEMPIRICAL_DFTB = [
    "gfn2-xtb",
    "gfn1-xtb",
    "dftb3-d4",
    "dftb3+mbd",    
]

NEURAL_NETWORKS = [
    "mace",
    "uma",
]

KNOWN_MODELS = SEMIEMPIRICAL_DFTB + NEURAL_NETWORKS
