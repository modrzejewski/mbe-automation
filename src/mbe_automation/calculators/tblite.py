#
# General threshold for xTB accuracy.
# Smaller value means tighter thresholds,
# Table from https://xtb-docs.readthedocs.io/en/latest/sp.html#accuracy:
#
# +------------------------------+-------------+------------+--------------+
# | Parameter                    | Accuracy=30 | Accuracy=1 | Accuracy=0.2 |
# +==============================+=============+============+==============+
# | Integral cutoff              | 20.0        | 25.0       | 32.0         |
# +------------------------------+-------------+------------+--------------+
# | Integral neglect             | 3.0e-7      | 1.0e-8     | 2.0e-9       |
# +------------------------------+-------------+------------+--------------+
# | SCC convergence / Eh         | 3.0e-5      | 1.0e-6     | 2.0e-7       |
# +------------------------------+-------------+------------+--------------+
# | Wavefunction convergence / e | 3.0e-3      | 1.0e-4     | 2.0e-5       |
# +------------------------------+-------------+------------+--------------+
#
TIGHT_ACCURACY_GFN2_XTB = 0.01

def GFN2_xTB(verbose=False):
    """
    Get a calculator for GFN2-xTB with tight accuracy
    settings.
    
    Checks for 'tblite' availability only when called.
    """
    try:
        import tblite
    except ImportError:
        raise ImportError(
            "The 'tblite' package is not installed. "
            "GFN2_xTB calculator cannot be created."
        )
        
    return tblite.ase.TBLite(
        method="GFN2-xTB",
        verbosity=(1 if verbose else 0),
        accuracy=TIGHT_ACCURACY_GFN2_XTB,
    )
