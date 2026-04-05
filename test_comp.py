import sys
sys.path.append("src")
from mbe_automation.structure.clusters import MolecularComposition
try:
    mc = MolecularComposition()
except Exception as e:
    print(e)
