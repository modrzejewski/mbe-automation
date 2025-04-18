import os
import os.path
import stat
import math
from . import directory_structure

def Make(QueueDirs, QueueMainScript, ClusterTypes, MonomerRelaxation,
         TemplateFile, InpDirs, LogDirs, Method):
    
    f = open(TemplateFile, "r")
    Template = f.read()
    f.close()
    MaxBlockSize = 100
    SlurmCommands = ""
    if MonomerRelaxation:
        SlurmCommands += f"#\n#{'monomers'.center(80)}\n#\n"
        for SystemType in ["monomers-relaxed", "monomers-supercell"]:
            for BasisType in ["small-basis", "large-basis"]:
                RelaxedXYZ = sorted(os.listdir(InpDirs[SystemType][BasisType]))
                NMonomers = len(RelaxedXYZ)
                system0 = "all"
                system1 = "unique"
                FilePath = os.path.abspath(os.path.join(QueueDirs[SystemType][BasisType], f"{system0}-{system1}.py"))
                D = {"FIRST_SYSTEM": system0,
                     "LAST_SYSTEM": system1,
                     "OFFSET": 1,
                     "INP_DIR": os.path.abspath(InpDirs[SystemType][BasisType]),
                     "LOG_DIR": os.path.abspath(LogDirs[SystemType][BasisType]),
                     "NTASKS": NMonomers,
                     "BASIS_TYPE": BasisType,
                     "SYSTEM_TYPE": SystemType
                     }
                s = Template.format(**D)
                f = open(FilePath, "w")
                f.write(s)
                f.close()
                SlurmCommands += f"""os.system("sbatch --array=1-{NMonomers} '{FilePath}'")\n"""
        
    for SystemType in ClusterTypes:
        SlurmCommands += f"#\n#{SystemType.center(80)}\n#\n"
        for BasisType in ("small-basis", "large-basis"):
            InputDir = InpDirs[SystemType][BasisType]
            if Method != "LNO-CCSD(T)":
                Files = sorted(os.listdir(InputDir))
            else:
                Files = sorted(os.listdir(os.path.join(InputDir, directory_structure.SUBSYSTEM_LABELS[SystemType][0])))
            NTasks = len(Files)
            NBlocks = NTasks // MaxBlockSize
            if NTasks % MaxBlockSize > 0:
                NBlocks += 1
            for b in range(1, NBlocks + 1):
                i0 = 1 + (b - 1) * MaxBlockSize
                i1 = min(b * MaxBlockSize, NTasks)
                d = math.ceil(math.log(NTasks, 10))
                system0 = str(i0-1).zfill(d)
                system1 = str(i1-1).zfill(d)
                FilePath = os.path.abspath(os.path.join(QueueDirs[SystemType][BasisType], f"{system0}-{system1}.py"))
                D = {"FIRST_SYSTEM": system0,
                     "LAST_SYSTEM": system1,
                     "OFFSET": i0,
                     "INP_DIR": os.path.abspath(InpDirs[SystemType][BasisType]),
                     "LOG_DIR": os.path.abspath(LogDirs[SystemType][BasisType]),
                     "NTASKS": NTasks,
                     "BASIS_TYPE": BasisType,
                     "SYSTEM_TYPE": SystemType
                    }
                s = Template.format(**D)
                f = open(FilePath, "w")
                f.write(s)
                f.close()
                SlurmCommands += f"""os.system("sbatch --array=1-{i1-i0+1} '{FilePath}'")\n"""
    #
    # Make QueueMainScript: a master script which runs all
    # queue jobs. The user can control the number of computed systems
    # by commenting out parts of QueueMainScript.
    #
    f = open(QueueMainScript, "w")
    s = """#!/usr/bin/env python3
import os
{Commands}
    """.format(Commands=SlurmCommands)
    f.write(s)
    f.close()
    #
    # Make QueueMainScript executable
    #    
    mode = os.stat(QueueMainScript).st_mode
    os.chmod(QueueMainScript, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Main queue script: {QueueMainScript}")
