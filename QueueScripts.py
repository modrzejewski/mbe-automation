import os
import os.path
import stat

def Make(QueueDirs, QueueMainScript, SystemTypes, TemplateFile, InpDirs, LogDirs):
    f = open(TemplateFile, "r")
    Template = f.read()
    f.close()
    MaxBlockSize = 1000
    SlurmCommands = ""
    for SystemType in SystemTypes:
        for BasisType in ("small-basis", "large-basis"):
            InputDir = InpDirs[SystemType][BasisType]
            Files = sorted(os.listdir(InputDir))
            NTasks = len(Files)
            NBlocks = NTasks // MaxBlockSize
            if NTasks % MaxBlockSize > 0:
                NBlocks += 1
            for b in range(1, NBlocks + 1):
                i0 = 1 + (b - 1) * MaxBlockSize
                i1 = min(b * MaxBlockSize, NTasks)
                FilePath = os.path.join(QueueDirs[SystemType][BasisType], f"{i0}-{i1}.py")
                D = {"FIRST": str(i0),
                     "LAST": str(i1),
                     "INP_DIR": InpDirs[SystemType][BasisType],
                     "LOG_DIR": LogDirs[SystemType][BasisType],
                     "NTASKS": NTasks,
                     "BASIS_TYPE": BasisType,
                     "SYSTEM_TYPE": SystemType,
                     "PART": str(b)
                    }
                s = Template.format(**D)
                f = open(FilePath, "w")
                f.write(s)
                f.close()
                SlurmCommands += f'os.system("sbatch --array=1-{i1-i0+1} {FilePath}")\n'
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
