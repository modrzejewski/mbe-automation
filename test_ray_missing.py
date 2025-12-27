
import sys
import builtins

# create a mock importer that fails for 'ray'
real_import = builtins.__import__

def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'ray':
        raise ImportError("No module named 'ray'")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = mock_import

try:
    import mbe_automation.calculators.core
    print("Import successful")
except NameError as e:
    print(f"Caught NameError: {e}")
except Exception as e:
    print(f"Caught unexpected error: {type(e).__name__}: {e}")
finally:
    # restore import
    builtins.__import__ = real_import
