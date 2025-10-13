import os
import importlib
import inspect

__all__ = []           # Optional: names exported via `from myfolder import *`
__all_classes__ = []   # List of class objects

package_name = __name__
current_dir = os.path.dirname(__file__)

for filename in os.listdir(current_dir):
    if filename.endswith('.py') and not filename.startswith('_') and filename != '__init__.py':
        module_name = filename[:-3]
        module = importlib.import_module(f'.{module_name}', package_name)

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == f'{package_name}.{module_name}':
                globals()[name] = obj             # Make class directly accessible
                __all__.append(name)              # (Optional) For wildcard imports
                __all_classes__.append(obj)       # Store the class object
