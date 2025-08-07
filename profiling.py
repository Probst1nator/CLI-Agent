import sys
import time

class ImportProfiler:
    def __init__(self, threshold=0.2):
        self.threshold = threshold
        self.import_times = {}

    def __call__(self, name, globals=None, locals=None, fromlist=(), level=0):
        start_time = time.time()
        
        # Use the original __import__ function to maintain correct behavior
        module = original_import(name, globals, locals, fromlist, level)
        
        end_time = time.time()
        
        import_time = end_time - start_time
        if import_time > self.threshold:
            self.import_times[name] = import_time
        
        return module

    def print_results(self):
        print(f"Imports taking longer than {self.threshold} seconds:")
        for name, duration in sorted(self.import_times.items(), key=lambda x: x[1], reverse=True):
            print(f"{name}: {duration:.2f} seconds")

# Store the original import function
original_import = __import__

# Create an instance of the ImportProfiler
profiler = ImportProfiler(threshold=0.2)

# Replace the built-in __import__ function with our profiler
sys.modules['builtins'].__import__ = profiler

# Your existing imports
#!/usr/bin/env python3

import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:")
# Suppress phonemizer warnings
warnings.filterwarnings("ignore", message="words count mismatch on*", module="phonemizer", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="phonemizer")  # Catch all phonemizer warnings






# Print the profiling results
profiler.print_results()

# Add an early exit after printing the results
sys.exit(0)

# The rest of your main.py code would go here, but it won't be executed due to the early exit