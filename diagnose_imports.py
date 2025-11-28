import sys, os
print('sys.executable=', sys.executable)
print('cwd=', os.getcwd())
print('sys.path[:10]=')
for p in sys.path[:10]:
    print(' ', p)
print('\nlisting cwd:')
for name in os.listdir('.'):
    print(' ', name)
try:
    import numpy as np
    import pandas as pd
    print('numpy ok', np.__version__, 'pandas ok', pd.__version__)
except Exception as e:
    import traceback
    print('IMPORT ERROR:')
    traceback.print_exc()
