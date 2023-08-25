#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path

#%% Initialize ----------------------------------------------------------------

stack_name = 'KASind1001_crop.tif'
stack = io.imread(Path('data') / stack_name)

#%% ---------------------------------------------------------------------------

