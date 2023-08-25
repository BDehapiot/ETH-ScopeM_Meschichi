#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path

#%% Initialize ----------------------------------------------------------------

stack_name = 'KASind1001_crop.tif'
stack = io.imread(Path('data') / stack_name)

#%% Parameters ----------------------------------------------------------------

sigma = 3
thresh_coeff = 1

#%% Process -------------------------------------------------------------------

from skimage.filters import gaussian, threshold_otsu

process = gaussian(stack, sigma=sigma)
mask = process > threshold_otsu(process) * thresh_coeff

#%% Display -------------------------------------------------------------------

viewer = napari.Viewer()
# viewer.add_image(stack, name='stack')
viewer.add_image(process, name='process')
viewer.add_image(mask, name='mask')