#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
from skimage.morphology import binary_dilation

#%% Parameters ----------------------------------------------------------------

idx = 5 # int or name

#%% Initialize ----------------------------------------------------------------

# Get paths
data_path = Path(Path.cwd(), 'data', 'local')
stack_paths = []
for path in data_path.iterdir():
    if path.is_dir():
        stack_paths.append(path)
        
# Get stack_path
if isinstance(idx, int):
    stack_path = Path(data_path, stack_paths[idx])
elif isinstance(idx, str):
    stack_path = Path(data_path, idx)
    
# Open metadata
metadata = pd.read_csv(Path(stack_path, f'{stack_path.name}_metadata.csv'))
voxel_ratio = metadata.loc[0, "voxel_ratio"]

# Open data
stack = io.imread(Path(stack_path, f'{stack_path.name}_stack.tif'))
nMasks = io.imread(Path(stack_path, f'{stack_path.name}_nMasks.tif')).astype(bool)
cMasks = io.imread(Path(stack_path, f'{stack_path.name}_cMasks.tif')).astype(bool)
tophat = io.imread(Path(stack_path, f'{stack_path.name}_tophat.tif'))

# Get scale
scale = [voxel_ratio, 1, 1]

# Get outlines
nOutlines, cOutlines = [], []
for nMask, cMask in zip(nMasks, cMasks):
    nOutlines.append(binary_dilation(nMask) ^ nMask)
    cOutlines.append(binary_dilation(cMask) ^ cMask)
nOutlines = np.stack(nOutlines)
cOutlines = np.stack(cOutlines)
        
#%% Display -------------------------------------------------------------------

# Check segmentation 2D
viewer = napari.Viewer()
viewer.add_image(stack, scale=scale, colormap='plasma')
viewer.add_image(tophat, scale=scale, colormap='plasma')
viewer.add_image(nOutlines, scale=scale, blending='additive', opacity=0.25)
viewer.add_image(cOutlines, scale=scale, blending='additive', opacity=0.5)

# Check segmentation 3D
viewer = napari.Viewer()
scale = [voxel_ratio, 1, 1]
viewer.add_image(stack, scale=scale)
viewer.add_image(tophat, scale=scale)
viewer.add_image(nMasks, scale=scale, rendering='attenuated_MIP', colormap='magenta', opacity=0.75)
viewer.add_image(cMasks, scale=scale, rendering='attenuated_MIP', colormap='green', opacity=0.5)
viewer.dims.ndisplay = 3