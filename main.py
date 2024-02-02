#%% Imports -------------------------------------------------------------------

import napari
from pathlib import Path
from functions import open_stack, format_stack, predict

#%% Inputs --------------------------------------------------------------------

# Paths
local_path = Path('D:/local_Meschichi/data')
stack_paths = []
for path in local_path.iterdir():
    if path.suffix == ".nd2":
        stack_paths.append(path)

#%%

from skimage.transform import resize
from skimage.filters import gaussian
from skimage.morphology import binary_dilation

# -----------------------------------------------------------------------------

idx = 20

# -----------------------------------------------------------------------------

# Open & format data
stack, metadata = open_stack(path)
rscale, rslice = format_stack(stack, metadata)

# Predict
probs = predict(rscale, rslice)
probs = resize(probs, stack.shape, order=1)
probs = gaussian(probs, sigma=(3, 5, 5))

#
nMask = probs > 0.375
nOutl = binary_dilation(nMask) ^ nMask

# Display
viewer = napari.Viewer()
viewer.add_image(stack)
viewer.add_image(nOutl, blending="additive")

# # Display
# viewer = napari.Viewer()
# viewer.add_image(rscale)
# viewer.add_image(predict)