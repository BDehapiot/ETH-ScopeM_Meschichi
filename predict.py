#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
import segmentation_models as sm
from skimage.filters import gaussian

# Functions
from functions import open_stack, get_patches, merge_patches

#%% Inputs --------------------------------------------------------------------

# Paths
local_path = Path('D:/local_Meschichi/data')
rscale_model_path = Path(Path.cwd(), "rscale_model_weights.h5")
rslice_model_path = Path(Path.cwd(), "rslice_model_weights.h5")
# stack_name = "KASind1.nd2"
stack_name = "KASind1003.nd2"

# Patches
size = 128
overlap = size // 8

#%% Pre-processing ------------------------------------------------------------

# Open data
path = Path(local_path, stack_name)
print("Open data       :", end='')
t0 = time.time()
rscale, rslice = open_stack(path)
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

# Extract patches
print("Extract patches :", end='')
t0 = time.time()
rscale_patches = np.stack(get_patches(rscale, size, overlap))
rslice_patches = np.stack(get_patches(rslice, size, overlap))
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

#%% Predict -------------------------------------------------------------------

# Define & compile model
model = sm.Unet(
    'resnet34', 
    input_shape=(None, None, 1), 
    classes=1, 
    activation='sigmoid', 
    encoder_weights=None,
    )
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', 
    metrics=['mse']
    )

# Predict (rscale)
model.load_weights(rscale_model_path) 
rscale_predict = model.predict(rscale_patches).squeeze()

# Predict (rslice)
model.load_weights(rslice_model_path) 
rslice_predict = model.predict(rslice_patches).squeeze()

# Merge patches
print("Merge patches   :", end='')
t0 = time.time()
rscale_predict = merge_patches(rscale_predict, rscale.shape, size, overlap)
rslice_predict = merge_patches(rslice_predict, rslice.shape, size, overlap)
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

# Merge predictions
rslice_predict = np.swapaxes(rslice_predict, 0, 1)
predict = (rscale_predict + rslice_predict) / 2
predict = gaussian(predict, sigma=2) # Parameter
predict = predict > 0.375 # Parameter

# Display
viewer = napari.Viewer()
viewer.add_image(rscale) 
viewer.add_image(predict, blending="additive", colormap="red", opacity=0.5) 

# -----------------------------------------------------------------------------