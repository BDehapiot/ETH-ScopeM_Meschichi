#%% Imports -------------------------------------------------------------------

import nd2
import napari
import numpy as np
from pathlib import Path
import segmentation_models as sm
from joblib import Parallel, delayed 
from skimage.transform import rescale

#%% Initialize ----------------------------------------------------------------

# Get stack name
# stack_name = 'KASind1.nd2'
stack_name = 'KASind1001.nd2'
# stack_name = 'KASind2001.nd2'
# stack_name = 'KASind3001.nd2'
# stack_name = 'KZLind1.nd2'
# stack_name = 'KALind1001.nd2'
# stack_name = 'KALind2001.nd2'
# stack_name = 'KALind3001.nd2'

# Get paths
data_path = Path(Path.cwd(), 'data', 'local')

# Load model
model_name = 'model_weights_0.5.h5'
rescale_factor = float(model_name[14:17])

#%% Parameters ----------------------------------------------------------------


#%% Pre-processing ------------------------------------------------------------

# Open data to predict (stack)
with nd2.ND2File(Path(data_path) / stack_name) as ndfile:
    stack = ndfile.asarray()
    stack_raw = stack.copy()
    voxel_size = list(ndfile.voxel_size())
    z_ratio = (voxel_size[2] / voxel_size[0]) * rescale_factor
    
# Format data to predict (stack)
stack = rescale(stack, (1, rescale_factor, rescale_factor), preserve_range=True)
pMax = np.percentile(stack, 99.9)
stack[stack > pMax] = pMax
stack = (stack / pMax)

#%% Predictions ---------------------------------------------------------------

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

# Predict probabilities
model.load_weights(Path(Path.cwd(), model_name))
probs = model.predict(stack).squeeze() 

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(stack, scale=[z_ratio, 1, 1])
# viewer.add_image(probs, scale=[z_ratio, 1, 1])
 
#%% Post-processing -----------------------------------------------------------

from skimage.filters import gaussian
from skimage.measure import label

# Parameters
sig = 2
thresh = 0.5

# Extract nMask and labels
probs = gaussian(probs, sigma=(sig / z_ratio, sig, sig))
nMask = probs > thresh
labels = label(nMask)

# Rescale nMask and labels
nMask = rescale(
    nMask, (1, 1/rescale_factor, 1/rescale_factor), 
    preserve_range=True, order=0,
    )
labels = rescale(
    labels, (1, 1/rescale_factor, 1/rescale_factor), 
    preserve_range=True, order=0,
    )

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(stack_raw, scale=[z_ratio, 1, 1])
# viewer.add_image(nMask, scale=[z_ratio, 1, 1])
# viewer.add_image(labels, scale=[z_ratio, 1, 1])

#%%

from skimage.morphology import white_tophat, disk

def tophat(plane):
    tophat = white_tophat(plane, footprint=disk(5))
    tophat = gaussian(tophat, sigma=sig, preserve_range=True)
    return tophat
outputs = Parallel(n_jobs=-1)(
    delayed(tophat)(plane)
    for plane in stack_raw
    )
tophat = np.stack([data for data in outputs])
tophat[nMask == 0] = 0

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(stack_raw, scale=[z_ratio, 1, 1])
# viewer.add_image(nMask, scale=[z_ratio, 1, 1])
# viewer.add_image(tophat, scale=[z_ratio, 1, 1])

#%%

from skimage.filters import threshold_otsu

cMask = np.zeros_like(tophat)
for lab in np.unique(labels):
    if lab > 0:
        idx = (labels == lab)
        values = tophat[idx]
        thresh = threshold_otsu(values)
        cMask[idx] = (values > thresh).astype(int)
        
# Display 
viewer = napari.Viewer()
viewer.add_image(stack_raw, scale=[z_ratio, 1, 1])
viewer.add_image(nMask, scale=[z_ratio, 1, 1])
viewer.add_image(cMask, scale=[z_ratio, 1, 1])


#%% Watershed -----------------------------------------------------------------

# from scipy.ndimage import distance_transform_edt
# from skimage.feature import peak_local_max
# from skimage.segmentation import watershed

# # Distance transform
# distance = distance_transform_edt(nMask, sampling=(z_ratio, 1, 1))

# # Local maxima
# local_max = peak_local_max(distance, min_distance=12, exclude_border=False)

# # # Display 
# viewer = napari.Viewer()
# viewer.add_image(distance, scale=[z_ratio, 1, 1])
# viewer.add_points(local_max, scale=[z_ratio, 1, 1])

#%%

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(stack, scale=[z_ratio, 1, 1])
# viewer.add_image(probs, scale=[z_ratio, 1, 1])
# viewer.add_image(nMask, scale=[z_ratio, 1, 1])
# viewer.add_labels(labels, scale=[z_ratio, 1, 1])