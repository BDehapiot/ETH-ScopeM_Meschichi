#%% Imports -------------------------------------------------------------------

import nd2
import napari
import numpy as np
from pathlib import Path
import segmentation_models as sm
from skimage.measure import label
from joblib import Parallel, delayed 
from skimage.transform import rescale
from skimage.segmentation import clear_border
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects, white_tophat, disk, ball

#%% Comments ------------------------------------------------------------------

'''
- There is something weird with raw data normalization using np.percentile. The
Idea is to reopen data without normalization for tophat transform. 
'''

#%% Initialize ----------------------------------------------------------------

# Get stack name
# stack_name = 'KASind1.nd2'
# stack_name = 'KASind1001.nd2'
# stack_name = 'KASind2001.nd2'
# stack_name = 'KASind3001.nd2'
# stack_name = 'KZLind1.nd2'
# stack_name = 'KZLind1001.nd2'
# stack_name = 'KZLind2001.nd2'
stack_name = 'KZLind3001.nd2'

# Get paths
data_path = Path(Path.cwd(), 'data', 'local')

# Load model
model_name = 'model_weights_0.5.h5'
rescale_factor = float(model_name[14:17])

#%% Parameters ----------------------------------------------------------------

# Nuclei mask (nMask)
prob_min = 0.5
prob_sigma = 2 * rescale_factor
clear_nBorder = False
min_nSize = 4096 * rescale_factor

# Chromocenter mask (cMask)
tophat_size = 4 * rescale_factor
tophat_sigma = 2 * rescale_factor
tophat_tresh_coeff = 1.5
min_cSize = 32 * rescale_factor

#%% Pre-processing ------------------------------------------------------------

# Open data to predict (stack)
with nd2.ND2File(Path(data_path) / stack_name) as ndfile:
    stack = ndfile.asarray()
    voxel_size = list(ndfile.voxel_size())
    z_ratio = (voxel_size[2] / voxel_size[0]) * rescale_factor
    
# Format data to predict (stack)
stack = rescale(stack, (1, rescale_factor, rescale_factor), preserve_range=True)
stack_raw = stack.copy()
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

# Get nMask and nLabels from probs (n for nuclei)
probs = gaussian(probs, sigma=(prob_sigma / z_ratio, prob_sigma, prob_sigma))
nMask = probs > prob_min
nBorder = nMask.copy()
if clear_nBorder:
    nMask = clear_border(nMask)
nSmall = nMask.copy()    
nMask = remove_small_objects(nMask, min_size=min_nSize)
nLabels = label(nMask)

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(stack, scale=[z_ratio, 1, 1])
# viewer.add_image(
#     nBorder, name='border nuclei', scale=[z_ratio, 1, 1], 
#     rendering='attenuated_MIP', colormap='magenta'
#     )
# viewer.add_image(
#     nSmall, name='small nuclei', scale=[z_ratio, 1, 1], 
#     rendering='attenuated_MIP', colormap='red'
#     )
# viewer.add_image(
#     nMask, scale=[z_ratio, 1, 1], 
#     rendering='attenuated_MIP', colormap='green'
#     )
# viewer.dims.ndisplay = 3

# -----------------------------------------------------------------------------

# Compute tophat transform
def tophat(plane):
    tophat = white_tophat(plane, footprint=disk(tophat_size))
    tophat = gaussian(tophat, sigma=tophat_sigma, preserve_range=True)
    return tophat
outputs = Parallel(n_jobs=-1)(
    delayed(tophat)(plane)
    for plane in stack_raw
    )
tophat = np.stack([data for data in outputs])
tophat[nMask == 0] = 0

# Display 
viewer = napari.Viewer()
viewer.add_image(stack_raw, scale=[z_ratio, 1, 1])
viewer.add_image(tophat, scale=[z_ratio, 1, 1], colormap='inferno')
viewer.dims.ndisplay = 3

# -----------------------------------------------------------------------------

# Get cMask from tophat (c for chromocenters)
cMask = np.zeros_like(tophat, dtype=bool)
for lab in np.unique(nLabels):
    if lab > 0:
        idx = (nLabels == lab)
        values = tophat[idx]
        thresh = threshold_otsu(values)
        cMask[idx] = (values > thresh * tophat_tresh_coeff)
cSmall = cMask.copy()
cMask = remove_small_objects(cMask, min_size=min_cSize)
                
# Display #1 
viewer = napari.Viewer()
viewer.add_image(stack_raw, scale=[z_ratio, 1, 1])
viewer.add_image(
    nMask, scale=[z_ratio, 1, 1], 
    rendering='attenuated_MIP', colormap='gray')
viewer.add_image(
    cSmall, name='small chromocenters', scale=[z_ratio, 1, 1], 
    rendering='attenuated_MIP', colormap='magenta')
viewer.add_image(
    cMask, scale=[z_ratio, 1, 1], 
    rendering='attenuated_MIP', colormap='green')
viewer.dims.ndisplay = 3

# Display #2

from skimage.morphology import binary_erosion
nMask_outlines = nMask ^ binary_erosion(nMask)
cMask_outlines = cMask ^ binary_erosion(cMask)

viewer = napari.Viewer()
viewer.add_image(stack_raw, scale=[z_ratio, 1, 1])
viewer.add_image(
    nMask_outlines, scale=[z_ratio, 1, 1], 
    rendering='attenuated_MIP', colormap='gray')
viewer.add_image(
    cMask_outlines, scale=[z_ratio, 1, 1], 
    rendering='attenuated_MIP', colormap='green')
viewer.dims.ndisplay = 3

#%% Watershed -----------------------------------------------------------------

# from scipy.ndimage import distance_transform_edt
# from skimage.feature import peak_local_max
# from skimage.segmentation import watershed

# # Distance transform
# cDist = distance_transform_edt(cMask, sampling=(z_ratio, 1, 1))

# # Local maxima
# local_max = peak_local_max(
#     cDist, min_distance=2, exclude_border=False
#     )

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(tophat, scale=[z_ratio, 1, 1])
# viewer.add_image(cDist, scale=[z_ratio, 1, 1])
# point_layer = viewer.add_points(local_max, scale=[z_ratio, 1, 1])
# point_layer.size = 2 # Change the size of the points
# point_layer.face_color = 'red'  # Change the face color of the points
# point_layer.edge_color = 'black'  # Change the edge color of the points
# point_layer.opacity = 1  # Change the opacity of the points
# point_layer.symbol = 'cross'  # Change the symbol used for each point

#%%

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(stack, scale=[z_ratio, 1, 1])
# viewer.add_image(probs, scale=[z_ratio, 1, 1])
# viewer.add_image(nMask, scale=[z_ratio, 1, 1])
# viewer.add_nLabels(nLabels, scale=[z_ratio, 1, 1])