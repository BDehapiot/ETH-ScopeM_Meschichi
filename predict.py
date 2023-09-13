#%% Imports -------------------------------------------------------------------

import nd2
import napari
import numpy as np
from pathlib import Path
import segmentation_models as sm
from skimage.transform import rescale

#%% Initialize ----------------------------------------------------------------

# Get stack name
# stack_name = 'KASind1.nd2'
stack_name = 'KASind1001.nd2'
# stack_name = 'KZLind1.nd2'
# stack_name = 'KALind1001.nd2'

# Get paths
data_path = Path(Path.cwd(), 'data', 'local')

# Load model
model_name = 'model_weights_0.5.h5'
rescale_factor = float(model_name[14:17])

#%% Parameters ----------------------------------------------------------------


#%% Data pre-processing -------------------------------------------------------

# Open data to predict (stack)
with nd2.ND2File(Path(data_path) / stack_name) as ndfile:
    stack = ndfile.asarray()
    voxel_size = list(ndfile.voxel_size())
    z_ratio = (voxel_size[2] / voxel_size[0]) * rescale_factor
    
# Format data to predict (stack)
stack = rescale(stack, (1, rescale_factor, rescale_factor), preserve_range=True)
pMax = np.percentile(stack, 99.9)
stack[stack > pMax] = pMax
stack = (stack / pMax)

#%% Data prediction -----------------------------------------------------------

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
 
#%% Data post-processing ------------------------------------------------------

from skimage.filters import gaussian
from skimage.measure import label

# Parameters
sig = 1
thresh = 0.5

# Post-processing
probs = gaussian(probs, sigma=(sig / z_ratio, sig, sig))
mask = probs > thresh
labels = label(mask)

# Display 
viewer = napari.Viewer()
viewer.add_image(stack, scale=[z_ratio, 1, 1])
viewer.add_image(probs, scale=[z_ratio, 1, 1])
viewer.add_image(mask, scale=[z_ratio, 1, 1])
viewer.add_labels(labels, scale=[z_ratio, 1, 1])
# viewer.add_image(test, scale=[z_ratio, 1, 1])