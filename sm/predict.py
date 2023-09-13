#%% Imports -------------------------------------------------------------------

import nd2
import napari
import random
import numpy as np
from skimage import io
import tensorflow as tf
from pathlib import Path
import segmentation_models as sm
from joblib import Parallel, delayed 
from skimage.transform import rescale

#%% Parameters ----------------------------------------------------------------

# Paths
data_path = Path(Path.cwd(), 'data', 'local')
stack_name = 'KASind1.nd2'
# stack_name = 'KZLind1.nd2'

# Training
rescale_factor = 0.25

#%% Predict -------------------------------------------------------------------

# Open & format prediction data
stack = nd2.imread(Path(data_path) / stack_name).squeeze()  
stack = rescale(stack, (1, rescale_factor, rescale_factor), preserve_range=True)
pMax = np.percentile(stack, 99.9)
stack[stack > pMax] = pMax
stack = (stack / pMax)

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

# Predict masks
model.load_weights(Path(Path.cwd(), 'model_masks_weights.h5'))
masks_probs = model.predict(stack).squeeze() 

# Predict centroids
model.load_weights(Path(Path.cwd(), 'model_centroids_weights.h5'))
centroids_prob = model.predict(stack).squeeze() 

# Display 
viewer = napari.Viewer()
viewer.add_image(stack, scale=[1.3675, 1, 1])
viewer.add_image(masks_probs, scale=[1.3675, 1, 1])
# viewer.add_image(centroids_prob, scale=[1.3675, 1, 1])
 
#%%

# from skimage.filters import gaussian
# from skimage.measure import label

# sig = 3
# thresh = 0.5
# probs = gaussian(probs, sigma=(sig / 2.735, sig, sig))
# mask = probs > thresh
# labels = label(mask)
# test = predict_images.copy()
# test[mask==0] = 0

# Display 
# viewer = napari.Viewer()
# viewer.add_image(predict_images, scale=[2.735, 1, 1])
# viewer.add_image(probs, scale=[2.735, 1, 1])
# viewer.add_image(mask, scale=[2.735, 1, 1])
# viewer.add_labels(labels, scale=[2.735, 1, 1])
# viewer.add_image(test, scale=[2.735, 1, 1])