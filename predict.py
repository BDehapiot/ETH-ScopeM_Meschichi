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
data_path = Path('D:/local_Meschichi/data')
data_path = Path(Path.cwd(), 'data', 'local')
# stack_name = 'KASind1.nd2'
stack_name = 'KZLind1.nd2'

# Training
rescale_factor = 0.5

#%% Predict -------------------------------------------------------------------

# Open & format prediction data
predict_images = nd2.imread(Path(data_path) / stack_name).squeeze()  
predict_images = rescale(predict_images, (1, rescale_factor, rescale_factor), preserve_range=True)
pMax = np.percentile(predict_images, 99.9)
predict_images[predict_images > pMax] = pMax
predict_images = (predict_images / pMax)

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
    'Adam', 
    loss='binary_crossentropy', 
    metrics=['mse']
    )

# Load the model
model.load_weights(Path(Path.cwd(), 'model_weights.h5'))

# Predict
probs = model.predict(predict_images).squeeze()  

#%%

from skimage.filters import gaussian
from skimage.measure import label

# sig = 3
# thresh = 0.5
# probs = gaussian(probs, sigma=(sig / 2.735, sig, sig))
# mask = probs > thresh
# labels = label(mask)
# test = predict_images.copy()
# test[mask==0] = 0

# Display 
viewer = napari.Viewer()
viewer.add_image(predict_images, scale=[2.735, 1, 1])
viewer.add_image(probs, scale=[2.735, 1, 1])
# viewer.add_image(mask, scale=[2.735, 1, 1])
# viewer.add_labels(labels, scale=[2.735, 1, 1])
# viewer.add_image(test, scale=[2.735, 1, 1])