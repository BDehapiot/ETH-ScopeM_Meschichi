#%% Imports -------------------------------------------------------------------

import nd2
import napari
import numpy as np
from skimage import io
from pathlib import Path

from cellpose import models
from cellpose.io import logger_setup
logger_setup();

#%% Parameters ----------------------------------------------------------------

# Paths
data_path = Path('D:/local_Meschichi/data')
model_path = Path(Path.cwd(), 'data', 'model') 
model_name = '2023-08-31_model_aug(False)_img(31)_split(0.25)_epochs(50)'
stack_name = 'KASind1.nd2'

#%% Predict -------------------------------------------------------------------
   
# Open model  
model = models.CellposeModel(
    pretrained_model=str(Path(model_path) / model_name),
    gpu=True
    )
    
# Paths
data_path = Path('D:/local_Meschichi/data')
model_path = Path(Path.cwd(), 'data', 'model') 
model_name = '2023-08-31_model_aug(False)_img(31)_split(0.25)_epochs(50)'
stack_name = 'KASind1.nd2'

# Predict
images = list(nd2.imread(Path(data_path) / stack_name).squeeze())  
predict_data = model.eval(
    images,
    channels=[0,0], 
    normalize=True,
    )

# # Display 
labels = np.stack([data for data in predict_data[0]])
probs = np.stack([data[2] for data in predict_data[1]])
viewer = napari.Viewer()
viewer.add_image(np.stack(images), scale=[5.47, 1, 1])
viewer.add_labels(labels, scale=[5.47, 1, 1])
viewer.add_image(probs, scale=[5.47, 1, 1])

