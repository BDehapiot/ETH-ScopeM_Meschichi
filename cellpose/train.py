#%% Imports -------------------------------------------------------------------

import napari
import random
import numpy as np
from skimage import io
from pathlib import Path
import albumentations as A
from datetime import datetime
import matplotlib.pyplot as plt
from joblib import Parallel, delayed 

from cellpose import models
from cellpose.io import logger_setup
logger_setup();

#%% Parameters ----------------------------------------------------------------

random.seed(42) 

# Paths
data_path = Path(Path.cwd(), 'data')
train_path = Path(Path.cwd(), 'data', 'train') 

# Data augmentation
iterations = 0
augment = True if iterations > 0 else False
operations = A.Compose([
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Transpose(p=0.5),
    A.GridDistortion(p=0.5),
    ])

# Training
valid_split = 0.25
n_epochs = 50

#%% Prepare data --------------------------------------------------------------

# Open training data
images, masks = [], []
for path in train_path.iterdir():
    if 'mask' in path.name:
        mask_path = str(path)
        image_path = mask_path.replace('_mask', '')
        images.append(io.imread(image_path))
        masks.append(io.imread(mask_path))
        
# # Display 
# viewer = napari.Viewer()
# viewer.add_image(np.stack(images))
# viewer.add_labels(np.stack(masks))
        
# -----------------------------------------------------------------------------

if augment:

    # Augment data
    def augment_data(images, masks, operations):      
        idx = random.randint(0, len(images) - 1)
        outputs = operations(image=images[idx], mask=masks[idx])
        return outputs['image'], outputs['mask']
    outputs = Parallel(n_jobs=-1)(
        delayed(augment_data)(images, masks, operations)
        for i in range(iterations)
        )
    images = [data[0] for data in outputs] 
    masks = [data[1] for data in outputs] 
    
    # Display 
    viewer = napari.Viewer()
    viewer.add_image(np.stack(images))
    viewer.add_labels(np.stack(masks))
    
# -----------------------------------------------------------------------------
    
# Split data  
train_images, train_masks, train_paths = [], [], []
valid_images, valid_masks, valid_paths = [], [], []
for i, (image, mask) in enumerate(zip(images, masks)):
    if i >= len(images) * valid_split:
        train_images.append(np.array(image))
        train_masks.append(np.array(mask))
    else:
        valid_images.append(np.array(image))
        valid_masks.append(np.array(mask))

#%% Train model ---------------------------------------------------------------

model = models.CellposeModel(gpu=True, model_type='nuclei')
model.train(
    train_data=train_images, 
    train_labels=train_masks, 
    test_data=valid_images, 
    test_labels=valid_masks, 
    channels=[0,0], 
    normalize=True,
    save_path=data_path,
    n_epochs=n_epochs,
    min_train_masks=1,
    model_name= (
        f'{datetime.today().date()}_model_'
        f'aug({augment})_'
        f'img({len(images)})_'
        f'split({valid_split})_'
        f'epochs({n_epochs})'
        )
    )

#%% 

import nd2

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
