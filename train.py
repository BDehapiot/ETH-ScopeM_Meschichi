#%% Imports -------------------------------------------------------------------

import nd2
import napari
import random
import numpy as np
from skimage import io
from pathlib import Path
import albumentations as A
import matplotlib.pyplot as plt
from joblib import Parallel, delayed 

from cellpose import models
from cellpose.io import imread
from cellpose.io import logger_setup
logger_setup();

random.seed(42) 

#%% Parameters ----------------------------------------------------------------

val_split = 0.25

# Data augmentation
iterations = 64
operations = A.Compose([
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Transpose(p=0.5),
    A.GridDistortion(p=0.5),
    ])

#%%

# Open training data
trn_path = Path('D:\\local_Meschichi\\data\\train')
images, masks = [], []
for path in trn_path.iterdir():
    if 'mask' in path.name:
        mask_path = str(path)
        image_path = mask_path.replace('_mask', '')
        images.append(io.imread(image_path))
        masks.append(io.imread(mask_path))
        
# # Display 
# viewer = napari.Viewer()
# viewer.add_image(np.stack(images))
# viewer.add_labels(np.stack(masks))
        
# Augment data
def augment_data(images, masks, operations):      
    idx = random.randint(0, len(images) - 1)
    outputs = operations(image=images[idx], mask=masks[idx])
    return outputs['image'], outputs['mask']
outputs = Parallel(n_jobs=-1)(
    delayed(augment_data)(images, masks, operations)
    for i in range(iterations)
    ) 

# Split data  
trn_images, trn_masks, trn_paths = [], [], []
val_images, val_masks, val_paths = [], [], []
for i, (image, mask) in enumerate(outputs):
    if i >= iterations * val_split:
        trn_images.append(np.array(image))
        trn_masks.append(np.array(mask))
    else:
        val_images.append(np.array(image))
        val_masks.append(np.array(mask))

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(np.stack(trn_images))
# viewer.add_labels(np.stack(trn_masks))

#%% Train ---------------------------------------------------------------------

model = models.CellposeModel(gpu=True, model_type='nuclei')
test = model.train(
    train_data=trn_images, 
    train_labels=trn_masks, 
    test_data=val_images, 
    test_labels=val_masks, 
    channels=[0,0], 
    normalize=False,
    save_path='D:\\local_Meschichi\\data\\train\\',
    n_epochs=50,
    min_train_masks=1,
    model_name='current',
    )

#%% Predict -------------------------------------------------------------------
      
# Open prediction images
prd_images = []
stack_name = 'KASind1.nd2'
prd_path = Path('D:\\local_Meschichi\\data')
stack = nd2.imread(Path(prd_path) / stack_name).squeeze()
for z in stack:
    prd_images.append(z)

# Predict
prd_data = model.eval(
    prd_images,
    channels=[0,0], 
    normalize=False,
    anisotropy=5.47,
    )

# Display 
prd_images = np.stack(prd_images)
prd_labels = np.stack([data for data in prd_data[0]])
prd_mask = prd_labels > 0
prd_probs = np.stack([data[2] for data in prd_data[1]])
viewer = napari.Viewer()
viewer.add_image(prd_images, scale=[5.47, 1, 1])
viewer.add_labels(prd_labels, scale=[5.47, 1, 1])
viewer.add_labels(prd_mask, scale=[5.47, 1, 1])
viewer.add_image(prd_probs, scale=[5.47, 1, 1])