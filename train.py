#%% Imports -------------------------------------------------------------------

import nd2
import napari
import random
import numpy as np
from skimage import io
from pathlib import Path
import albumentations as A
import matplotlib.pyplot as plt
import segmentation_models as sm
from joblib import Parallel, delayed 
from skimage.transform import rescale

#%% Parameters ----------------------------------------------------------------

random.seed(42) 

# Paths
data_path = Path(Path.cwd(), 'data')
train_path = Path(data_path, 'train') 

# Training
rescale_factor = 0.5
validation_split = 0.2
n_epochs = 30
batch_size = 8

# Data augmentation
iterations = 60
augment = True if iterations > 0 else False
operations = A.Compose([
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Transpose(p=0.5),
    A.GridDistortion(p=0.5),
    ])

#%% Prepare data --------------------------------------------------------------

# Open training data
train_images, train_masks = [], []
for path in train_path.iterdir():
    if 'mask' in path.name:
        
        # Get paths
        mask_path = str(path)
        image_path = mask_path.replace('_mask', '')
        
        # Open data
        image = io.imread(image_path)
        mask = io.imread(mask_path) > 0

        # Open & rescale data
        image = rescale(image, rescale_factor, preserve_range=True)
        mask = rescale(mask, rescale_factor, preserve_range=True)
        
        # Append train_images & train_masks lists
        train_images.append(image)
        train_masks.append(mask)
                
# Format training data
train_images = np.stack(train_images)
train_masks = np.stack(train_masks).astype('uint8')
pMax = np.percentile(train_images, 99.9)
train_images[train_images > pMax] = pMax
train_images = ((train_images / pMax)*255).astype('uint8')

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(train_images)
# viewer.add_image(train_masks)
       
# -----------------------------------------------------------------------------

if augment:

    # Augment data
    def augment_data(train_images, train_masks, operations):      
        idx = random.randint(0, len(train_images) - 1)
        outputs = operations(image=train_images[idx,...], mask=train_masks[idx,...])
        return outputs['image'], outputs['mask']
    outputs = Parallel(n_jobs=-1)(
        delayed(augment_data)(train_images, train_masks, operations)
        for i in range(iterations)
        )
    train_images = np.stack([data[0] for data in outputs])
    train_masks = np.stack([data[1] for data in outputs])
    
    # # Display 
    # viewer = napari.Viewer()
    # viewer.add_image(train_images)
    # viewer.add_labels(train_masks)                    

#%% Train model ---------------------------------------------------------------

from tensorflow.keras.utils import normalize
train_images = normalize(train_images)

print(np.max(train_images))

# Define & compile model
preprocess_input = sm.get_preprocessing('resnet34')
model = sm.Unet(
    'resnet34', 
    input_shape=(None, None, 1), 
    classes=1, 
    activation='sigmoid', 
    encoder_weights=None,
    )
model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
    )

# Train model
history = model.fit(
    x=train_images,
    y=train_masks,
    validation_split=validation_split,
    batch_size=batch_size,
    epochs=n_epochs,
)

# Plot training results
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%% Predict -------------------------------------------------------------------

# # Paths
# data_path = Path('D:/local_Meschichi/data')
# stack_name = 'KASind1.nd2'

# # Open & format prediction data
# predict_images = nd2.imread(Path(data_path) / stack_name).squeeze()  
# predict_images = rescale(predict_images, (1, rescale_factor, rescale_factor), preserve_range=True)
# pMax = np.percentile(predict_images, 99.9)
# predict_images[predict_images > pMax] = pMax
# predict_images = ((predict_images / pMax)*255).astype('uint8')

# # Predict
# probs = model.predict(predict_images).squeeze()  

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(predict_images)
# viewer.add_image(probs)

