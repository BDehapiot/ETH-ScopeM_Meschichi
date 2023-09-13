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
from skimage.transform import rescale
from skimage.measure import regionprops

import tensorflow as tf
from tensorflow import keras
from keras_unet_collection import models

#%% Parameters ----------------------------------------------------------------

random.seed(42) 

# Paths
data_path = Path(Path.cwd(), 'data')
train_path = Path(data_path, 'train') 

# Training
rescale_factor = 0.5
validation_split = 0.2
n_epochs = 50
batch_size = 4

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

#%% Prepare data --------------------------------------------------------------

# Open training data
images, labels, masks = [], [], []
for path in train_path.iterdir():
    if 'labels' in path.name:
        
        # Get paths
        label_path = str(path)
        image_path = label_path.replace('_labels', '')
        
        # Open data
        image = io.imread(image_path)
        label = io.imread(label_path)
        mask = label > 0

        # Open & rescale data
        image = rescale(image, rescale_factor, preserve_range=True)
        label = rescale(label, rescale_factor, order=0, preserve_range=True)
        mask = rescale(mask, rescale_factor, order=0, preserve_range=True)
        
        # Append lists
        images.append(image)
        labels.append(label)
        masks.append(mask)
                
# Format training data
images = np.stack(images)
labels = np.stack(labels)
masks = np.stack(masks).astype(float)
pMax = np.percentile(images, 99.9)
images[images > pMax] = pMax
images = (images / pMax).astype(float)

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(images)
# viewer.add_image(masks)

# -----------------------------------------------------------------------------

if augment:

    # Augment data
    def augment_data(images, masks, operations):      
        idx = random.randint(0, len(images) - 1)
        outputs = operations(image=images[idx,...], mask=masks[idx,...])
        return outputs['image'], outputs['mask']
    outputs = Parallel(n_jobs=-1)(
        delayed(augment_data)(images, masks, operations)
        for i in range(iterations)
        )
    images = np.stack([data[0] for data in outputs])
    masks = np.stack([data[1] for data in outputs])
    
    # # Display 
    # viewer = napari.Viewer()
    # viewer.add_image(images)
    # viewer.add_image(masks)

#%% Train model ---------------------------------------------------------------

# Define & compile model
model = models.unet_2d(
    (None, None, 1), 
    [64, 128, 256, 512, 1024], 
    n_labels=1,
    stack_num_down=2,
    stack_num_up=1,
    activation='ReLU',
    output_activation='Softmax',
    )
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', 
    metrics=['mse']
    )

# Train model
callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss')]
history = model.fit(
    x=images,
    y=masks,
    validation_split=validation_split,
    batch_size=batch_size,
    epochs=n_epochs,
    callbacks=callbacks,
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
