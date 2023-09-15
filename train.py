#%% Imports -------------------------------------------------------------------

import nd2
import napari
import random
import numpy as np
from skimage import io
import tensorflow as tf
from pathlib import Path
import albumentations as A
import matplotlib.pyplot as plt
import segmentation_models as sm
from joblib import Parallel, delayed 
from skimage.transform import rescale

#%% Initialize ----------------------------------------------------------------

# Get paths
data_path = Path(Path.cwd(), 'data')
train_path = Path(data_path, 'train') 

#%% Parameters ----------------------------------------------------------------

# Data pre-processing
rescale_factor = 0.5

# Data augmentation
iterations = 100
random.seed(42) 

# Train model
validation_split = 0.2
n_epochs = 100
batch_size = 8

#%% Pre-processing ------------------------------------------------------------

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
       
#%% Augmentation --------------------------------------------------------------

augment = True if iterations > 0 else False

if augment:
    
    # Define augmentation operations
    operations = A.Compose([
        A.VerticalFlip(p=0.5),              
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.GridDistortion(p=0.5),
        ])

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
    # viewer.add_image(train_images)
    # viewer.add_labels(train_masks)     

#%% Model training ------------------------------------------------------------

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

# Train model
callbacks = [tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss')]
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

# Save model
model.save_weights(Path(Path.cwd(), f'model_weights_{rescale_factor}.h5'))

#%% Predictions ---------------------------------------------------------------

# Paths
data_path = Path(Path.cwd(), 'data', 'local')
stack_name = 'KASind1.nd2'
stack_name = 'KZLind1.nd2'

# Open & format prediction data
stack = nd2.imread(Path(data_path) / stack_name).squeeze()  
stack = rescale(stack, (1, rescale_factor, rescale_factor), preserve_range=True)
pMax = np.percentile(stack, 99.9)
stack[stack > pMax] = pMax
stack = (stack / pMax)

# Predict
probs = model.predict(stack).squeeze()  

# Display 
viewer = napari.Viewer()
viewer.add_image(stack)
viewer.add_image(probs)   