#%% Imports -------------------------------------------------------------------

import nd2
import napari
import random
import numpy as np
from skimage import io
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.transform import rescale
from skimage.measure import regionprops
from scipy.ndimage import distance_transform_edt

#%% Parameters ----------------------------------------------------------------

random.seed(42) 

# Paths
data_path = Path(Path.cwd(), 'data')
train_path = Path(data_path, 'train') 

# Training
rescale_factor = 0.5
validation_split = 0.33
n_epochs = 30
batch_size = 4

#%% Prepare data --------------------------------------------------------------

# Open training data
train_images, train_labels = [], []
for path in train_path.iterdir():
    if 'mask' in path.name:
        
        # Get paths
        labels_path = str(path)
        image_path = labels_path.replace('_mask', '')
        
        # Open data
        image = io.imread(image_path)
        labels = io.imread(labels_path)

        # Open & rescale data
        image = rescale(image, rescale_factor, preserve_range=True)
        labels = rescale(labels, rescale_factor, order=0, preserve_range=True)
        
        # Append train_images & train_labels lists
        train_images.append(image)
        train_labels.append(labels)
                
# Format training data
train_images = np.stack(train_images)
train_labels = np.stack(train_labels)
pMax = np.percentile(train_images, 99.9)
train_images[train_images > pMax] = pMax
train_images = (train_images / pMax).astype(float)

# Get edm data
train_edm = []
for labels in train_labels:
    props = regionprops(labels)
    edm = labels > 0
    edm = distance_transform_edt(edm)
    train_edm.append(edm)
train_edm = np.stack(train_edm)
train_edm = train_edm / np.max(train_edm)

# viewer = napari.Viewer()
# viewer.add_image(train_edm)
# viewer.add_image(train_labels)
 
#%% Train model ---------------------------------------------------------------

train_images = train_images[:, :, :, np.newaxis]
train_edm = train_edm[:, :, :, np.newaxis]

# Create a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(512, 512, 1)),  # Grayscale image, values [0, 1]
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D(),
    tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss')]
history = model.fit(
    x=train_images,
    y=train_edm,
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

# # Save model
# model.save_weights(Path(Path.cwd(), 'model_weights.h5'))

#%% Predict -------------------------------------------------------------------

# Paths
data_path = Path('D:/local_Meschichi/data')
data_path = Path(Path.cwd(), 'data', 'local')
# stack_name = 'KASind1.nd2'
stack_name = 'KZLind1.nd2'

# Training
rescale_factor = 0.5

# Open & format prediction data
predict_images = nd2.imread(Path(data_path) / stack_name).squeeze()  
predict_images = rescale(predict_images, (1, rescale_factor, rescale_factor), preserve_range=True)
pMax = np.percentile(predict_images, 99.9)
predict_images[predict_images > pMax] = pMax
predict_images = (predict_images / pMax)
predict_images = predict_images[:, :, :, np.newaxis]

# Predict
probs = model.predict(predict_images).squeeze()  

# Display 
viewer = napari.Viewer()
viewer.add_image(predict_images, scale=[2.735, 1, 1])
viewer.add_image(probs, scale=[2.735, 1, 1])
