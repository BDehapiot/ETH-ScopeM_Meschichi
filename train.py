#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path

# bdtools
from bdtools.models import preprocess, augment
from bdtools.models.unet import UNet

#%% Inputs --------------------------------------------------------------------

# preprocess()
patch_size = 256
patch_overlap = 0
img_norm = "global"
msk_type = "edt"

# UNet build()
save_name = f"{patch_size}_{msk_type}"
backbone = "resnet18"
activation = "sigmoid"
downscale_steps = 1

# UNet train()
epochs = 100
batch_size = 32 
validation_split = 0.2
metric = "soft_dice_coef"
learning_rate = 0.0005
patience = 20

#%% Initialize ----------------------------------------------------------------

train_path = Path("data", "train")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Load data
    imgs, msks = [], []
    for path in list(train_path.glob("*.tif")):
        if "mask" in path.name:
            msks.append(io.imread(path))   
            imgs.append(io.imread(str(path).replace("_mask", "")))
    imgs = np.stack(imgs)
    msks = np.stack(msks)
     
    # Preprocess
    imgs, msks = preprocess(
        imgs, msks=msks, 
        patch_size=patch_size, 
        patch_overlap=patch_overlap,
        img_norm=img_norm,
        msk_type=msk_type,
        )
        
    # # Augment
    # imgs, msks = augment(
    #     imgs, msks, 5000, 
    #     gamma_p=0.0, gblur_p=0.0, noise_p=0.0, flip_p=0.5, distord_p=0.5
    #     )
    
    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(imgs)
    # viewer.add_image(msks)
    
    # Train
    unet = UNet(
        save_name=save_name,
        backbone=backbone,
        activation=activation,
        downscale_steps=downscale_steps, 
        )

    unet.train(
        imgs, msks, 
        X_val=None, y_val=None,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        metric=metric,
        learning_rate=learning_rate,
        patience=patience,
        )