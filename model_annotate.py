#%% Imports -------------------------------------------------------------------

import random
import napari
import numpy as np
from skimage import io
from pathlib import Path

#%% Inputs --------------------------------------------------------------------

# Paths
local_path = Path('D:/local_Meschichi/data')
stock_path = Path(Path.cwd(), 'data', 'stock') 

# Parameters
random.seed(42)
contrast_limits = (0, 1)
brush_size = 5

#%%

metadata = []
for path in stock_path.iterdir():
    if "mask" not in path.name:
        metadata.append({
            "name"  : path.name,
            "path"  : path,
            })
  
#%%

# initialize viewer
viewer = napari.Viewer()
viewer.text_overlay.visible = True

# Open first image
while True:
    idx   = random.randint(0, len(metadata))
    path  = metadata[idx]["path"]
    if not Path(str(path).replace(".tif", "_mask.tif")).exists():
        image = io.imread(path)
        mask  = np.zeros_like(image, dtype="uint8")
        viewer.add_image(image, name="image", metadata=metadata[idx])
        viewer.add_labels(mask, name="mask")
        viewer.layers["image"].contrast_limits = contrast_limits
        viewer.layers["mask"].brush_size = brush_size
        viewer.layers["mask"].mode = 'paint'
        viewer.text_overlay.text = path.name
        break 
    
def next_image():
    
    # Save previous mask
    path = viewer.layers["image"].metadata["path"]
    path = Path(str(path).replace(".tif", "_mask.tif"))
    io.imsave(path, viewer.layers["mask"].data, check_contrast=False)  
    
    # Open next image
    while True:
        idx   = random.randint(0, len(metadata))
        path  = metadata[idx]["path"]
        if not Path(str(path).replace(".tif", "_mask.tif")).exists():
            image = io.imread(path)
            mask  = np.zeros_like(image, dtype="uint8")
            viewer.layers["image"].data = image
            viewer.layers["image"].metadata = metadata[idx]
            viewer.layers["mask" ].data = mask
            viewer.text_overlay.text = path.name
            viewer.reset_view()
            break 
    
# Shortcut
@napari.Viewer.bind_key('Enter', overwrite=True)
def next_image_key(viewer):
    next_image()

napari.run()    
