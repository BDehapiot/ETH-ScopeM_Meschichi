#%% Imports -------------------------------------------------------------------

import numpy as np
from pathlib import Path

# functions
from functions import open_stack, format_stack

# bdmodel
from bdmodel.predict import predict


#%% Parameters ----------------------------------------------------------------

# Path
data_path = Path("D:/local_Meschichi/data")
# data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Meschichi\data")

#%% Function(s) ---------------------------------------------------------------

# Batch
# batch = False
# overwrite = True
# stack_name = "FIN_01_00"
# stack_name = "KAS_01_02"
# stack_name = "KZL_01_00"
stack_name = "KZL_01_03"

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    path = data_path / (stack_name + ".nd2")
    
    # Load & format data
    stack, metadata = open_stack(path)
    rscale, rslice = format_stack(stack, metadata)
    
    # Predict
    rscale_prds = predict(
        rscale, 
        Path("bdmodel", "model_rscale_128"), 
        img_norm="global",
        patch_overlap=0,
        )
    rslice_prds = predict(
        rslice, 
        Path("bdmodel", "model_rslice_128"),
        img_norm="global",
        patch_overlap=0,
        )
    
    # Merge predictions
    rslice_prds = np.swapaxes(rslice_prds, 0, 1)
    prds_avg = np.mean(np.stack((rscale_prds, rslice_prds), axis=0), axis=0) 
    prds_std = np.std(np.stack((rscale_prds, rslice_prds), axis=0), axis=0) 
    prds = prds_avg - prds_std
    
    # Display
    # import napari
    # viewer = napari.Viewer()
    # viewer.add_image(rscale)
    # viewer.add_image(prds_avg)
    # viewer.add_image(prds_std)
    # viewer.add_image(prds)
    
    # viewer.add_image(rscale)
    # viewer.add_image(rscale_prds)
    # viewer.add_image(rslice)
    # viewer.add_image(rslice_prds)


    
