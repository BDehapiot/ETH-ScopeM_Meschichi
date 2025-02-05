#%% Imports -------------------------------------------------------------------

import random 
from skimage import io
from pathlib import Path

# Functions
from functions import open_stack, format_stack

#%% Inputs --------------------------------------------------------------------

# Paths
remote_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Meschichi\data")
train_path = Path(Path.cwd(), 'data', 'train') 

# Parameters
random.seed(42)
nPlane = 1

#%% Extract -------------------------------------------------------------------

# for path in remote_path.iterdir():
#     if path.suffix == '.nd2':
                
#         # Open & format data
#         stack, metadata = open_stack(path)
#         rscale, rslice = format_stack(stack, metadata)
          
#         # Select and save random plane (2D)
#         for i in range(nPlane): 
#             rand1 = random.randint(0, rscale.shape[0] - 1)
#             rand2 = random.randint(0, rslice.shape[0] - 1)
#             io.imsave(
#                 Path(train_path, f'{path.stem}_rscale_{rand1:03d}.tif'),
#                 rscale[rand1,...].astype("uint16"), check_contrast=False,
#                 )
#             io.imsave(
#                 Path(train_path, f'{path.stem}_rslice_{rand2:03d}.tif'),
#                 rslice[rand2,...].astype("uint16"), check_contrast=False,
#                 )
            
#%% Extract NEO ---------------------------------------------------------------

z = 4

for path in remote_path.iterdir():
    if path.suffix == '.nd2' and "NEO" in path.name:
                
        # Open & format data
        stack, metadata = open_stack(path)
        rscale, rslice = format_stack(stack, metadata)
          
        # Select and save random plane (2D)
        for i in range(nPlane): 
            io.imsave(
                Path(train_path, f'{path.stem}_rscale_{z:03d}.tif'),
                rscale[z,...].astype("uint16"), check_contrast=False,
                )