#%% Imports -------------------------------------------------------------------

import random 
from skimage import io
from pathlib import Path

# Functions
from functions import open_stack

#%% Inputs --------------------------------------------------------------------

# Parameters
random.seed(42)
nPlane = 2

#%% Initialize ----------------------------------------------------------------

# Paths
data_path = Path("D:/local_Meschichi/data")
train_path = Path(Path.cwd(), 'data', 'train') 

#%% Extract -------------------------------------------------------------------

for path in data_path.iterdir():
    if path.suffix == '.nd2':
                
        # Open & format data
        stack, metadata = open_stack(path)
        
        # Select and save random plane(s) (2D)
        for i in range(nPlane):
            z = random.randint(0, stack.shape[0] - 1)
            io.imsave(
                Path(train_path, f'{path.stem}_z{z:02d}.tif'),
                stack[z, ...].astype("uint16"), check_contrast=False,
                )
            
#%% Extract NEO ---------------------------------------------------------------

max_z = 4

for path in data_path.iterdir():
    if path.suffix == '.nd2' and "NEO" in path.name:
                
        # Open & format data
        stack, metadata = open_stack(path)
          
        # Select and save random plane (2D)
        for i in range(nPlane): 
            z = random.randint(0, max_z - 1)
            io.imsave(
                Path(train_path, f'{path.stem}_z{z:02d}.tif'),
                stack[z,...].astype("uint16"), check_contrast=False,
                )