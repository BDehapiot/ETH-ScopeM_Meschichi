#%% Imports -------------------------------------------------------------------

import random 
from skimage import io
from pathlib import Path

# Functions
from functions import open_stack, format_stack

#%% Inputs --------------------------------------------------------------------

# Paths
local_path = Path('D:/local_Meschichi/data')
stock_path = Path(Path.cwd(), 'data', 'stock') 

# Parameters
random.seed(42)
nPlane = 5

#%% Extract -------------------------------------------------------------------

for path in local_path.iterdir():
    if path.suffix == '.nd2':
                
        # Open & format data
        stack, metadata = open_stack(path)
        rscale, rslice = format_stack(stack, metadata)
          
        # Select and save random plane (2D)
        for i in range(nPlane): 
            rand1 = random.randint(0, rscale.shape[0] - 1)
            rand2 = random.randint(0, rslice.shape[0] - 1)
            io.imsave(
                Path(stock_path, f'{path.stem}_rscale_{rand1:03d}.tif'),
                rscale[rand1,...].astype("float32"), check_contrast=False,
                )
            io.imsave(
                Path(stock_path, f'{path.stem}_rslice_{rand2:03d}.tif'),
                rslice[rand2,...].astype("float32"), check_contrast=False,
                )
