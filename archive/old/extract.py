#%% Imports -------------------------------------------------------------------

import nd2
from skimage import io
from pathlib import Path
import random 

#%% Parameters ----------------------------------------------------------------

random.seed(44)

# Paths
data_path = Path('D:/local_Meschichi/data')
stock_path = Path(Path.cwd(), 'data', 'stock') 

#%% Extract -------------------------------------------------------------------

for stack_path in data_path.iterdir():
    if stack_path.suffix == '.nd2':
        
        # open stack
        stack = nd2.imread(stack_path) 
        stack_name = stack_path.stem
        
        # extract random z_plane
        z_rand = random.randint(0, stack.shape[0] - 1)
        z_plane = stack[z_rand,...]
        
        # save extracted z_plane
        io.imsave(
            Path(stock_path, f'{stack_name}_z{z_rand}.tif'),
            z_plane, check_contrast=False,
            )