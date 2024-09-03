#%% Imports -------------------------------------------------------------------

from skimage import io
from pathlib import Path
from functions import open_stack, display

#%% Parameters ----------------------------------------------------------------

# Path
# data_path = Path(Path.cwd(), "data", "local") 
data_path = Path("D:/local_Meschichi/data")
# data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Meschichi\data")

# Select (stack)
stack_name = "KZLind3" # if batch == False 

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    # Load data
    stack, metadata = open_stack(data_path / (stack_name + ".nd2"))
    nLabels = io.imread(data_path / stack_name / (stack_name + "_nLabels.tif"))
    cLabels = io.imread(data_path / stack_name / (stack_name + "_cLabels.tif"))
    tophat = io.imread(data_path / stack_name / (stack_name + "_tophat.tif"))

    # Display
    display(stack, metadata, nLabels, cLabels, tophat)
    
