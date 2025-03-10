![Python Badge](https://img.shields.io/badge/Python-3.10-rgb(69%2C132%2C182)?logo=python&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))
![TensorFlow Badge](https://img.shields.io/badge/TensoFlow-2.10-rgb(255%2C115%2C0)?logo=TensorFlow&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))
![CUDA Badge](https://img.shields.io/badge/CUDA-11.2-rgb(118%2C185%2C0)?logo=NVIDIA&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))
![cuDNN Badge](https://img.shields.io/badge/cuDNN-8.1-rgb(118%2C185%2C0)?logo=NVIDIA&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))    
![Author Badge](https://img.shields.io/badge/Author-Benoit%20Dehapiot-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))
![Date Badge](https://img.shields.io/badge/Created-2023--07--26-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))
![License Badge](https://img.shields.io/badge/Licence-GNU%20General%20Public%20License%20v3.0-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))    

# ETH-ScopeM_Meschichi  
3D segmentation of plant nuclei sub-domains

## Index
- [Installation](#installation)
- [Outputs](#outputs)
- [Comments](#comments)

## Installation

Pease select your operating system

<details> <summary>Windows</summary>  

### Step 1: Download this GitHub Repository 
- Click on the green `<> Code` button and download `ZIP` 
- Unzip the downloaded file to a desired location

### Step 2: Install Miniforge (Minimal Conda installer)
- Download and install [Miniforge](https://github.com/conda-forge/miniforge) for your operating system   
- Run the downloaded `.exe` file  
    - Select "Add Miniforge3 to PATH environment variable"  

### Step 3: Setup Conda 
- Open the newly installed Miniforge Prompt  
- Move to the downloaded GitHub repository
- Run one of the following command:  
```bash
# TensorFlow with GPU support
mamba env create -f environment_tf_gpu.yml
# TensorFlow with no GPU support 
mamba env create -f environment_tf_nogpu.yml
```  
- Activate Conda environment:
```bash
conda activate Meschichi
```
Your prompt should now start with `(Meschichi)` instead of `(base)`

</details> 

<details> <summary>MacOS</summary>  

### Step 1: Download this GitHub Repository 
- Click on the green `<> Code` button and download `ZIP` 
- Unzip the downloaded file to a desired location

### Step 2: Install Miniforge (Minimal Conda installer)
- Download and install [Miniforge](https://github.com/conda-forge/miniforge) for your operating system   
- Open your terminal
- Move to the directory containing the Miniforge installer
- Run one of the following command:  
```bash
# Intel-Series
bash Miniforge3-MacOSX-x86_64.sh
# M-Series
bash Miniforge3-MacOSX-arm64.sh
```   

### Step 3: Setup Conda 
- Re-open your terminal 
- Move to the downloaded GitHub repository
- Run one of the following command: 
```bash
# TensorFlow with GPU support
mamba env create -f environment_tf_gpu.yml
# TensorFlow with no GPU support 
mamba env create -f environment_tf_nogpu.yml
```  
- Activate Conda environment:  
```bash
conda activate Meschichi
```
Your prompt should now start with `(Meschichi)` instead of `(base)`

</details>

## Outputs

### nData.csv - nuclei data

```bash
# Nuclei data
    - nLabel    # IDs
    - nVolume   # volume (µm3)  
    - nCtrd     # centroid coords (zyx)
    - nMajor    # fitted ellipse major axis length (µm)
    - nMinor    # fitted ellipse minor axis length (µm)
    - nMMRatio  # minor / major axis ratio

# Associated chromocenters data
    - n_cLabel  # associated chromocenter IDs
    - n_cNumber # number of cromocenters
    - n_cVolume # cumulative volume of chromocenters (µm3)
    - n_cnRatio # chromocenter / nuclei volume ratio
```

### cData.csv - chromocenters data

```bash
# Chromocenters data data
    - cLabel    # IDs
    - nVolume   # volume (µm3)  
    - cCtrd     # centroid coords (zyx)
    - cMajor    # fitted ellipse major axis length (µm)
    - cMinor    # fitted ellipse minor axis length (µm)
    - cMMRatio  # minor / major axis ratio
    - cInt      # intensity mean (raw intensities)
    - cEDMb     # distance from nucleus periphery (µm)
    - cEDMc     # distance from nucleus centroid (µm)

# Associated nuclei data
    - c_nLabel  # associated nucleus ID
```

## Comments
### Meeting 24/05/2024
- Quantifications    
    - number of chromocenters (**done**)
    - volume of nuclei, chromocenters and ratio (**done**)
    - brightness of chromocenters DAPI (background substraction ?) (**done**)
    - shape descriptors (nuclei & chromocenters) (**done**)
    - distance nuclei border vs. chromocenter centroid (**done**)
    - distance nuclei centroid vs. chromocenter centroid (**done**)
    - statistics (avg, variance, std...)
- Other
    - create a remote folder (**done**)