#%% Imports -------------------------------------------------------------------

import nd2
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
import segmentation_models as sm
from joblib import Parallel, delayed

# Skimage
from skimage.feature import peak_local_max
from skimage.transform import rescale, resize
from skimage.measure import label, regionprops
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import clear_border, expand_labels
from skimage.morphology import (
    disk, remove_small_objects, white_tophat
    )

# Scipy
from scipy.ndimage import distance_transform_edt

#%% Functions (GPU) -----------------------------------------------------------

def limit_vram(vram):

    import tensorflow as tf    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    memory_config = tf.config.experimental\
        .VirtualDeviceConfiguration(memory_limit=vram)
    
    if gpus:
        try:
            tf.config.experimental\
                .set_virtual_device_configuration(gpus[0], [memory_config])
        except RuntimeError as e:
            print(e)

#%% Functions (open_stack) ----------------------------------------------------

def open_stack(path, metadata=True):
    
    # Read nd2 file
    with nd2.ND2File(path) as ndfile:
        stack = ndfile.asarray()
        nZ, nY, nX = stack.shape
        vY, vX, vZ = ndfile.voxel_size()
    
    if metadata:
        
        metadata = {
            "nZ" : nZ, "nY" : nY, "nX" : nX, 
            "vZ" : vZ, "vY" : vY, "vX" : vX,
            }
    
        return stack, metadata
    
    return stack

#%% Functions (format_stack) --------------------------------------------------

def format_stack(stack, metadata, normalize=True):
           
    # Rescale & reslice (isotropic voxel)
    ratio = metadata["vY"] / metadata["vZ"]
    rscale = rescale(stack, (1, ratio, ratio), order=0)
    rslice = np.swapaxes(rscale, 0, 1)
    
    if normalize:
        pMax = np.percentile(rscale, 99.9)
        rscale[rscale > pMax] = pMax
        rslice[rslice > pMax] = pMax
        rscale = (rscale / pMax).astype(float)
        rslice = (rslice / pMax).astype(float)
        
    return rscale, rslice

#%% Functions (get_patches) ---------------------------------------------------

def get_patches(arr, size, overlap):
    
    # Get dimensions
    if arr.ndim == 2: nT = 1; nY, nX = arr.shape 
    if arr.ndim == 3: nT, nY, nX = arr.shape
    
    # Get variables
    y0s = np.arange(0, nY, size - overlap)
    x0s = np.arange(0, nX, size - overlap)
    yMax = y0s[-1] + size
    xMax = x0s[-1] + size
    yPad = yMax - nY
    xPad = xMax - nX
    yPad1, yPad2 = yPad // 2, (yPad + 1) // 2
    xPad1, xPad2 = xPad // 2, (xPad + 1) // 2
    
    # Pad array
    if arr.ndim == 2:
        arr_pad = np.pad(
            arr, ((yPad1, yPad2), (xPad1, xPad2)), mode='reflect') 
    if arr.ndim == 3:
        arr_pad = np.pad(
            arr, ((0, 0), (yPad1, yPad2), (xPad1, xPad2)), mode='reflect')         
    
    # Extract patches
    patches = []
    if arr.ndim == 2:
        for y0 in y0s:
            for x0 in x0s:
                patches.append(arr_pad[y0:y0 + size, x0:x0 + size])
    if arr.ndim == 3:
        for t in range(nT):
            for y0 in y0s:
                for x0 in x0s:
                    patches.append(arr_pad[t, y0:y0 + size, x0:x0 + size])
            
    return patches

#%% Functions (merge_patches) -------------------------------------------------

def merge_patches(patches, shape, size, overlap):
    
    # Get dimensions 
    if len(shape) == 2: nT = 1; nY, nX = shape
    if len(shape) == 3: nT, nY, nX = shape
    nPatch = len(patches) // nT

    # Get variables
    y0s = np.arange(0, nY, size - overlap)
    x0s = np.arange(0, nX, size - overlap)
    yMax = y0s[-1] + size
    xMax = x0s[-1] + size
    yPad = yMax - nY
    xPad = xMax - nX
    yPad1 = yPad // 2
    xPad1 = xPad // 2

    # Merge patches
    def _merge_patches(patches):
        count = 0
        arr = np.full((2, nY + yPad, nX + xPad), np.nan)
        for i, y0 in enumerate(y0s):
            for j, x0 in enumerate(x0s):
                if i % 2 == j % 2:
                    arr[0, y0:y0 + size, x0:x0 + size] = patches[count]
                else:
                    arr[1, y0:y0 + size, x0:x0 + size] = patches[count]
                count += 1 
        arr = np.nanmean(arr, axis=0)
        arr = arr[yPad1:yPad1 + nY, xPad1:xPad1 + nX]
        return arr
        
    if len(shape) == 2:
        arr = _merge_patches(patches)

    if len(shape) == 3:
        patches = np.stack(patches).reshape(nT, nPatch, size, size)
        arr = Parallel(n_jobs=-1)(
            delayed(_merge_patches)(patches[t,...])
            for t in range(nT)
            )
        arr = np.stack(arr)
        
    return arr

#%% Functions (predict) -------------------------------------------------------

def predict(rscale, rslice):
            
    # Define & compile model
    model = sm.Unet(
        'resnet18', # ResNet 18, 34, 50, 101 or 152 
        input_shape=(None, None, 1), 
        classes=1, 
        activation='sigmoid', 
        encoder_weights=None,
        )
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy', 
        metrics=['mse']
        )
    
    # Model paths
    for model_path in Path.cwd().iterdir():
        if "rscale_model_weights" in model_path.name: 
            rscale_model_path = model_path
        if "rslice_model_weights" in model_path.name: 
            rslice_model_path = model_path

    # Predict (rscale)
    model.load_weights(rscale_model_path) 
    size = int(rscale_model_path.stem.split("_")[-1].split("-")[0])
    overlap = int(rscale_model_path.stem.split("_")[-1].split("-")[1])    
    rscale_patches = np.stack(get_patches(rscale, size, overlap))
    rscale_probs = model.predict(rscale_patches, verbose=0).squeeze()
    rscale_probs = merge_patches(rscale_probs, rscale.shape, size, overlap)

    # Predict & merge patches (rslice)
    model.load_weights(rslice_model_path) 
    size = int(rslice_model_path.stem.split("_")[-1].split("-")[0])
    overlap = int(rslice_model_path.stem.split("_")[-1].split("-")[1])
    rslice_patches = np.stack(get_patches(rslice, size, overlap))
    rslice_probs = model.predict(rslice_patches, verbose=0).squeeze()
    rslice_probs = merge_patches(rslice_probs, rslice.shape, size, overlap)
    
    # Merge predictions
    rslice_probs = np.swapaxes(rslice_probs, 0, 1)
    probs = (rscale_probs + rslice_probs) / 2
    
    return probs

#%% Functions (segment) -------------------------------------------------------

def segment(
        path,
        # nMask
        lmax_dist=5,
        lmax_prom=0.15,
        prob_thresh=0.25,
        clear_nBorder=True,
        min_nSize=4096,
        # cMask
        tophat_size=3,
        tophat_sigma=1,
        tophat_tresh_coeff=1.25,
        min_cSize=32,
        ):

    def _get_tophat(plane):
        tophat = white_tophat(plane, footprint=disk(tophat_size)) 
        tophat = gaussian(tophat, sigma=tophat_sigma, preserve_range=True)
        return tophat
    
    # Load & format data
    stack, metadata = open_stack(path)
    rscale, rslice = format_stack(stack, metadata)
    
    # Predict
    probs = predict(rscale, rslice)
    
    # Nuclei
    lmax = peak_local_max(
        probs, exclude_border=False,
        min_distance=lmax_dist, threshold_abs=lmax_prom,
        )
    pMask = probs > 0.4 # parameter(s)
    pMask = remove_small_objects(pMask, min_size=8) # parameter(s)
    pMax = np.zeros_like(probs, dtype=int)
    pMax[(lmax[:, 0], lmax[:, 1], lmax[:, 2])] = 1
    nLabels = label(pMax)
    nLabels = expand_labels(nLabels, distance=10) 
    nLabels[pMask == 0] = 0
    nLabels = expand_labels(nLabels, distance=10)
    pMax = resize(pMax, stack.shape, order=0)
    nLabels = resize(nLabels, stack.shape, order=0)
    nMask = resize(probs, stack.shape, order=1) > prob_thresh
    if clear_nBorder:
        nMask = clear_border(nMask)
    nMask = remove_small_objects(nMask, min_size=min_nSize)
    nLabels[nMask == 0] = 0
    
    # Tophat transform
    tophat = Parallel(n_jobs=-1)(
        delayed(_get_tophat)(plane)
        for plane in stack
        )
    tophat = np.stack(tophat)

    # Chromocenters
    cMask = np.zeros_like(tophat, dtype=bool)
    for lab in np.unique(nLabels):
        if lab > 0:
            idx = (nLabels == lab)
            values = tophat[idx]
            thresh = threshold_otsu(values)
            cMask[idx] = (values > thresh * tophat_tresh_coeff)
    cMask = remove_small_objects(cMask, min_size=min_cSize)
    cLabels = label(cMask)
    
    # Format outputs
    outputs = {
        "stack"    : stack,
        "metadata" : metadata,
        "probs"    : probs.astype("float32"),
        "nLabels"  : nLabels.astype("uint16"),
        "cLabels"  : cLabels.astype("uint16"),
        "tophat"   : tophat.astype("float32"),
        }
    
    return outputs

#%% Functions (Measure) -------------------------------------------------------

def measure(outputs, rFactor=0.5):

    # Get variables
    stack = outputs["stack"]
    nLabels = outputs["nLabels"]
    cLabels = outputs["cLabels"]
    metadata = outputs["metadata"]
    
    # Rescale (isotropic voxel)
    ratio = metadata["vZ"] / metadata["vY"]
    stack = rescale(stack, (ratio * rFactor, rFactor, rFactor), order=0)
    nLabels = rescale(nLabels, (ratio * rFactor, rFactor, rFactor), order=0)
    cLabels = rescale(cLabels, (ratio * rFactor, rFactor, rFactor), order=0)
    cMask = cLabels > 0
    
    # nData -------------------------------------------------------------------
    
    nData = {
        
        # Nuclei
        "nLabel" : [], "nArea" : [], "nCtrd" : [],
        "nSolid" : [], "nMajor" : [], "nMinor" : [], "nMinMajRatio" : [],
        
        # Associated chromocenters
        "cLabels" : [], "cNumber" : [],
        "cArea" : [], "cnAreaRatio" : [],
        
        }
    
    for nProps in regionprops(nLabels):
        
        # Nuclei
        nData["nLabel"].append(nProps.label)
        nData["nArea"].append(nProps.area)
        nCtrd = nProps.centroid
        nCtrd = [int(ctrd) for ctrd in nCtrd]
        nData["nCtrd"].append(nCtrd)
        nData["nSolid"].append(nProps.solidity)
        nData["nMajor"].append(nProps.axis_major_length)
        nData["nMinor"].append(nProps.axis_minor_length)
        nData["nMinMajRatio"].append(
            nProps.axis_minor_length / nProps.axis_major_length)
    
        # Associated chromocenters    
        unique = np.unique(cLabels[nLabels == nProps.label])
        unique = unique[unique != 0]
        nData["cLabels"].append(unique)
        nData["cNumber"].append(unique.shape[0])
        cArea = np.sum(cMask[nLabels == nProps.label])
        nData["cArea"].append(cArea)
        nData["cnAreaRatio"].append(cArea / nProps.area)
    
    nData = pd.DataFrame(nData)
    
    # cData -------------------------------------------------------------------
    
    # Compute EDMs
    nCtrd = np.stack([ctrd for ctrd in nData["nCtrd"]])
    nCtrd = ((nCtrd[:, 0], nCtrd[:, 1], nCtrd[:, 2]))
    EDMb = distance_transform_edt(nLabels > 0) * (metadata["vY"] / rFactor)
    EDMc = np.zeros_like(nLabels, dtype=bool)
    EDMc[nCtrd] = True
    EDMc = distance_transform_edt(np.invert(EDMc)) * (metadata["vY"] / rFactor)
    EDMc[EDMb == 0] = 0
    
    cData = {
        
        # Chromocenters
        "cLabel" : [], "cArea" : [], "cCtrd" : [],
        "cSolid" : [], "cMajor" : [], "cMinor" : [], "cMinMajRatio" : [],
        "cInt" : [], "cEDMb" : [], "cEDMc" : [], 
        
        # Associated nuclei
        "nLabel" : [],
        
        }
    
    for cProps in regionprops(cLabels, intensity_image=stack):
        
        # Chromocenters
        cData["cLabel"].append(cProps.label)
        cData["cArea"].append(cProps.area)
        cCtrd = cProps.centroid
        cCtrd = [int(ctrd) for ctrd in cCtrd]
        cData["cCtrd"].append(cCtrd)
        cData["cSolid"].append(cProps.solidity)
        cData["cMajor"].append(cProps.axis_major_length)
        cData["cMinor"].append(cProps.axis_minor_length)
        cData["cMinMajRatio"].append(
            cProps.axis_minor_length / cProps.axis_major_length)
        cData["cInt"].append(cProps.intensity_mean) # To be discussed
        cData["cEDMb"].append(EDMb[cCtrd[0], cCtrd[1], cCtrd[2]])
        cData["cEDMc"].append(EDMc[cCtrd[0], cCtrd[1], cCtrd[2]])
        
        # Associated nuclei
        cData["nLabel"].append(np.max(nLabels[cLabels == cProps.label]))
    
    cData = pd.DataFrame(cData)
    
    # Format outputs
    outputs["nData"] = nData
    outputs["cData"] = cData
    
    return outputs

#%% Functions (save) ----------------------------------------------------------

def save(path, outputs):  
    save_path = path.parent / (path.stem)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Dataframes
    outputs["nData"].to_csv(
        save_path / f"{path.stem}_nData.csv", index=False, float_format='%.3f')
    outputs["cData"].to_csv(
        save_path / f"{path.stem}_cData.csv", index=False, float_format='%.3f')
    
    # Images
    io.imsave(
        save_path / f"{path.stem}_stack.tif",
        outputs["stack"], check_contrast=False,
        )
    io.imsave(
        save_path / f"{path.stem}_probs.tif",
        outputs["probs"], check_contrast=False,
        )
    io.imsave(
        save_path / f"{path.stem}_nLabels.tif",
        outputs["nLabels"], check_contrast=False,
        )
    io.imsave(
        save_path / f"{path.stem}_cLabels.tif",
        outputs["cLabels"], check_contrast=False,
        )
    io.imsave(
        save_path / f"{path.stem}_tophat.tif",
        outputs["tophat"], check_contrast=False,
        )
