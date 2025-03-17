#%% Imports -------------------------------------------------------------------

import os
import nd2
import time
import pickle
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# bdtools
from bdtools.nan import nan_filt
from bdtools.norm import norm_pct
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
from bdtools.models.unet import UNet

# Skimage
from skimage.filters import gaussian
from skimage.transform import rescale, resize
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border, watershed
from skimage.morphology import (
    disk, remove_small_objects, h_maxima, white_tophat
    )

# Scipy 
from scipy.ndimage import distance_transform_edt

# Napari
import napari
from napari.layers.labels.labels import Labels

# Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QPushButton, QRadioButton, QLineEdit,
    QGroupBox, QVBoxLayout, QHBoxLayout, 
    QWidget, QLabel
    )

#%% Inputs --------------------------------------------------------------------

# Procedure
overwrite = {
    "predict"  : 0,
    "nProcess" : 0,
    "cProcess" : 0,
    "analyse"  : 0,
    }

# Parameters
df = 3
nProcess_params = {
    "sigma" : 2, 
    "thresh" : 0.2, 
    "h" : 0.15,
    "min_size" : 2048,
    "clear_nBorder" : True,
    }
cProcess_params = {
    "radius" : 4, 
    "sigma" : 4,
    "thresh" : 0.25,
    "min_size" : 64,
    }

#%% Initialize ----------------------------------------------------------------

data_path = Path("D:/local_Meschichi/data")
# data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Meschichi\data")
model_name = "model_512_edt_5000-1700_2_34"
paths = list(data_path.glob("*.nd2"))

#%% Function(s) ---------------------------------------------------------------

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

def iso_downscale(stk, metadata, df=1, order=0):
    zy_ratio = metadata["vZ"] / metadata["vY"]
    stk = rescale(stk, (1, 1/df, 1/df), order=order, preserve_range=True) 
    stk = rescale(stk, (zy_ratio/df, 1, 1), order=order, preserve_range=True)
    return stk

def tophat(stk, radius):
    tph = []
    for slc in stk:
        tph.append(white_tophat(slc, footprint=disk(radius)))
    return np.stack(tph)

#%% Function : predict() ------------------------------------------------------

def predict(path, df=1):

    print(f"predict() - {path.stem} : ", end="", flush=True)
    t0 = time.time()
    
    # Load data
    stk, metadata = open_stack(path)
    
    # Predict
    unet = UNet(load_name=model_name)
    prd = unet.predict(stk, verbose=0) 
    
    # Downscale
    stk = iso_downscale(stk, metadata, df=df, order=1)
    prd = iso_downscale(prd, metadata, df=df, order=1)
    
    # Save
    dir_path = data_path / path.stem
    io.imsave(
        dir_path / (path.stem + f"_df{df}_stack.tif"), 
        stk.astype("uint16"), check_contrast=False
        )
    io.imsave(
        dir_path / (path.stem + f"_df{df}_predictions.tif"), 
        prd.astype("float32"), check_contrast=False
        )
    with open(str(dir_path / (path.stem + "_metadata.pkl")), 'wb') as file:
        pickle.dump(metadata, file)
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
#%% Function : nProcess() -----------------------------------------------------

def nProcess(
        path, 
        df=1, 
        sigma=2, 
        thresh=0.2, 
        h=0.15,
        min_size=2048,
        clear_nBorder=True,
        ):

    # print(f"nProcess() - {path.stem} : ", end="", flush=True)
    # t0 = time.time()

    # Load data
    dir_path = data_path / path.stem
    prd = io.imread(dir_path / (path.stem + f"_df{df}_predictions.tif"))
    
    # Initialize
    sigma //= df
    min_size //= df  
    
    # Segment (watershed)
    prd = gaussian(prd, sigma=sigma)
    nMask = prd > thresh
    nMask = remove_small_objects(nMask, min_size=min_size)
    if clear_nBorder:
        nMask = clear_border(nMask)
    nMarkers = h_maxima(prd, h)
    nMarkers[nMask == 0] = 0
    nMarkers = label(nMarkers)
    nLabels = watershed(-prd, nMarkers, mask=nMask)
        
    # Save
    io.imsave(
        dir_path / (path.stem + f"_df{df}_nLabels.tif"), 
        nLabels.astype("uint16"), check_contrast=False
        )
    
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")

#%% Function : cProcess() -----------------------------------------------------

def cProcess(
        path, 
        df=1, 
        radius=4, 
        sigma=4,
        thresh=0.25,
        min_size=64,
        ):
        
    # print(f"cProcess() - {path.stem} : ", end="", flush=True)
    # t0 = time.time()
    
    # Load data
    dir_path = data_path / path.stem
    stk = io.imread(dir_path / (path.stem + f"_df{df}_stack.tif"))
    nLabels = io.imread(dir_path / (path.stem + f"_df{df}_nLabels.tif"))
    
    # Initialize
    radius //= df
    sigma //= df  
    min_size //= df 
    nMask = nLabels > 0
        
    # Segment (tophat)
    tph = tophat(stk, radius)
    tph = gaussian(tph, sigma=sigma)
    med = nan_filt(stk, mask=nMask, kernel_size=7, iterations=1)       
    tph = tph / med
    tph = norm_pct(tph, mask=nMask)
    cMask = tph > thresh
    cMask = remove_small_objects(cMask, min_size=min_size)
    cLabels = label(cMask)
        
    # Save
    io.imsave(
        dir_path / (path.stem + f"_df{df}_cLabels.tif"), 
        cLabels.astype("uint16"), check_contrast=False
        )
    
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
    
#%% Function : analyse() ------------------------------------------------------

def analyse(path, df=1):
    
    # print(f"analyse()  - {path.stem} : ", end="", flush=True)
    # t0 = time.time()
    
    # Load data
    dir_path = data_path / path.stem
    stk = io.imread(dir_path / (path.stem + f"_df{df}_stack.tif"))
    nLabels = io.imread(dir_path / (path.stem + f"_df{df}_nLabels.tif"))
    cLabels = io.imread(dir_path / (path.stem + f"_df{df}_cLabels.tif"))
    with open(str(dir_path / (path.stem + "_metadata.pkl")), 'rb') as file:
        metadata = pickle.load(file)
        
    # Initialize
    voxSize = metadata["vY"] * df
    typ, sample, position = tuple(path.stem.split("_"))
        
    # nData -------------------------------------------------------------------
    
    nData = {
        
        # Info
        "type"     : [],
        "sample"   : [],
        "position" : [],
        
        # Nuclei
        "nLabel"   : [],
        "nVolume"  : [],
        "nCtrd"    : [],
        "nMajor"   : [],
        "nMinor"   : [],
        "nMMRatio" : [],
        
        # Chromocenters
        "n_cLabels"  : [],
        "n_cNumber"  : [],
        "n_cVolume"  : [],
        "n_cnRatio"  : [],
        
        }   
    
    for nProps in regionprops(nLabels):
        
        # Info
        nData["type"].append(typ)
        nData["sample"].append(sample)
        nData["position"].append(position)
        
        # Nuclei
        nData["nLabel"].append(nProps.label)
        nVolume = nProps.area * (voxSize ** 3)
        nData["nVolume"].append(nVolume)
        nCtrd = nProps.centroid
        nCtrd = [int(ctrd) for ctrd in nCtrd]
        nData["nCtrd"].append(nCtrd)
        try:
            nMajor = nProps.axis_major_length * voxSize
            nMinor = nProps.axis_minor_length * voxSize
            nMMRatio = nMinor / nMajor
            nData["nMajor"].append(nMajor)
            nData["nMinor"].append(nMinor)
            nData["nMMRatio"].append(nMMRatio)
        except ValueError:
            nData["nMajor"].append(np.nan)
            nData["nMinor"].append(np.nan)
            nData["nMMRatio"].append(np.nan)

        # Chromocenters
        unique = np.unique(cLabels[nLabels == nProps.label])
        unique = unique[unique != 0]
        nData["n_cLabels"].append(unique)
        nData["n_cNumber"].append(unique.shape[0])
        cMask = cLabels > 0
        cVolume = np.sum(cMask[nLabels == nProps.label]) * (voxSize ** 3)
        nData["n_cVolume"].append(cVolume)
        nData["n_cnRatio"].append(cVolume / nVolume)
        
    # EDMs --------------------------------------------------------------------
        
    # Compute EDM (2nd rescaling)
    nCtrd = np.stack([ctrd for ctrd in nData["nCtrd"]])
    nCtrd = ((
        (nCtrd[:, 0]).astype(int), 
        (nCtrd[:, 1]).astype(int), 
        (nCtrd[:, 2]).astype(int),
        ))
    EDMb = distance_transform_edt(nLabels > 0)
    EDMc = np.zeros_like(EDMb, dtype=bool)
    EDMc[nCtrd] = True
    EDMc = distance_transform_edt(np.invert(EDMc))
    EDMc[EDMb == 0] = 0
    EDMb = resize(EDMb, nLabels.shape, order=0)
    EDMc = resize(EDMc, nLabels.shape, order=0)
        
    # cData -------------------------------------------------------------------

    cData = {
        
        # Info
        "type"     : [],
        "sample"   : [],
        "position" : [],
        
        # Chromocenters
        "cLabel"   : [],
        "cVolume"  : [],
        "cCtrd"    : [],
        "cMajor"   : [],
        "cMinor"   : [],
        "cMMRatio" : [],
        "cInt"     : [],
        "cEDMb"    : [],
        "cEDMc"    : [], 
        
        # Nuclei
        "c_nLabel"   : [],
        
        }
    
    for cProps in regionprops(cLabels, intensity_image=stk):
        
        # Info
        cData["type"].append(typ)
        cData["sample"].append(sample)
        cData["position"].append(position)
        
        # Chromocenters
        cData["cLabel"].append(cProps.label)
        cVolume = cProps.area * (voxSize ** 3)
        cData["cVolume"].append(cVolume)
        cCtrd = cProps.centroid
        cCtrd = [int(ctrd) for ctrd in cCtrd]
        cData["cCtrd"].append(cCtrd)
        try:
            cMajor = cProps.axis_major_length * voxSize
            cMinor = cProps.axis_minor_length * voxSize
            cMMRatio = cMinor / cMajor
            cData["cMajor"].append(cMajor)
            cData["cMinor"].append(cMinor)
            cData["cMMRatio"].append(cMMRatio)
        except ValueError:
            cData["cMajor"].append(np.nan)
            cData["cMinor"].append(np.nan)
            cData["cMMRatio"].append(np.nan)
        cData["cInt"].append(cProps.intensity_mean) # To be discussed
        cEDMb = EDMb[cCtrd[0], cCtrd[1], cCtrd[2]] * voxSize
        cEDMc = EDMc[cCtrd[0], cCtrd[1], cCtrd[2]] * voxSize
        cData["cEDMb"].append(cEDMb)
        cData["cEDMc"].append(cEDMc)
        
        # Associated nuclei
        cData["c_nLabel"].append(np.max(nLabels[cLabels == cProps.label]))
    
    # Save --------------------------------------------------------------------

    # # Correct centroids
    # for i, nCtrd in enumerate(nData["nCtrd"]):
    #     nData["nCtrd"][i] = [int(c * df) for c in nCtrd]
    # for i, cCtrd in enumerate(cData["cCtrd"]):
    #     cData["cCtrd"][i] = [int(c * df) for c in cCtrd]
    
    # Dataframes
    pd.DataFrame(nData).to_csv(
        dir_path / f"{path.stem}_df{df}_nData.csv", index=False, float_format='%.3f')
    pd.DataFrame(cData).to_csv(
        dir_path / f"{path.stem}_df{df}_cData.csv", index=False, float_format='%.3f')

    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")

#%% Function : main() ---------------------------------------------------------

def main(paths, df=df, overwrite=overwrite):
    
    for path in paths:
        
        # Create dir
        dir_path = data_path / path.stem
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
        
        # Predict
        prd_path = dir_path / (path.stem + f"_df{df}_predictions.tif")
        if not prd_path.exists() or overwrite["predict"]:
            predict(path, df=df)
            
    # nProcess()
    tmp_paths = []
    for path in paths:
        dir_path = data_path / path.stem
        prd_path = dir_path / (path.stem + f"_df{df}_predictions.tif")
        nlbl_path = dir_path / (path.stem + f"_df{df}_nLabels.tif")
        if prd_path.exists() and \
            (not nlbl_path.exists() or overwrite["nProcess"]):
            tmp_paths.append(path)
    print(f"nProcess() ({len(tmp_paths)} stacks) : ", end="", flush=True)
    t0 = time.time()
    Parallel(n_jobs=-1)(
        delayed(nProcess)(path, df=df, **nProcess_params)
        for path in tmp_paths
        )
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # cProcess()
    tmp_paths = []
    for path in paths:
        dir_path = data_path / path.stem
        nlbl_path = dir_path / (path.stem + f"_df{df}_nLabels.tif")
        clbl_path = dir_path / (path.stem + f"_df{df}_cLabels.tif")
        if nlbl_path.exists() and\
            (not clbl_path.exists() or overwrite["cProcess"]):
            tmp_paths.append(path)
    print(f"cProcess() ({len(tmp_paths)} stacks) : ", end="", flush=True)
    t0 = time.time()
    Parallel(n_jobs=-1)(
        delayed(cProcess)(path, df=df, **cProcess_params)
        for path in tmp_paths
        )
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # analyse()
    tmp_paths = []
    for path in paths:
        dir_path = data_path / path.stem
        clbl_path = dir_path / (path.stem + f"_df{df}_cLabels.tif")
        nData_path = dir_path / f"{path.stem}_df{df}_nData.csv"
        if clbl_path.exists() and\
            (not nData_path.exists() or overwrite["analyse"]):
            tmp_paths.append(path)
    print(f"analyse()  ({len(tmp_paths)} stacks) : ", end="", flush=True)
    t0 = time.time()
    Parallel(n_jobs=-1)(
        delayed(analyse)(path, df=df)
        for path in tmp_paths
        )
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")

#%% Function : results() ------------------------------------------------------

def results(paths, df=df):

    # Load data    
    nData_paths = list(data_path.rglob("*nData.csv"))
    cData_paths = list(data_path.rglob("*cData.csv"))
    
    # Merge data
    nData = []
    for path in nData_paths:
        nData.append(pd.read_csv(path))
    nData = pd.concat(nData, ignore_index=True)
    
    cData = []
    for path in cData_paths:
        cData.append(pd.read_csv(path))
    cData = pd.concat(cData, ignore_index=True)
    
    # Save
    pd.DataFrame(nData).to_csv(
        data_path / f"_df{df}_nData_merged.csv", index=False, float_format='%.3f')
    pd.DataFrame(cData).to_csv(
        data_path / f"_df{df}_cData_merged.csv", index=False, float_format='%.3f')

#%% Function : plot() ---------------------------------------------------------

def plot(df=df):
    
    # Load data 
    nData = pd.read_csv(data_path / f"_df{df}_nData_merged.csv")
    cData = pd.read_csv(data_path / f"_df{df}_cData_merged.csv")
        
    # Initialize
    measures = ["nVolume", "nMMRatio"]
    nM = len(measures)
    types = np.unique(nData["type"])
    nT = len(types)
    
    # Plot
    fig, axes = plt.subplots(nM, 1, figsize=(8, 3 * nM))
    
    for m, measure in enumerate(measures):

        for t, typ in enumerate(types):
            
            # data
            data = np.array(nData.loc[nData["type"] == typ, [measure]])
            avg = np.nanmean(data)
            std = np.nanstd(data)
            sem = std  / np.sqrt(len(data))
            n = len(data)
            
            # bars
            axes[m].bar(
                t, avg, 
                yerr=sem, capsize=5,
                color="lightgray", alpha=1, label=typ,
                )
            
            # text
            axes[m].text(
                (t + 0.5) / nT, 0.05, f"{n}", size=10, color="k",
                transform=axes[m].transAxes, ha="center", va="center",
                )
            
            # titles & axis
            axes[m].set_title(measure)
            axes[m].set_xlim(-0.5, nT - 0.5)
            axes[m].set_xticks(np.arange(nT))
            axes[m].set_xticklabels(types, rotation=90)
            
    plt.tight_layout()
    plt.savefig(data_path / "_nData_graphs.png", format="png")

#%% Class : display() ---------------------------------------------------------

class Display:
    
    def __init__(self, paths):
        self.paths = paths
        self.idx = 0
        self.show = True
        self.init_data()
        self.init_viewer()

    def init_data(self):
        
        # Load
        path = self.paths[self.idx]
        dir_path = data_path / path.stem
        self.stk = io.imread(dir_path / (path.stem + f"_df{df}_stack.tif"))
        self.prd = io.imread(dir_path / (path.stem + f"_df{df}_predictions.tif"))
        self.nLabels = io.imread(dir_path / (path.stem + f"_df{df}_nLabels.tif"))
        self.cLabels = io.imread(dir_path / (path.stem + f"_df{df}_cLabels.tif"))
        
        # Format
        self.nMask = (self.nLabels > 0).astype(int)
        self.cMask = (self.cLabels > 0).astype(int)
        self.cMask *= 3
        
    def init_viewer(self):
        
        # Create viewer
        self.viewer = napari.Viewer()
        self.viewer.add_image(
            self.stk, name="stack", visible=True,
            contrast_limits=[
                np.percentile(self.stk,  0.01),
                np.percentile(self.stk, 99.99) * 2,
                ],
            gamma=0.75,
            )
        self.viewer.add_image(
            self.prd, name="predictions", visible=False, 
            contrast_limits=[0, 2],
            colormap="magma",
            blending="additive",
            )
        self.viewer.add_labels(
            self.nMask, name="nuclei", visible=True,
            blending="translucent_no_depth",
            )
        self.viewer.add_labels(
            self.cMask, name="chromocenters", visible=True,
            blending="opaque",            
            ) 
        
        # 3D display
        self.viewer.dims.ndisplay = 3
        self.viewer.layers["nuclei"].iso_gradient_mode = "smooth"
        self.viewer.layers["chromocenters"].iso_gradient_mode = "smooth"
        
        # Create "select stack" menu
        self.stk_group_box = QGroupBox("Select stack")
        stk_group_layout = QVBoxLayout()
        self.btn_next_image = QPushButton("Next Image")
        self.btn_prev_image = QPushButton("Previous Image")
        self.dia_cytotype = QLineEdit()
        self.dia_cytotype.setPlaceholderText("specify cytotype")
        stk_group_layout.addWidget(self.btn_next_image)
        stk_group_layout.addWidget(self.btn_prev_image)
        stk_group_layout.addWidget(self.dia_cytotype)
        self.stk_group_box.setLayout(stk_group_layout)
        self.btn_next_image.clicked.connect(self.next_stack)
        self.btn_prev_image.clicked.connect(self.prev_stack)
        self.dia_cytotype.textChanged.connect(self.select_cytotype)
        
        # Create "display" menu
        self.dsp_group_box = QGroupBox("Display")
        dsp_group_layout = QHBoxLayout()
        self.rad_mask = QRadioButton("mask")
        self.rad_labels = QRadioButton("labels")
        self.rad_predictions = QRadioButton("predictions")
        self.rad_mask.setChecked(True)
        dsp_group_layout.addWidget(self.rad_mask)
        dsp_group_layout.addWidget(self.rad_labels)
        dsp_group_layout.addWidget(self.rad_predictions)
        self.dsp_group_box.setLayout(dsp_group_layout)
        self.rad_mask.toggled.connect(
            lambda checked: self.show_masks() if checked else None)
        self.rad_labels.toggled.connect(
            lambda checked: self.show_labels() if checked else None)
        self.rad_predictions.toggled.connect(
            lambda checked: self.show_predictions() if checked else None)
        

        # Create texts
        self.info_image = QLabel()
        self.info_image.setFont(QFont("Consolas"))
        self.info_image.setText(
            f"{self.paths[self.idx].stem}"
            )
        self.info_short = QLabel()
        self.info_short.setFont(QFont("Consolas"))
        self.info_short.setText(
            "prev/next stack  : page down/up \n"
            "hide/show layers : Enter"
            )
        
        # Create layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.stk_group_box)
        self.layout.addWidget(self.dsp_group_box)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.info_image)
        self.layout.addWidget(self.info_short)

        # Create widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.viewer.window.add_dock_widget(
            self.widget, area="right", name="Painter") 
        
        # Shortcuts
        
        @self.viewer.bind_key("PageDown", overwrite=True)
        def previous_image_key(viewer):
            self.prev_stack()
        
        @self.viewer.bind_key("PageUp", overwrite=True)
        def next_image_key(viewer):
            self.next_stack()
        
        @Labels.bind_key("Enter", overwrite=True)
        def toogle_lm_key(viewer):
            if self.show:
                self.hide_layers()
            else:
                self.show_layers()
                
    def update_stack(self):
        self.viewer.layers["stack"].data = self.stk
        self.viewer.layers["predictions"].data = self.prd
        if self.rad_mask.isChecked():
            self.show_masks()
        if self.rad_labels.isChecked():
            self.show_labels()
        if self.rad_predictions.isChecked():
            self.show_predictions()
                
    def update_txt(self):
        self.info_image.setText(
            f"{self.paths[self.idx].stem}"
            )
                
    def next_stack(self):
        if self.idx < len(self.paths):
            self.idx += 1
            self.init_data()
            self.update_stack()
            self.update_txt()
            
    def prev_stack(self):
        if self.idx > 0:
            self.idx -= 1
            self.init_data()
            self.update_stack()
            self.update_txt()
                
    def select_cytotype(self):
        cytotype = self.dia_cytotype.text()
        if len(cytotype) == 3:
            for i, path in enumerate(self.paths):
                if path.stem[:3] == cytotype:
                    self.idx = i
                    self.init_data()
                    self.update_stack()
                    self.update_txt()
                    break
                
    def show_masks(self):
        self.viewer.layers["nuclei"].visible = True
        self.viewer.layers["chromocenters"].visible = True
        self.viewer.layers["predictions"].visible = False
        self.viewer.layers["nuclei"].data = self.nMask
        self.viewer.layers["chromocenters"].data = self.cMask
        
    def show_labels(self):
        self.viewer.layers["nuclei"].visible = True
        self.viewer.layers["chromocenters"].visible = True
        self.viewer.layers["predictions"].visible = False
        self.viewer.layers["nuclei"].data = self.nLabels
        self.viewer.layers["chromocenters"].data = self.cLabels
        
    def show_predictions(self):
        self.viewer.layers["nuclei"].visible = False
        self.viewer.layers["chromocenters"].visible = False
        self.viewer.layers["predictions"].visible = True
        
    def hide_layers(self):
        self.show = False
        if self.rad_mask.isChecked() or self.rad_labels.isChecked():
            self.viewer.layers["nuclei"].visible = False
            self.viewer.layers["chromocenters"].visible = False
        if self.rad_predictions.isChecked():
            self.viewer.layers["predictions"].visible = False
        
    def show_layers(self):
        self.show = True
        if self.rad_mask.isChecked() or self.rad_labels.isChecked():
            self.viewer.layers["nuclei"].visible = True
            self.viewer.layers["chromocenters"].visible = True
        if self.rad_predictions.isChecked():
            self.viewer.layers["predictions"].visible = True
        
#%% Execute -------------------------------------------------------------------

# if __name__ == "__main__":
#     main(paths)
#     results(paths)
#     plot()
#     Display(paths)
    
#%% Tests ---------------------------------------------------------------------

def detect_stable_slices(
        arr, 
        winsize=10, 
        crr_thresh=0.98, 
        grd_thresh=0.002, 
        plot=False
        ):
    
    global crr_valid, grd_valid, valid
    
    def image_correlation(arr0, arr1):
        arr0_flat = norm_pct(arr0).ravel()
        arr1_flat = norm_pct(arr1).ravel()
        return np.corrcoef(arr0_flat, arr1_flat)[0, 1]
    
    # Image correlation
    crr = []
    for z in range(wsize, stk.shape[0]):
        crr.append(image_correlation(stk[z - wsize, ...], stk[z, ...]))
    crr = np.stack(crr)
    grd = np.gradient(crr)
    
    # Detect valid idxs
    crr_valid = np.where(crr > crr_thresh)[0]
    # grd_valid = np.where((grd >= -grd_thresh) & (grd <= grd_thresh))[0]
    # valid = np.intersect1d(crr_valid, grd_valid)
    
    if plot:
        
        fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)    
        
        axes[0].plot(crr, color="gray", linewidth=3)
        axes[0].axhline(y=crr_thresh, color="k", linestyle=":", linewidth=1)
        axes[1].plot(grd, color="gray", linewidth=3)
        axes[1].axhline(y=grd_thresh, color="k", linestyle=":", linewidth=1)
        axes[1].axhline(y=-grd_thresh, color="k", linestyle=":", linewidth=1)
        
        for idx in crr_valid:
            axes[0].axvspan(
                idx - 0.5, idx + 0.5, 
                ymin=0, ymax=1, facecolor="gray", alpha=0.1
                )
        for idx in grd_valid:
            axes[1].axvspan(
                idx - 0.5, idx + 0.5, 
                ymin=0, ymax=1, facecolor="gray", alpha=0.1
                )
        for idx in valid:
            axes[0].axvspan(
                idx - 0.5, idx + 0.5, 
                ymin=0, ymax=0.05, facecolor="red", alpha=0.5
                )
            axes[1].axvspan(
                idx - 0.5, idx + 0.5, 
                ymin=0, ymax=0.05, facecolor="red", alpha=0.5
                )

        # Axes
        y_low, y_high, y_step = 0.5, 1.05, 0.05
        axes[0].set_xlim(0, len(crr))
        axes[0].set_ylim(y_low, y_high)
        axes[0].set_yticks(np.arange(y_low, y_high, y_step))
        axes[0].grid(True, alpha=0.5)
        
        y_low, y_high, y_step = -0.05, 0.05, 0.01
        axes[1].set_xlim(0, len(crr))
        axes[1].set_ylim(y_low, y_high)
        axes[1].set_yticks(np.arange(y_low, y_high, y_step))
        axes[1].grid(True, alpha=0.5)

        plt.tight_layout()
        plt.show()
        
# Inputs
name = "SBG_01_04"
wsize = 5
idxs = random_unique_ints = np.random.choice(
    len(paths), size=len(paths), replace=False)

for idx in idxs:
                
    # Load data
    path = paths[idx]
    dir_path = data_path / path.stem
    stk = io.imread(dir_path / (path.stem + f"_df{df}_stack.tif"))
    
    # Detect stable slices
    detect_stable_slices(stk, plot=True)
    
    # Ask user whether to continue
    user_input = input("Press Enter to continue or type 'q' to quit: ")
    if user_input.lower() == "q":
        print("Loop stopped by user.")
        break
        
    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(stk)