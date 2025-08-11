# ER-Structure-Analysis

This repository contains a lightweight pipeline to import ND2 movies, segment the endoplasmic reticulum (ER) based on luminal or membrane fluorescent markers, and export quantitative measurements and figures for the manuscript:

> **“Endoplasmic reticulum disruption stimulates nuclear membrane mechanotransduction.”**

---

## 📌 Overview

This program processes **3D fluorescence microscopy** datasets of the ER and produces:

- Consistent ER segmentation masks (per frame / per z‑slice)
- Morphology metrics (e.g., **median ER perimeter, area, and circularity per FOV**) over time and within regions of interest
- Optional segmentation‑overlay images/GIFs to visualize segmentation quality

## Data flow

1. `Image_Import.py` → converts raw **.nd2** movies into a single **.hdf5** files that store the image stack as a NumPy array.
2. `ER_Analysis_Program.py` → loads the `.hdf5`, runs segmentation/quantification (via `ER_Analysis_Module.py`), and writes:
   - Segmentation results (**.hdf5**)
   - Measurements (**.csv**)
   - A quick plot summarizing **ER perimeter, area, and circularity per FOV over time** (**.png**)
3. `Segmentation_result_visualization_ER.py` *(optional)* → generates segmentation‑mask overlays and movies for QC.

---

## Visualization of Analysis Workflow and Segmentation Result

![ER Analysis Workflow](/Pipeline%20Image/ER%20Anlaysis%20Pipeline.png)

![Visualization of ER Segmentation](/Pipeline%20Image/Screen%20Recording%202025-08-11%20at%202.23.54 AM.gif)

## 📂 Repository layout

```
.
├── environment.yaml.  #Dependency specification (Conda/Mamba).
├── Image_Import.py.   #Import raw `.nd2` movies → export a consolidated `.hdf5` (NumPy array inside HDF5).
├── ER_Analysis_Module.py #Reusable functions for I/O, preprocessing, segmentation, feature extraction.
├── ER_Analysis_Program.py #Load the `.hdf5`, run segmentation & metrics via the module, save results.
├── Segmentation_result_visualization_ER.py #Plot overlays and summarize segmentation outputs.
└── aicssegmentation/    # (optional) accessory functions that support the main parogram
```

---

## ⚙ Installation

Use **micromamba** (recommended) or **conda**.

```bash
# Micromamba
micromamba create -n er_analysis --file environment.yaml
micromamba activate er_analysis
```

```bash
# Conda
conda env create -f environment.yaml
conda activate er_analysis
```

> The environment name can be anything; match the value under `name:` in `environment.yaml`.

---

## 🚀 Quick start (GUI file selection)

1. **Open this folder** in **Visual Studio Code** or **PyCharm**.
2. **Run Image_Import.py**\
   • Select the the directory that contains raw **.nd2** movie(s).\
   • For each **.nd2** movie, the script writes a single **.hdf5** file containing the image stack (NumPy array inside HDF5).
3. **Run ER_Analysis_Program.py**\
   • Select the `.hdf5` file from step 2.\
   • The program produces:\
   – **Segmentation results** (`.hdf5`)\
   – **Measurement results** (`.csv`)\
   – **Quick plot** of ER perimeter, area, and circularity per FOV over time (`.png`)

4. *(Optional)* **Run Segmentation_result_visualization_ER.py**`` to inspect segmentation overlays and examine its accuracy.


---

## 📤 Outputs

- **HDF5 (segmentation)**: labeled ER masks and/or processed stacks
- **CSV (metrics)**: per‑frame metrics (e.g., median ER area, perimeter, and circularity per FOV over time)
- **Quick plot (**``**)**: summary plot generated from the CSV metrics

---

## 🧪 Reproducibility

- The **.hdf5** exported by `Image_Import.py` is the canonical intermediate consumed by the analysis program.
- Keep acquisition metadata (pixel size, z‑step, channels, exposure) with each dataset.
- Pin versions in `environment.yaml` (consider `conda env export --from-history`).

---

## ❓ Troubleshooting

- **ND2 import errors** → ensure the ND2 reader dependency in `environment.yaml` is installed and working.
- **Large movies** → downsample/crop for a quick test; process in tiles/batches if supported.
- Please send questions to Zhouyang Shen: joeshenz123@gmail.com



