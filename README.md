# ER-Structure-Analysis

This repository contains a lightweight pipeline to import ND2 movies, segment the endoplasmic reticulum (ER) based on luminal or membrane fluorescent markers, and export quantitative measurements for the manuscript:

> **“Endoplasmic reticulum disruption stimulates nuclear membrane mechanotransduction.”**

---

## 📌 Overview

This program processes **3D fluorescence microscopy** datasets of the ER and produces:

- Consistent ER segmentation masks (per frame / per z‑slice)
- Morphology metrics (e.g., **ER perimeter, area, and circularity per FOV**) over time and within regions of interest
- Optional segmentation‑overlay images/movies embedded in Napari Viewer to visualize segmentation quality

## Data flow

1. `Image_Import.py` → converts raw **.nd2** movies into a single **.hdf5** files that store the image stack as a NumPy array.
2. `ER_Analysis_Program.py` → loads the `.hdf5`, runs segmentation/quantification (via `ER_Analysis_Module.py`), and writes:
   - Segmentation results (**.hdf5**)
   - Measurements (**.csv**)
   - A quick plot summarizing **ER perimeter, area, and circularity per FOV over time** (**.png**)
3. `Segmentation_result_visualization_ER.py` *(optional)* → generates segmentation‑mask and raw images overlay in Napari Viewer for the users to assess segmentation accuracy

---

## Visualization of Analysis Workflow and Segmentation Result

![ER Analysis Workflow](/Pipeline%20Image/ER%20Anlaysis%20Pipeline.png)

![Visualization of ER Segmentation](/Pipeline%20Image/Screen%20Recording%202025-08-11%20at%209.42.52%20AM.gif)

## 📂 Repository layout

| File / Directory                          | Purpose                                                                                                                                            |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `environment.yaml`                        | Specify all dependencies for setting up the Conda/Mamba environment.                                                                               |
| `Image_Import.py`                         | Import raw `.nd2` movies and export them as a consolidated `.hdf5` file containing NumPy arrays.                                                   |
| `ER_Analysis_Module.py`                   | Provide reusable functions for data I/O, preprocessing, segmentation, and feature extraction.                                                      |
| `ER_Analysis_Program.py`                  | Load `.hdf5` data, run segmentation and metric calculations using the analysis module, and save the results.                                       |
| `Segmentation_result_visualization_ER.py` | Present segmentation alongside intermediate results and raw data in the Napari viewer, enabling users to compare and assess segmentation accuracy. |
| `aicssegmentation/` *(optional)*          | Contain accessory functions that support the main program.                                                                                         |


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



