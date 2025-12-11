# ER-Structure-Analysis

This repository contains a lightweight pipeline to import ND2 movies, segment the endoplasmic reticulum (ER) based on luminal or membrane fluorescent markers, and export quantitative measurements for the manuscript:

 **Endoplasmic reticulum disruption stimulates nuclear membrane mechanotransduction.**
<BR>**DOI: https://www.nature.com/articles/s41556-025-01820-9**


## 📌 Overview

This program processes **3D fluorescence microscopy** datasets of the ER and produces:

- Consistent ER segmentation masks (per frame)
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

 ### ER Segmentation Workflow ###
![ER Analysis Workflow](Pipeline%20Image/ER%20Anlaysis%20Pipeline.png)


 ### Sample Movies of ER Segmentation Masks Overlay
![Visualization of ER Segmentation](Pipeline%20Image/Screen%20Recording%202025-08-11%20at%209.42.52%20AM.gif)

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

Use **micromamba** (recommended) or **conda**. We have provided an environmenmt.yaml file that contains all the packages (with specific versions) for the analysis pipeline.

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

## System Requirement

The scripts were tested with Python 3.11 on Mac OS Sequoia Version 15.4.1

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
- **CSV (metrics)**: per‑frame metrics (e.g., normalized ER area, perimeter, and circularity per FOV over time)
- **Quick plot (**``**)**: summary plot generated from the CSV metrics

---

## 🧪 Reproducibility

- The **.hdf5** exported by `Image_Import.py` is the canonical intermediate consumed by the analysis program.
- Keep acquisition metadata (pixel size, z‑step, channels, exposure) with each dataset.
- Pin versions in `environment.yaml` (consider `conda env export --from-history`).

---

## Demo and Walkthrough

- ### We have provided the demo data and screenshots of the key steps to demonstrate the analysis workflow.
  - The demo data is stored in the **Demo** directory. It is cropped from the original timelapse movies and only contains 5 timepoints.

- ### Below are the walkthrough of this analysis program.
   
   - **Step 1: Run ER_Analysis_Program.py**
  ![Step1](Demo%20Step/Step1.png)
    
   - **Step 2 – Choose segmentation method. Select the appropriate segmentation method based on how the ER is labeled (e.g., EGFP-KDEL vs. Sec61B), the microscope used to acquire the images, and whether the user wants to analyze the full movie or only a subset of frames.**
  ![Step 2](Demo%20Step/Step2.png)

   - **Step 3: Program is running**
  ![Step 3](Demo%20Step/Step3.png)

   - **Step 4: Use Napari viewer to visualize all intermediate steps and generate quantitative plots before saving in the directory**
  ![Step 4](Demo%20Step/Step4.png)


## ❓ Troubleshooting

- If you have any questions, feel free to email joeshenz123@gmail.com




