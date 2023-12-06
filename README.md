# TMA-dearray-stain-separation

This repository contains Python scripts for the analysis of Tissue Microarray (TMA) core images, specifically designed to support the work published in the publication titled: **"GASP-1 overexpression is involved in the development of BPH and progression of early-stage prostatic malignant diseases to prostate cancer."** (Manuscript not submitted yet.)

## Overview

The scripts in this repository are organized as follows:

- `TMA-dearray.py`: This script extracts individual cores from a TMA and saves them to a specified location as PNG. It takes a text file with positions as input, generated using Qupath. (https://qupath.github.io/)
  
- `color_separate_functions.py`: This module contains functions for the analysis of TMA core images. It includes functions for color separation, nuclei segmentation, and intensity analysis.

- `color_separate.py`: This script performs batch color separation and nuclei segmentation on multiple TMA core images. Results are saved in separate directories labeled inside the 'results' directory.

## Installation

The code has been tested on Python 3.9.18

Before running the scripts, ensure you have the required dependencies installed. You can install them using the following:

```bash
pip install -r requirements.txt
```
Please note that openslide installation requires additional steps.  

Ensure OpenSlide is installed by executing the following commands: (It should be already installed if you've executed the above pip command).
1. Install openslide-python:
    ```
    pip install openslide-python
    ```
2. Download the latest Windows binaries from https://openslide.org/download/.
3. Extract the contents to a location for easy reference, preferably in the openslide directory in site-packages
On our system: "C:/Users/Admin/Anaconda3/envs/Py39_base/Lib/site-packages/openslide/openslide-win64-20231011/bin"

In case you are unsure of your site packages directory, use the following script to find the location:

```
import sys
for p in sys.path:
    print(p)
```

## Results Preview

![TMA Dearray Code Result](https://github.com/bnsreenu/TMA-dearray-stain-separation/blob/main/images/TMA-dearray.jpg)

### **Color Separation (Segregation):**
![Color Separation Result](https://github.com/bnsreenu/TMA-dearray-stain-separation/blob/main/images/segregated.jpg)
