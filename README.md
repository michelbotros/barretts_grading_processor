# Dysplasia segmentation of Barrett's Esophagus on H&E Whole Slide Images

In this project, we developed a 2-stage AI system for histopathological assessment of BE-related dysplasia using deep learning to enhance the efficiency and accuracy of the pathology workflow. The AI system was developed and trained on 290 whole-slide images (WSIs) that were annotated at glandular and tissue levels. The system was designed to identify individual glands, grade dysplasia, and assign a WSI-level diagnosis 

This repository contains all the code and model weights required to run the first stage of the AI system that performs dysplasia segmentation.

## Project details:
* **Name:** Dysplasia segmentation of Barrett's Esophagus on H&E Whole Slide Images
* **Model on grand-challenge:** [Grand-challenge algorithm](https://grand-challenge.org/algorithms/grading-of-dysplasia-in-barretts-esophagus/)
* **Responsible:** [Michel Botros](https://qurai.amsterdam/researcher/michel-botros/)
* **Publication:** [Deep Learning for Histopathological Assessment of Esophageal Adenocarcinoma Precursor Lesions](https://www.modernpathology.org/article/S0893-3952(24)00111-X/fulltext#%20)

### Contents
It contains the following:
* A set of example bash scripts (.sh) to test the algorithm locally and export it as a container for upload.
* An inference script (`inference.py`) that:
  * (1) reads and input H&E WSI (expects input images of 0.25mpp from Philips IntelliSite Ultrafast scanner)
  * (2) detects regions-of-interest in the WSI
  * (3) segments dysplasia in regions-of-interest
  * (4) writes the output to a heatmap tiff file (0-255) to display dysplastic regions
* Script for loading the models (`load_models.py`)
* The model weights (`resources/`)

### I/O
Uses the [whole-slide-data](https://github.com/DIAGNijmegen/pathology-whole-slide-data) package for reading and writing.
- reads from the `/input/images/he-staining`
- outputs to `/output/images/barretts-esophagus-dysplasia-heatmap`
  
---
### To-do:
* (1) smoothen output: decrease the step size but requires speed up (see 3) 
* (2) fix roster like output structure
  * artefact from writing? converting to 0-255 range with dtype UInt8?
  * check if possible to replicate w numpy and matplotlib
* (3) Optimize speed
  * optimize extract_segmentation(): make better gpu memory
  * better ROI detection or select less ROIs
    