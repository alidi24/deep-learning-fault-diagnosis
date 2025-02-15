# Wind Turbine Drivetrain Fault Diagnosis

A deep learning fault classification model for wind turbine drivetrain bearings using a combined PCA-CNN approach.

## Overview

Implementation of the fault detection method proposed in:

"Fault detection of offshore wind turbine drivetrains in different environmental conditions through optimal selection of vibration measurements" by Dibaj et al. (2023)
DOI: [10.1016/j.renene.2022.12.049](https://doi.org/10.1016/j.renene.2022.12.049)


This project implements a combined Principal Component Analysis (PCA) and Convolutional Neural Network (CNN) approach (Fig. 1) to detect bearing faults in a 5-MW NREL reference drivetrain high-fidelity model. The model is capable of identifying faults in main bearings, low-speed shaft planet bearings, and high-speed shaft bearings (Fig. 2). Fig. 3 illustrates the overall procedure of the multi-measurement fault classification approach proposed in the paper. According to the paper's findings, measurements from sensors A1A and A3A yield the highest detection accuracy for the studied fault locations. Therefore, the current implementation uses time series data from these two sensors as input.


If you find this work useful, please consider citing the original paper:
@article{DIBAJ2023161,
title = {Fault detection of offshore wind turbine drivetrains in different environmental conditions through optimal selection of vibration measurements},
journal = {Renewable Energy},
volume = {203},
pages = {161-176},
year = {2023},
issn = {0960-1481},
doi = {https://doi.org/10.1016/j.renene.2022.12.049},
url = {https://www.sciencedirect.com/science/article/pii/S0960148122018341},
author = {Ali Dibaj and Zhen Gao and Amir R. Nejad},
keywords = {Offshore wind turbine, Drivetrain system, Fault detection, Optimal vibration measurements},
}


![Fig. 1. CNN model architecture](figures/cnn-arch.png)
![Fig. 2. Fault and measurement locations on drivetrain schematic layout.](figures/gearbox-schematic.png)
![Fig. 3. Overall procedure of multi-measurement fault detection approach proposed in the paper.](figures/overall-procedure.png)


## Dataset

The original vibration dataset used in this project, along with additional information, is available at: 
DOI: [10.5281/zenodo.7674842](https://doi.org/10.5281/zenodo.7674842)

The HuggingFace version of this dataset (containing the target sensors data used in this project) is available at:
[Link to be added]

## Features

- Vibration-based fault detection using optimal sensor placement
- Analysis under three environmental conditions
- Time series signal processing
- PCA for dimension reduction and feature extraction
- CNN-based fault classification
- Multi-class fault diagnosis
  

## Getting Started

[Instructions for setup and usage to be added]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




