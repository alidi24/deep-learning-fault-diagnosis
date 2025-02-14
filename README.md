# Wind Turbine Drivetrain Fault Diagnosis

A deep learning fault classification model for wind turbine drivetrain bearings using a combined PCA-CNN approach.

## Overview

Implementation of the fault detection method proposed in:

"Fault detection of offshore wind turbine drivetrains in different environmental conditions through optimal selection of vibration measurements" by Dibaj et al. (2023)
DOI: [10.1016/j.renene.2022.12.049](https://doi.org/10.1016/j.renene.2022.12.049)

The model uses combined Principal Component Analysis (PCA) and Convolutional Neural Network (CNN) (Figure 1) to detect bearing faults - main, intermediate-stage planet, and high-speed-stage bearings, (Figure 2) - in a 5-MW NREL reference drivetrain high-fidelity model.

## Dataset

The vibration data used in this project is available at:  
DOI: [10.5281/zenodo.7674842](https://doi.org/10.5281/zenodo.7674842)

## Features

- Vibration-based fault detection using optimal sensor placement
- Analysis under three environmental conditions
- Time series signal processing
- PCA for dimension reduction and feature extraction
- CNN-based fault classification
- Multi-class fault diagnosis
  

## Getting Started

[Instructions for setup and usage to be added]

## Licence

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
