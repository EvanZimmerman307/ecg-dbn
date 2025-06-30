# ECG-based Chagas Disease Prediction using Dynamic Bayesian Networks

This repository contains code and resources for predicting Chagas disease from electrocardiogram (ECG) data using Dynamic Bayesian Networks (DBNs). DBNs are used to model the relationship between the timing of ECG wave features and the likelihood of Chagas disease.

## Dataset

See [Detection of Chagas Disease from the ECG: The George B. Moody PhysioNet Challenge 2025](https://moody-challenge.physionet.org/2025/) for formulating your own dataset. 

## Project Structure

- **PhysionetStarterCode/**: Contains the starter kit code for the 2025 PhysioNet Challenge that this work is based on
  - Contains scripts for processing raw ecg data

- **NeurokitDelineation/**: Code for executing wave delineation (extracting wave features) from 12-lead ecg readings using the Neurokit2 framework

- **Notebooks/**: Jupyer notebooks for EDA and initial DBN experiments

- ```chagas_delineation_loader.py```: data loader script for neurokit wave delineation
- ```delineation_feature_experimentation.py ```: core classes and methods for building/experimenting with DBNs on wave delineation data
- ```delineation_interval_experiment.py```, ``` delineation_no_static.py```: DBN experiments with different wave delineation features

## Requirements

Install dependencies with:
```bash
pip install -r requirements.txt
```
## Getting Started

### Data Preprocessing

See **PhysionetStarterCode/** for how to download and process raw ecg data.

### Wave Delineation

See **NeurokitDelineation/** to extract wave delineation data from 12-lead ecg data.

### Model Fitting and Evaluation

```bash
python delineation_feature_experimentation.py
```
- Requires csv file of wave delineation data (see ```NeurokitDelineation/main.py```)
- Requires a folder with ecg data in wfdb format

## References

- Neurokit2: https://github.com/neuropsychology/NeuroKit/tree/master/neurokit2
- pgmpy Dynamic Bayesian Network: https://pgmpy.org/models/dbn.html