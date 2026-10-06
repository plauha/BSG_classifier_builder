# BSG_classifier_builder
Train locally fine-tuned bird sound recognition models.

This repository contains the codes for training the bird sound classification models described in "Bird Sounds Global - model builder: An end-to-end workflow for building locally fine-tuned bird classifiers" and example scripts for analyzing new data with the models.

### Train BSG models
Folder 'Train BSG models' contains codes for preprocessing the training data, training the classification models, and evaluating the trained classifiers. The training data and some large files required for running the codes will be published in Zenodo: (LINK WILL BE ADDED)

In addition to the training data, following large files are here either missing or truncated here and should also be obtained from Zenodo: 

- irmatrix/irmatrix.mat

- BirdNet_results/all_birdnet_results.csv

To run the model training pipeline, follow the instructions on the README file under 'Train BSG models'. For performing the full training pipeline from collecting recordings to annotating them and training the models, users must for now contact the developers of BSG to get their audio uploaded on BSG portal. 

### Run BSG models
Folder 'Run BSG models' contains the trained classifiers and codes for analyzing new audio data with the classifiers. Place the audio data to be analyzed under folder test_audio and run the classifier using either 'Run BSG models on new data.py' Python code or 'Run BSG models on new data.ipynb' Jupyter notebook. For more detailed instructions and prerequisites, see the README file under 'Run BSG models'.

All models can also be run through the desktop application available at:
https://laji.fi/theme/sirkku
