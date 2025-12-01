# BSG_classifier_builder
Train locally fine-tuned bird sound recognition models.

This repository contains the codes for training the bird sound classification models described in "Bird Sounds Global - model builder: An end-to-end workflow for building locally fine-tuned bird classifiers" and example scripts for analyzing new data with the models.

Folder Train BSG models contains codes for preprocessing the training data, training the classification models, and evaluating the trained classifiers. The training data and some large files required for running the codes are available at Zenodo: 
10.5281/zenodo.17734985

Following large files are here either missing or truncated here and should be obtained from Zenodo: 
irmatrix/irmatrix.mat
BirdNet_results/all_birdnet_results.csv

Folder Run BSG models contains the trained classifiers and codes for running them on new data.
