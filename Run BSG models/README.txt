All scripts that require interaction from the user are available in both Python files (.py) and Jupyter notebooks (.ipynb). Additionally the repository contains some .py files which are used in the pipeline, but do not need to be modified by the user.

######################################
# Prerequisites 
######################################

Running the model requires some python libraries, which can be installed with:
pip install tensorflow==2.14.0
pip install pandas
pip install librosa
pip install numpy==1.26.0
pip install resampy

Additionally convolutional base of BirdNET-Analyzer v2.4 is needed and can be obtained from: 
https://zenodo.org/records/15050749
The model file (BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite) should be placed under folder BN-2.4_tflite.

######################################
# Running the model                    
######################################

The model can be run using either 'Run BSG models on new data.py' Python code or 'Run BSG models on new data.ipynb' Jupyter notebook.

To run the analysis, please define in the scripts the threshold for model detections and the path to the model file. Only predictions with confidence score above the threshold will be saved in results. For example:
  
threshold = 0.3 
model_folder='models/Finland/model_v4/'
model_name='BSG_birds_Finland_v4_4.keras' 






