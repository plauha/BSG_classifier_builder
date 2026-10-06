All scripts that require interaction from the user are available in both Python files (.py) and Jupyter notebooks (.ipynb). Additionally the repository contains some .py files which are used in the pipeline, but do not need to be modified by the user.

Please note, that for performing the full pipeline (Steps 1-7) from collecting recordings to annotating them and training the models, users must for now contact the developers of BSG to get their audio uploaded on BSG portal. 
Training the model (steps 2-6) can be executed with those recordings that have been annotated so far and published on Zenodo (LINK WILL BE ADDED HERE). 

######################################
# Prerequisites 
######################################

Running the model requires some python libraries, which can be installed with:
pip install tensorflow==2.14.0
pip install pandas
pip install librosa
pip install colorednoise
pip install noisereduce
python -m pip install resampy

Additionally following resources are needed:
 - convolutional base of BirdNET-Analyzer v2.4 can be obtained from: https://zenodo.org/records/15050749
The model file (BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite) should be placed under folder BirdNET-Analyzer-main.

- xeno-canto.py can be installed either by:
  pip install xeno-canto
  or by 
  git clone https://github.com/ntivirikin/xeno-canto-py
  The only file required for BSG pipeline is xenocanto.py.
  
- Original recordings for BSG templates listed in BSG_data/BSG_template_files_metadata.csv can be obtained from https://www.xeno-canto.org and https://www.macaulaylibrary.org

######################################
# Step 1
# Preparing audio data for BSG
######################################
Folder '1. Prepare audio data for BSG portal' contains scripts for selecting and preprocessing the recordings to be uploaded to BSG portal for annotation. The readme-file under '1. Prepare audio data for BSG portal' shows, how step 1 can be run on command line.   
Please note, that uploading recordings to BSG is currently possible only by contacting the developers of BSG. 

################################################################
# After Step 1: Users annotate the recordings on BSG portal: 
# https://bsg.laji.fi/
################################################################

##########################################################
# Steps 2-6
# Training the locally fine-tuned classification model
##########################################################

# 2. Collect and preprocess annotated data
This script is used to preprocess the data and labels produced on BSG portal for model training. For the training data set published on Zenodo (LINK WILL BE ADDED), this step has already been applied!

# 3. Define target species list
This script is used to define the species list for the locally fine-tuned classification model. To run the script, specify the path where species list will be saved and the site id(s) from BSG portal corresponding to the target location:
model_path = 'models/Spain_Portugal/'
site_list = [622, 718, 655, 750, 643]
The species list should be manually checked in case of redundant species caused by erroneous annotations. Additional species can also be included, if necessary, for example if a species is known to occur in the area, but is not included in the current BSG annotations.

# 4a. Get global data for xeno-canto
This script is used to download global training data set from xeno-canto.org. To run the script, specify following variables:

- Path to species list created in step 3, for example:
path_to_species_list = "models/Spain_Portugal/species_list.csv"

- Path to metadata of previously processed xeno-canto recordings (if such exist):
previous_clips = pd.read_csv("---/xc_clips.csv")

- Running batch number, to keep track, when each xeno-canto recording has been downloaded:
batch_no = "_16"

- Paths where xeno-canto recordings will be temporarily saved, where metadata will be saved, and where possible previously downloaded xeno-canto recordings have been saved:
path_to_new_recs = '---/' 
path_to_old_recs = '---/' 
path_to_metadata = '---/' 

# 4b. Find vocalizations from global data and save new training data
This script is used to extract bird calls from weakly labelled xeno-canto recordings. To run the script, specify following variables:

- Running batch number, same as in step 4a:
batch_no = "_16"

- Path to species list created in step 3, for example:
path_to_species = 'models/Spain_Portugal/species_list.csv' 

- Paths where xeno-canto recordings and their metadata are saved (same as in step 4a)
path_to_metadata = '---/' 
path_to_recs = '---/'

- If some of the species in the target species list are not recognized by BirdNET, user must manually input the timepoints (in seconds) around which 3 sec frames are extracted from xeno-canto recording by running the code after "Manual processing for species BirdNET doesn't recognize". If such manual process is carried out, set variable include_manual to True:
include_manual = True

# 4c. Create train-test split
This script is used to prepare the training data for model training and create a train-test-split. To run the script, specify following variables:

- Path to model, same as in step 3:
path_to_model = 'models/Spain_Portugal/

- Path where BSG templates and their metadata have been saved (in step 2). Original recordings for BSG templates listed in BSG_data/BSG_template_files_metadata.csv can be obtained from https://www.xeno-canto.org and https://www.macaulaylibrary.org
path_to_templates = "---/" 
path_to_template_metadata = "---/bsg_templates.csv"

- Path to the metadata of xeno-canto recordings (same as in step 4a):
path_to_xc_metadata = "---/xc_clips.csv" 

- If external noise metadata, such as recordings from ESC (https://doi.org/10.1145/2733373.2806390) or human speech from Common Voice (arXiv preprint:1912.06670) are used, specify path to the metadata. Metadata format should be the following:
file_name,original_class,species_code
5-263501-A-25.wav,footsteps,nobird
3-118487-A-26.wav,laughing,human
2-133863-A-11.wav,sea_waves,nobird
...
path_to_noise_metadata = "---/noise_clips.csv" 

- Path to clipped training data from xeno-canto:
path_to_xc_audio = '---/' 

- Path to external noise (see above), if such data is used:
path_to_noise_audio = '---/' 

- Under 'Global data', set the proportion of training data se tto be used for validation to confirm that model training does not fail. For example, use 1/20 of data for validation:
val_prop = 20 

- Under 'Local data' set the paths to BSG soundscapes and their labels (files available from Zenodo, LINK WILL BE ADDED), and the proportion of local training data to be used for final evaluation:
path_to_bsg_metadata = '---/BSG_soundscapes.csv'
path_to_bsg_labels = '---/BSG_labels.csv'
path_to_bsg_audio = '---/'
val_prop = 10

# 5. Train the model
Locally fine-tuned models are fitted with this script. To run the script, specify following variables:

- Paths to the training data and output path, where model will be saved, for example:
path_to_model = 'models/Spain_Portugal/'
path_out = 'models/Spain_Portugal/model_v1/'

- When training with global data: If training data contains a validation set, how many cpus are available when training and for how many epochs, the model should be trained. If model is trained eg. on a shared computing cluster with limited time per job, variable round can be used to first train the model for a smaller number of epochs and continue training later. For example:
with_val = True
tflite_threads = 5
epochs = 10
round=1

- When training with local data: Same as above with local data, for example:
with_val = True
tflite_threads = 5
epochs = 15
round=2

Once, the model is trained, the training loss and accuracy can be plotted by defining path to the model and the number of training rounds:
model_name = 'Spain_Portugal/'
rounds = 2

# 6. Test models on validation data
This script shows, how model performance has been evaluated.
To run the code, specify following variables:

- Path to model and number of training rounds (equal to the number of training_history files under model folder) to plot training loss and accuracy. For example:
model_name = 'models/Spain_Portugal/'
rounds = 2 # in how many runs was the model trained

- Name of model file to analyze validation data. For example:
model_version = 'model_v1.2.tflite'

For comparison with BirdNET, the same validation set should be analyzed with BirdNET by running following command on command line:
python3 analyze.py --i validation_data/ --o birdnet_results/ --min_conf 0.01

##############################################
# 7. Converting a new keras model into tflite
##############################################

To convert a keras model trained with BSG pipeline into a .tflite format for easier distribution run either 'Convert .keras model to .tflite.py' Python code or 'Convert .keras model to .tflite.ipynb' Jupyter notebook. 

To convert the model, please define in the script the path to the original .keras file. For example:

model_path = 'models/Spain_Portugal/BSG_birds_SpainAndPortugal_v1_2.keras'

All published BSG models have already been converted into .tflite and can be found under 'Run BSG models'/models/. 


