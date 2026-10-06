import librosa
import os
import numpy as np
import pandas as pd
import random
import pickle
from IPython.utils import io
import operator
import shutil
import scipy 

# Data and paths
#Collect training and test data and set weights

# Define necessary paths
path_to_model = 'models/Argentina_Chaco/'
path_to_species_list = path_to_model + 'species_list.csv'

# Run only once to save the class labels
species_list = pd.read_csv(path_to_species_list)  
species_list = species_list[['species_code', 'common_name', 'scientific_name']]
no_bird_list = pd.DataFrame({'species_code':['nobird', 'human'], 'common_name':['No bird', 'Human'], 'scientific_name':['No bird', 'Human']})
species_list = pd.concat([no_bird_list, species_list]) # combine species list with non-bird classes
species_list['class'] = list(range(len(species_list)))

labels = {}    
for i in range(len(species_list)): # Create and save dictionary of classes
    labels[species_list['species_code'].iloc[i]] = species_list['class'].iloc[i]

with open(path_to_model + 'classes.pkl', 'wb') as f:
    pickle.dump(labels, f)
species_list.to_csv(path_to_model + 'classes.csv', index=False)
print(f"Saved following {len(labels)} labels:")
print(labels)

## Global data
# Collect all data and set weights

path_to_templates = "---/" # path where BSG templates have been saved
path_to_template_metadata = "---/bsg_templates.csv" # path where template metadata is saved
path_to_xc_metadata = "---/xc_clips.csv" # path to metadata of xeno-canto clips
path_to_noise_metadata = "---/noise_clips.csv" # path to noise metadata

# Load labels
with open(path_to_model + 'classes.pkl', 'rb') as f:
    labels = pickle.load(f)
    
n_classes = len(labels)

data_paths = []
weights_dict = {}
labels_dict = {}
high_q = {}
print("Defining weights...")

# BSG templates
print("    BSG templates...")
bsg_temp = pd.read_csv(path_to_template_metadata)
bsg_temp = bsg_temp[['Species_code', 'Filename']]
bsg_temp.columns =['species_code', 'file_name']
bsg_temp = bsg_temp.loc[bsg_temp['species_code'].isin(labels.keys())]
bsg_temp.reset_index(drop=True, inplace=True)
for i in range(len(bsg_temp)):
    labels_dict[bsg_temp['file_name'].iloc[i]] = keras.utils.to_categorical(labels[bsg_temp['species_code'].iloc[i]], num_classes=n_classes)
    weights_dict[bsg_temp['file_name'].iloc[i]] = 5
    high_q[bsg_temp['file_name'].iloc[i]] = True
    data_paths.append(path_to_templates + bsg_temp['file_name'].iloc[i])
    
# Xeno-canto clips and noise
print("    Xeno-canto clips...")
xc = pd.read_csv(path_to_xc_metadata)
xc = xc[['species_code', 'file_name']]
xc = xc.loc[xc['species_code'].isin(labels.keys())]
xc.reset_index(drop=True, inplace=True)
noise = pd.read_csv(path_to_noise_metadata)
sp_counts = pd.concat([xc['species_code'], noise['species_code']]).value_counts()
sp_counts = pd.DataFrame(sp_counts)
sp_counts['weight'] = 0
sp_counts.columns = ['count', 'weight']

for i in range(len(sp_counts)):
    sp_counts['weight'].iloc[i] = np.min([300/sp_counts['count'].iloc[i], 5.0]) # weights between 0.2 - 5
#    sp_counts['species_code'].loc['madcuh1']
sp_counts['total_weight']=sp_counts['count']*sp_counts['weight']

print(sp_counts.to_string())

path_to_xc_audio = '---/' # path to xeno-canto clips
path_to_noise_audio = '---/' # # path to noise clips

print("Assigning weights...")
print("    Xeno-canto data...")
for i in range(len(xc)):
    labels_dict[xc['file_name'].iloc[i]] = keras.utils.to_categorical(labels[xc['species_code'].iloc[i]], num_classes=n_classes)
    weights_dict[xc['file_name'].iloc[i]] = sp_counts.at[xc['species_code'].iloc[i], 'weight'] # weights between 0.2 - 5
    high_q[xc['file_name'].iloc[i]] = True
    data_paths.append(path_to_xc_audio + xc['file_name'].iloc[i])
print("    Non-bird vocalizations...")
for i in range(len(noise)):
    labels_dict[noise['file_name'].iloc[i]] = keras.utils.to_categorical(labels[noise['species_code'].iloc[i]], num_classes=n_classes)
    weights_dict[noise['file_name'].iloc[i]] = sp_counts.at[xc['species_code'].iloc[i], 'weight']
    high_q[noise['file_name'].iloc[i]] = True
    data_paths.append(path_to_noise_audio + noise['file_name'].iloc[i])
print("Complete.")

# Split to train, validation and test set and save to training_data/

val_prop = 20 # select every n:th element to test set

# split to train and test
val_data = data_paths[::val_prop]
train_data = np.setdiff1d(data_paths, val_data)

try:
    os.mkdir(path_to_model + 'global_training_data/')
except:
    print("Folder .../training_data already exists")
    
# Copy training data to own folder
for d in [train_data, val_data]:
    print(f"Copying {len(d)} files...")
    count = 0
    for f in d:
        f_s = f.split('/')
        f_n = f_s[len(f_s)-1]
        shutil.copyfile(f, path_to_model + 'global_training_data/' + f_n)
        count = count+1
        if count % 100 == 0:
            print(f"Processed {count}/{len(d)} ({count/len(d)*100:.2f} %) files      ", end = "\r")
    print("P")
print("Complete.")

train_data = [f.split('/')[-1] for f in train_data]
val_data = [f.split('/')[-1] for f in val_data]

# Save weights, labels data_paths, etc
p = path_to_model + 'global_training_data/metadata/'
os.mkdir(p)

with open(p + 'weights.pkl', 'wb') as f:
    pickle.dump(weights_dict, f)
with open(p + 'labels.pkl', 'wb') as f:
    pickle.dump(labels_dict, f)
with open(p + 'high_q.pkl', 'wb') as f:
    pickle.dump(high_q, f)
with open(p + 'train_set.pkl', 'wb') as f:
    pickle.dump(train_data, f)
with open(p + 'val_set.pkl', 'wb') as f:
    pickle.dump(val_data, f)

shutil.copyfile(path_to_model + 'classes.pkl', p + 'classes.pkl') 

## Local data

path_to_bsg_metadata = '---/BSG_soundscapes.csv' # path to BSG soundscape metadata
path_to_bsg_labels = '---/BSG_labels.csv' # path to BSG labels
path_to_bsg_audio = '---/' # path where BSG clips have been saved 

# Load labels
with open(path_to_model + 'classes.pkl', 'rb') as f:
    labels = pickle.load(f)

n_classes = len(labels)
site_list = np.load(path_to_model + 'site_list.npy')

data_paths = []
weights_dict = {}
labels_dict = {}
high_q = {}

print("Defining weights...")

# Load all metadata and get class counts
bsg_ssc = pd.read_csv(path_to_bsg_metadata)
bsg_ssc = bsg_ssc.loc[bsg_ssc['site_id'].isin(site_list)]
bsg_ssc.reset_index(drop=True, inplace=True)
bsg_ssc_labels = pd.read_csv(path_to_bsg_labels) 
bsg_ssc_labels = bsg_ssc_labels.loc[bsg_ssc_labels['site_id'].isin(site_list)]

sp_counts_loc = pd.DataFrame({'count':0}, index=pd.Index(labels.keys(), name='species'))
val_counts = bsg_ssc_labels['species'].value_counts()
for k in val_counts.keys():
    sp_counts_loc.loc[k, 'count'] = val_counts[k]
sp_counts_loc = sp_counts_loc.sort_values('count', ascending=False)
sp_counts_loc['weight'] = 0.0

for sp in list(sp_counts_loc.index):
    sp_counts_loc.loc[sp, 'weight'] = np.min([500/sp_counts_loc.loc[sp, 'count'], 2]) # weights between 0 - 2
sp_counts_loc['total_weight']=sp_counts_loc['count']*sp_counts_loc['weight']
sp_counts_loc['missing'] = 500 - sp_counts_loc['total_weight'].round().astype('int')
print("Done")

print(sp_counts_loc.to_string())

# BSG soundscapes
print("Collecting data...")
print("    BSG soundscapes...")
for i in range(len(bsg_ssc)):
    temp_labs = bsg_ssc_labels.loc[bsg_ssc_labels['file_name'] == bsg_ssc['file_name'].iloc[i]]
    lab = np.zeros(n_classes, dtype='float32')
    for j in range(len(temp_labs)):
        if temp_labs['species'].iloc[j] in labels.keys(): # skip species that are not included in the model
            lab[labels[temp_labs['species'].iloc[j]]] = temp_labs['occurrence'].iloc[j] 
    labels_dict[bsg_ssc['file_name'].iloc[i]] = lab
    w = 0 # initialize weight
    for sp in temp_labs['species']:
        if sp in labels.keys():
            w_new = sp_counts_loc['weight'].loc[sp]
            w = np.max([w, w_new]) # select weight of the rarest species
    if w == 0: # species that is not contained in the model
        w = 1 
    weights_dict[bsg_ssc['file_name'].iloc[i]] = w
    high_q[bsg_ssc['file_name'].iloc[i]] = False
    data_paths.append(path_to_bsg_audio + bsg_ssc['file_name'].iloc[i])

sp_counts_loc['total_weight_2']=sp_counts_loc['total_weight']
# Supplement with BSG templates if total weight less than 500
print("    BSG templates...")
bsg_temp = pd.read_csv(path_to_template_metadata)
bsg_temp = bsg_temp[['Species_code', 'Filename']]
bsg_temp.columns =['species_code', 'file_name']
bsg_temp = bsg_temp.loc[bsg_temp['species_code'].isin(labels.keys())]
bsg_temp.reset_index(drop=True, inplace=True)
for sp in list(sp_counts_loc.index):
    if sp_counts_loc['missing'].loc[sp] > 0:
        bsg_temp_sp = bsg_temp.loc[bsg_temp['species_code']==sp]
        n_templates = len(bsg_temp_sp)
        if n_templates > 0:
            for i in range(len(bsg_temp_sp)):
                labels_dict[bsg_temp_sp['file_name'].iloc[i]] = keras.utils.to_categorical(labels[bsg_temp_sp['species_code'].iloc[i]], num_classes=n_classes)
                weights_dict[bsg_temp_sp['file_name'].iloc[i]] = 1
                high_q[bsg_temp_sp['file_name'].iloc[i]] = True
                data_paths.append(path_to_templates + bsg_temp_sp['file_name'].iloc[i])
        sp_counts_loc.loc[sp, 'total_weight_2'] = sp_counts_loc['total_weight_2'].loc[sp] + n_templates
sp_counts_loc['missing_2'] = 500 - sp_counts_loc['total_weight_2'].round().astype('int')

sp_counts_loc['total_weight_final']=sp_counts_loc['total_weight_2'] 
# Supplement with xeno-canto data if total weight still less than 500
print("    Xeno-canto clips...")
xc = pd.read_csv(path_to_xc_metadata)
xc = xc[['species_code', 'file_name']]
xc = xc.loc[xc['species_code'].isin(labels.keys())]
xc.reset_index(drop=True, inplace=True)
for sp in list(sp_counts_loc.index):
    missing = sp_counts_loc['missing_2'].loc[sp]
    if missing > 0:
        xc_sp = xc.loc[xc['species_code']==sp]
        xc_sp = xc_sp.sample(min((2*missing, len(xc_sp))))
        n_templates = len(xc_sp)
        if n_templates >= missing:
            w = 0.5
        else:
            w = 1
        if n_templates > 0:
            for i in range(len(xc_sp)):
                labels_dict[xc_sp['file_name'].iloc[i]] = keras.utils.to_categorical(labels[xc_sp['species_code'].iloc[i]], num_classes=n_classes)
                weights_dict[xc_sp['file_name'].iloc[i]] = w
                high_q[xc_sp['file_name'].iloc[i]] = True
                data_paths.append(path_to_xc_audio + xc_sp['file_name'].iloc[i])          
        sp_counts_loc.loc[sp, 'total_weight_final'] = sp_counts_loc['total_weight_2'].loc[sp] + w*n_templates
print("    Noise...")
noise = pd.read_csv(path_to_noise_metadata)
noise = noise[['species_code', 'file_name']]
for sp in list(['human', 'nobird']):
    missing = sp_counts_loc['missing_2'].loc[sp]
    if missing > 0:
        noise_sp = noise.loc[noise['species_code']==sp]
        noise_sp = noise_sp.sample(min((2*missing, len(noise_sp))))
        n_templates = len(noise_sp)
        if n_templates >= missing:
            w = 0.5
        else:
            w = 1
        if n_templates > 0:
            for i in range(len(noise_sp)):
                labels_dict[noise_sp['file_name'].iloc[i]] = keras.utils.to_categorical(labels[noise_sp['species_code'].iloc[i]], num_classes=n_classes)
                weights_dict[noise_sp['file_name'].iloc[i]] = w
                high_q[noise_sp['file_name'].iloc[i]] = True
                data_paths.append(path_to_noise_audio + noise_sp['file_name'].iloc[i])          
        sp_counts_loc.loc[sp, 'total_weight_final'] = sp_counts_loc['total_weight_2'].loc[sp] + w*n_templates

sp_counts_loc = sp_counts_loc.sort_values('total_weight_final', ascending=False)        

print("Data collected.")

print(sp_counts_loc.to_string())

# Check that all species are covered
set(labels.keys()) - set(sp_counts_loc.index)

# Make sure that original and cleaned version of the same clip end up in the same set:

val_prop = 10 # select every n:th element to test set

files = [f for f in data_paths if f.split('/')[-2] == 'BSG_soundscapes']
fids = [f.split('/')[-1].split('_')[0] for f in files]
fids = np.unique(fids)
val_fids = fids[::val_prop]
val_data = [f for f in files if f.split('/')[-1].split('_')[0] in val_fids]

os.mkdir(path_to_model + 'local_training_data/')

train_data = np.setdiff1d(data_paths, val_data)

# Copy training data to own folder
for d in [train_data, val_data]:
    print(f"Copying {len(d)} files...")
    count = 0
    for f in d:
        f_s = f.split('/')
        f_n = f_s[len(f_s)-1]
        shutil.copyfile(f, path_to_model + 'local_training_data/' + f_n)
        count = count+1
        if count % 100 == 0:
            print(f"Processed {count}/{len(d)} ({count/len(d)*100:.2f} %) files      ", end = "\r")
    print("P")
print("Complete.")

train_data = [f.split('/')[-1] for f in train_data]
val_data = [f.split('/')[-1] for f in val_data]

# Save weights, labels data_paths, etc
p = path_to_model + 'local_training_data/metadata/'
os.mkdir(p)

with open(p + 'weights.pkl', 'wb') as f:
    pickle.dump(weights_dict, f)
with open(p + 'labels.pkl', 'wb') as f:
    pickle.dump(labels_dict, f)
with open(p + 'high_q.pkl', 'wb') as f:
    pickle.dump(high_q, f)
with open(p + 'train_set.pkl', 'wb') as f:
    pickle.dump(train_data, f)
with open(p + 'val_set.pkl', 'wb') as f:
    pickle.dump(val_data, f)
shutil.copyfile(path_to_model + 'classes.pkl', p+ 'classes.pkl') 