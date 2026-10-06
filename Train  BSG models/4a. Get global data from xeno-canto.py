import xenocanto as xc 
import pandas as pd
import json
import numpy as np
import wget
import time
import shutil
import os

# List species that have not been downloaded previously

path_to_species_list = "models/Spain_Portugal/species_list.csv"

previous_clips = pd.read_csv("---/xc_clips.csv") # path to metadata of previously processed xeno-canto clips

downloaded_species = np.unique(previous_clips['species_code'])

species_list = pd.read_csv(path_to_species_list)
species_list

species_to_download = list(set(species_list['species_code']) - set(downloaded_species))
new_species = species_list[species_list['species_code'].isin(species_to_download)]
xc_species = pd.read_csv("BSG_results/xc_species.csv")
print(f"Number of new species: {len(new_species)}")
new_species

# Retrieve the metadata for files to be downloaded from xeno-canto

batch_no = "_16" # running batch number 

out_path = "BSG_results/xc_files_to_download_batch" + batch_no + ".csv"
path_to_new_recs = '---/' # path where downloaded xeno-canto recordings are saved
path_to_old_recs = '---/' # path where previous xeno-canto recordings have been saved
path_to_metadata = '---/' # path where metadata of xeno-canto recordings is saved
n_per_species = 500 # up to 500 recordings from each species 

xc_data_files = []
for i in range(len(new_species)):
    sp_eng = new_species['xc_common_name'].iloc[i]
    sp_lat = new_species['xc_scientific_name'].iloc[i]
    n_files = xc_species["No."].loc[xc_species['Scientific name']==sp_lat].iloc[0]
    print(f"Downloading {sp_eng}, {n_files} recordings")
    if(n_files < n_per_species): 
        # download metadata
        %run xenocanto -m {sp_lat} 
        json_file = "dataset/metadata/" + ''.join(sp_lat.split()) + "/page1.json"
        # join to same file
        with open(json_file) as file:
            sp_data = json.load(file)
        species_data = pd.json_normalize(sp_data, record_path =['recordings'])
        if(len(xc_data_files) < 1):
            xc_data_files = species_data
        else:
            xc_data_files = pd.concat([xc_data_files, species_data])
    else:
        # download metadata one quality class at time
        count = 0
        quality = ["A", "B", "C", "D", "E"]
        q_i = 0
        while (count < n_per_species)&(q_i < 5):
            %run xenocanto -m {sp_lat} q:{quality[q_i]}
            json_file = "dataset/metadata/" + ''.join(sp_lat.split()) + "q_" + quality[q_i] + "/page1.json"
            with open(json_file) as file:
                sp_data = json.load(file)
            species_data = pd.json_normalize(sp_data, record_path =['recordings'])
            species_data = species_data.iloc[0:min(len(species_data), n_per_species-count)]
            count = count + len(species_data)
            if(len(xc_data_files) < 1):
                xc_data_files = species_data
            else:
                xc_data_files = pd.concat([xc_data_files, species_data])
            q_i = q_i+1   

xc_data_files.to_csv(out_path, index = False)
xc_data_files = pd.read_csv(out_path)

# Remove too long and short recordings
save = [0] *len(xc_data_files)
for i in range(len(xc_data_files)):
    duration = xc_data_files.loc[i, 'length']
    dur = sum(x * int(t) for x, t in zip([60, 1], duration.split(":"))) 
    if dur > 3 and dur < 600:
        save[i] = 1
xc_data_files['save'] = save
xc_data_files = xc_data_files.loc[xc_data_files['save'] == 1]
    
xc_data_files.to_csv(out_path, index = False)

# Download the audio files

in_path = "BSG_results/xc_files_to_download_batch" + batch_no + ".csv"

metadata = pd.read_csv(in_path)
# Download one batch at a time and sleep in between in order to not cause too much traffic to the site
batch_size = 100 # nbr of files to load at once
time_to_sleep = 60*1 # time in seconds between consecutive batches

i = 0

while(i < len(metadata)):
    url = metadata.loc[i, 'file']
    try:
        wget.download(url, out = path_to_new_recs + str(metadata.loc[i, 'id']) + batch_no + '.mp3')
    except:
        print("")
        print(f"File {url} ({i}) not found.")
    i = i+1
    if(i % batch_size == 0):
        print("")
        print(f"{i} / {len(metadata)} ({round(i/len(metadata)*100, 2)}%) completed, sleeping for {time_to_sleep/60} minutes...")
        time.sleep(time_to_sleep)

print()
print("Completed!")        

in_path = "BSG_results/xc_files_to_download_batch" + batch_no + ".csv"

metadata = pd.read_csv(in_path)
metadata['id'] = metadata['id'].astype(str) + batch_no

# Check that metadata matches the files and remove missing rows/files manually
new_files = os.listdir(path_to_new_recs)
print(f"Files downloaded: {len(new_files)}")
print(f"Files in metadata: {len(metadata)}")
print("Difference between sets:")
new_files2 = metadata['id'] + '.mp3'
print(list(set(new_files) - set(new_files2)))
print(list(set(new_files2) - set(new_files)))

# if there are failed downloads

#failed_downloads = list(set(new_files2) - set(new_files))
#failed_downloads = [f[:-4] for f in failed_downloads]
#metadata = metadata.loc[~metadata['id'].isin(failed_downloads)]

# Check that metadata matches the files and remove missing rows/files manually
#new_files = os.listdir(path_to_new_recs)
#print(f"Files downloaded: {len(new_files)}")
#print(f"Files in metadata: {len(metadata)}")
#print("Difference between sets:")
#new_files2 = metadata['id'] + '.mp3'
#print(list(set(new_files) - set(new_files2)))
#print(list(set(new_files2) - set(new_files)))

# Move all audio files from the temporary folder to final storage
for file_name in new_files:
    shutil.move(path_to_new_recs + file_name, path_to_old_recs + file_name)

# Join metadata
print("Joining metadata...")
all_metadata = pd.read_csv(path_to_metadata + '/xc_downloaded_files.csv')
print(f"Initially {all_metadata.shape}")
print(f"Without duplicates: {all_metadata.drop_duplicates().shape}")
new_metadata = metadata[all_metadata.columns]
new_metadata.to_csv(path_to_metadata + '/xc_downloaded_files_batch' + batch_no + '.csv', index = False) 
print(f"Added {new_metadata.shape}")
new_metadata = pd.concat([all_metadata, new_metadata], ignore_index = True)
print(f"Equals {new_metadata.shape}")
print(f"Without duplicates: {new_metadata.drop_duplicates().shape}")
new_metadata.to_csv(path_to_metadata + 'xc_downloaded_files_new.csv', index = False) 

# Recheck that metadata matches the files
print("Check that metadata matches the audio files...")
metadata_files = new_metadata['id']
metadata_files = [str(f) + '.mp3' for f in metadata_files]                           
audio_files = os.listdir(path_to_old_recs)
print(f"Audio files: {len(audio_files)}")
print(f"Rows in metadata: {len(metadata_files)}")
print(list(set(audio_files) - set(metadata_files)))
print(list(set(metadata_files) - set(audio_files)))

# Replace old metadata with new one once confirmed that everything is in order
os.remove(in_path) # remove temporary metadata for xeno-canto download
os.remove(path_to_metadata + 'xc_downloaded_files_batch' + batch_no + '.csv') # remove temporary metadata
os.remove(path_to_metadata + 'xc_downloaded_files.csv') # remove old metadata
os.rename(path_to_metadata + 'xc_downloaded_files_new.csv', 
          path_to_metadata + 'xc_downloaded_files.csv')

# Finally remove all ceno-canto metadata files from dataset/metadata/
os.remove("dataset/metadata")
os.mkdir("dataset/metadata")