import pandas as pd
import os
import numpy as np
import soundfile as sf
import librosa
from IPython.utils import io

from functions import butter_bandpass, butter_bandpass_filter
from augmentation import pad

# Annotation results are first obtaned from BSG portal
# This is currently not openly available

# exclude non-bird test sites from data 
excluded_sites=[10001, 10002, 10003, 10005, 10006, 10007] 

#################
# BSG templates
#################

data = pd.read_csv("BSG_data/bsg_results.tsv", sep = "\t")
data['url'] = [f.split(".laji.fi/")[1][0:7] for f in data['url']]
metadata = pd.read_csv("BSG_data/BSG_template_files_metadata.csv") 

path_in = "---/"  # path to original audio files from xeno-canto/Macaulay library
path_to_birdsound_library = "---/" # path where created templates are saved
path_to_template_metadata = "---/bsg_templates.csv" # path where template metadata is saved

sp_ind = 0
prev_sp = ""
print("Saving annotated BSG templates...")
n_row = len(data)
template_metadata = pd.DataFrame()

for i in range(n_row):
    splitted = data['file_name'].iloc[i].split('_')
    sp = data['scientific_name'].iloc[i]
    source = data['url'].iloc[i]
    sp_code = data['species_code'].iloc[i]
    start_min = 0
    if  source == 'xeno-ca': # source: xeno_canto
        source = 'xenocanto'
        fid = splitted[0].split('.')[0] 
    elif source == 'cornell': # source: Macaulay library
        source = 'macaulay'
        fid = splitted[0]
        end = splitted[1].split('.')[0]
        start_min = int(end)
    else:
        print("Invalid source!")
    sp_sci = metadata['Species_sci'].loc[(metadata['File'] == int(fid)) & (metadata['Source']==source)].iloc[0]
    if(sp_sci != sp):
        print(f"File {i}: Species names not matching: {sp} <-> {sp_sci}")
    path = metadata['filename'].loc[(metadata['File'] == int(fid)) & (metadata['Source']==source)].iloc[0]
    path_full = path_in + source + '/' + path 
    x1 = data['x1'].iloc[i]
    x2 = data['x2'].iloc[i]
    y1 = data['y1'].iloc[i]
    y2 = data['y2'].iloc[i]
    if(x2-x1 > 3): # define the start and end of the clip
        start = x1
        stop = x2
    else:
        start = np.max([((x2+x1)/2)-1.5,0])
        stop = x1+3
    with io.capture_output() as captured:
        sig, sr = librosa.load(path_full, sr = 48000, offset = start_min*600+start, duration = stop-start) # original version
    sig2 = pad(sig, x1- start, x2-start, target_len = len(sig), sr=48000) # time- and frequency cropped version
    sig2 = butter_bandpass_filter(sig2, [y1,y2], 48000, 12) 
    # SAVE
    if(sp_code == prev_sp):
        sp_ind = sp_ind +1
    else:
        prev_sp = sp_code
        sp_ind = 1
    filename_out = sp_code + '_' + str(sp_ind)
    sig = librosa.resample(sig, orig_sr=48000, target_sr=24000) # resample to 24 kHz
    sig2 = librosa.resample(sig2, orig_sr=48000, target_sr=24000)
    sf.write(path_to_birdsound_library + filename_out + '_orig.wav', sig, 24000)
    sf.write(path_to_birdsound_library + filename_out + '_cleaned.wav', sig2, 24000)
    # METADATA
    template_metadata = pd.concat([template_metadata, pd.DataFrame({"Filename": [filename_out + '_orig.wav', filename_out + '_cleaned.wav'], 
                                                                    "Species_code": [sp_code, sp_code], "Species": [sp, sp], 
                                                                    "original":[1, 0], "file_id":[fid, fid], "source":[source, source]})])
    if i % 100 == 0:
        print(f"{i}/{n_row} ({round(100*(i/n_row), 2)} %)...", end='\r')
        template_metadata.to_csv(path_to_template_metadata, index = False)

template_metadata.to_csv(path_to_template_metadata, index = False)
print("Complete!                             ")        

###########################
# BSG soundscapes
###########################

# Collect data from all users to one file 
species_annotations_path = 'BSG_data/bsg_identification_results/species_annotations/' 
annotations_path = 'BSG_data/bsg_identification_results/annotations/' 
users = os.listdir(species_annotations_path)

additional_taxa = {'Bombina variegata': ['fr_ybtoad', 'Yellow-bellied Toad'],
                   'Barbastella barbastellus':['bat_1', 'bat_1'], 'Eptesicus nilssonii':['bat_2', 'bat_2'],'Eptesicus serotinus':['bat_3', 'bat_3'],
                   'Hypsugo savii':['bat_4', 'bat_4'], 'Miniopterus schreibersii':['bat_5', 'bat_5'],'Myotis alcathoe':['bat_6', 'bat_6'], 
                   'Myotis blythii':['bat_7', 'bat_7'], 'Myotis capaccinii':['bat_8', 'bat_8'],'Myotis crypticus':['bat_9', 'bat_9'], 
                   'Myotis daubentonii':['bat_10', 'bat_10'], 'Nyctalus lasiopterus':['bat_11', 'bat_11'], 'Nyctalus leisleri':['bat_12', 'bat_12'], 
                   'Nyctalus noctula':['bat_13', 'bat_13'], 'Pipistrellus kuhlii':['bat_14', 'bat_14'], 'Pipistrellus nathusii':['bat_15', 'bat_15'],
                   'Pipistrellus pipistrellus':['bat_16', 'bat_16'], 'Pipistrellus pygmaeus':['bat_17', 'bat_17'], 'Plecotus auritus':['bat_18', 'bat_18'], 
                   'Plecotus austriacus':['bat_19', 'bat_19'],'Rhinolophus euryale':['bat_20', 'bat_20'], 'Rhinolophus ferrumequinum':['bat_21', 'bat_21'],
                   'Rhinolophus hipposideros':['bat_22', 'bat_22'], 'Tadarida teniotis':['bat_23', 'bat_23'], 'Vespertilio murinus':['bat_24', 'bat_24'],
                   'Rhinolophus\xa0mehelyi':['bat_25', 'bat_25'], 'Mix Myo50':['bat_26', 'bat_26'], 'Mix Myo30':['bat_27', 'bat_27'], 'Plecotus sp':['bat_28', 'bat_28'],
                   'Mix EptNycVes':['bat_29', 'bat_29'], 'Mix TadNyc':['bat_30', 'bat_30'],
                   'Aethiomerus madagassus':['bug_1', 'bug_1'], 'Ambylakis sp':['bug_2', 'bug_2'], 'Listroscelidinae ngen nsp':['bug_3', 'bug_3'],
                   'Neozvenella nsp':['bug_4', 'bug_4'], 'Odontolakis sp':['bug_5', 'bug_5'], 'Odontolakis virescens':['bug_6', 'bug_6'],
                   'Oecanthus brevicauda':['bug_7', 'bug_7'],'Parasimodera nsp':['bug_8', 'bug_8'], 'Paragryllodes sp':['bug_9', 'bug_9']}

data_10s = pd.DataFrame()

for i, u in enumerate(users):
    user_data1 = pd.read_csv(species_annotations_path + u, sep = '\t')
    user_data2 = pd.read_csv(annotations_path + u, sep = '\t')
    d ={"species_annotation_id":'', "recording_id":user_data2['recording_id'], "annotation_id":user_data2['annotation_id'], 
       "species_code":"other", "scientific_name":"Other", "common_name":"Other", "occurrence":user_data2['contains_unknown_birds'].astype('int')}
    user_data1 = pd.concat([user_data1, pd.DataFrame(d)], ignore_index=True)
    user_data1['id_by'] = u[:-4]
    user_data1['occurrence'] = user_data1['occurrence'].replace(2, 0.5)
    # manually fix non-bird taxa
    if any(user_data1.isna().any()):
        for j in range(len(user_data1)):
            if pd.isna(user_data1['species_code'].iloc[j]):
                user_data1.loc[j, 'species_code'] = additional_taxa[user_data1['scientific_name'].iloc[j]][0]
                user_data1.loc[j, 'common_name'] = additional_taxa[user_data1['scientific_name'].iloc[j]][1]
    data_10s = pd.concat([data_10s, user_data1], ignore_index = True)
    print(f"Processed user {u} ({i+1}/{len(users)})     ", end = "\r")
print("Data from all users collected.       ")

# Combine the data from different users

print("Combining annotations...")
final_data = pd.DataFrame()
rec_ids = np.unique(data_10s['recording_id'])
n_rec = len(rec_ids)
count = 0
for rec_id in rec_ids : # loop through recordings
    rec_data = data_10s.loc[data_10s['recording_id'] == rec_id]
    for sp in np.unique(rec_data['species_code']): # loop through species
        if (sp == "other"): # save whether there are other species or not, and the number of users who agree on this
            other_sps_occ = np.min(rec_data['occurrence'].loc[rec_data['species_code'] == "other"]) 
            d = {"species_code":[sp], "recording_id":rec_id, "occurrence":other_sps_occ, 
                   "n_users":len(rec_data.loc[((rec_data['species_code'] == "other") & (rec_data['occurrence'] == other_sps_occ))])}
            final_data = pd.concat([final_data, pd.DataFrame(d)])
        else: 
            sp_occ = []
            for u in np.unique(rec_data['id_by']): # loop through users
                user_sp_id = rec_data[((rec_data['species_code'] == sp) & (rec_data['id_by'] == u))]['occurrence']
                if len(user_sp_id)>0:
                    sp_occ.append(user_sp_id.iloc[0]) # Collect all annotations from different users for the same species
                    if(len(user_sp_id)>1):
                        print("MAYDAY!!")
                elif (rec_data['occurrence'].loc[((rec_data['species_code'] == "other") & (rec_data['id_by']==u))].iloc[0]==0):
                    sp_occ.append(0) # Save negative identification if user states that the recording does not contain other species 
            d = {"species_code":[sp], "recording_id":rec_id, "occurrence":np.mean(sp_occ), 
                   "n_users":len(sp_occ)}
            final_data = pd.concat([final_data, pd.DataFrame(d)])
    count = count+1
    if count % 100 == 0:
        print(f"Processed {count}/{len(rec_ids)} ({round(count/len(rec_ids)*100, 2)} %) recordings", end = "\r")
print("All annotations combined.                                    ")    

# Add metadata for files and save the data 

recordings_path = 'BSG_data/bsg_identification_results/recordings.tsv' 
mp3_buffers_path = 'BSG_data/mp3_20s.txt'

recordings = pd.read_csv(recordings_path, sep = '\t')
recording_splits = pd.read_csv(mp3_buffers_path, sep = ' ', header = None, names = ['file_name', 'start_buff', 'sample', 'end_buff'])
recordings = recordings[['recording_id', 'site_id', 'file_name']]
final_data = final_data.merge(recordings, on='recording_id', how='left')
final_data = final_data.merge(recording_splits, on='file_name', how='left')

final_data = final_data.loc[~final_data['site_id'].isin(excluded_sites)] # remove data from excluded sites

missing_metadata = final_data.loc[final_data['start_buff'].isna()]
if(len(missing_metadata) > 0):
    print(f"Check that BSG_data/mp3_20s.txt is up to date. Metadata missing for {len(missing_metadata)} files:")
    print(np.unique(missing_metadata['file_name']))

final_data.to_csv("BSG_results/10s_annotations.csv", index = False)
print("10s annotations saved.")

# find and save clips

path_species = 'BSG_data/bsg_identification_results/species_annotations/'
path_boxes = 'BSG_data/bsg_identification_results/species_annotation_boxes/'
path_info = 'BSG_data/bsg_identification_results/annotations/'
path_recordings = 'BSG_data/bsg_identification_results/recordings.tsv'
path_mp3_buffers = 'BSG_data/mp3_20s.txt'

path_to_soundscapes = '---/' # path to original soundscape recordings
path_to_birdsound_library = '---/' # path where created clips are saved

raw_box_data_out = 'BSG_results/raw_box_data.csv'
processed_box_data_out = 'BSG_results/processed_box_data.csv'
bsg_metadata_out = '---/' # path where clip metadata is saved

recordings = pd.read_csv(path_recordings, sep = "\t")
recordings = recordings[['recording_id','site_id','file_name']]
recordings = recordings.loc[~recordings['site_id'].isin(excluded_sites)] # remove data from excluded sites

######################################################################
# Preprocess boxes and join small boxes that are close to each other #
######################################################################

print("Preprocessing box data...")
# Combine data from all users
file_list = os.listdir(path = path_boxes) # list users who have made identifications
data = pd.DataFrame()
for i in range(len(file_list)):
    temp_data1 = pd.read_csv(path_boxes + file_list[i], sep = '\t') # read user-specific files
    temp_data2 = pd.read_csv(path_species + file_list[i], sep = '\t')
    temp_data3 = pd.read_csv(path_info + file_list[i], sep = '\t')
    temp_data3 = temp_data3[['recording_id', 'has_boxes_for_all_bird_sounds']]
    temp_data = pd.merge(temp_data1, temp_data2, on="species_annotation_id", how='left') # merge files
    temp_data = pd.merge(temp_data, temp_data3, on="recording_id", how='left')
    temp_data['occurrence'] = temp_data['occurrence'].replace(2, 0.5) # unsure annotations correspond to 0.5
    data = pd.concat([data, temp_data]) # join with data from other users
data=data.reset_index(drop=True)
data.to_csv(raw_box_data_out, index=False)

# Combine boxes from same species that are next to each other
# If several users have drawn different boxes for the same species, the boxes will be joined here
data = data.sort_values(by=['recording_id','species_code', 'area_x1'], ascending = [True, True, True])
data = pd.concat([data, pd.DataFrame({"species_annotation_id":[0], "area_x1":[0], "area_x2":[0], "area_y1":[0], "area_y2":[0], "overlaps_with_other_species":False, "recording_id":[0], "annotation_id":[0], 
                                      "species_code":"last_row", "scientific_name":"last_row", "common_name":"last_row", "occurrence":[0], "has_boxes_for_all_bird_sounds":False})])
data=data.reset_index(drop=True)
# manually fix non-bird taxa
print("  Fixing names for non-bird taxa...")
if any(data.isna().any()):
    for j in range(len(data)):
        if pd.isna(data['species_code'].iloc[j]):
            data.loc[j, 'species_code'] = additional_taxa[data['scientific_name'].iloc[j]][0]
            data.loc[j, 'common_name'] = additional_taxa[data['scientific_name'].iloc[j]][1]
check_data_len = len(data)
data['sound_type'] = data['sound_type'].fillna(0) # replace unnecessary sound_type na:s with 0
data = data.dropna()
check_data_len2 = len(data)
if check_data_len != check_data_len2:
    print(f"Omitted {check_data_len - check_data_len2} rows due to NAs.")
data = data.loc[data['recording_id'].isin(recordings['recording_id'])] # remove data from excluded sites
data=data.reset_index(drop=True)

print("  Joining boxes...")
annotations_10s = pd.read_csv("BSG_results/10s_annotations.csv")
i = 0
prev_rec = 0
row_idxs = []
while(True):
    if(i == len(data)-1):
        break
    j = i
    while((data['recording_id'].iloc[j+1] == data['recording_id'].iloc[j]) & (data['species_code'].iloc[j+1] == data['species_code'].iloc[j]) & (data['area_x1'].iloc[j+1]-data['area_x2'].iloc[j]<1)):
        j = j+1
    if data['recording_id'].iloc[j] != prev_rec: # select all boxes from same recording for overlapping comparison
        prev_rec = data['recording_id'].iloc[j]
        rec_data = data.loc[data['recording_id'] == prev_rec]
    sub_data = data.iloc[i:j+1] # select all boxes to join
    #save box only if majority of annotators agrees that species is present in recording
    if annotations_10s['occurrence'].loc[((annotations_10s['species_code']==data['species_code'].iloc[i]) & (annotations_10s['recording_id']==data['recording_id'].iloc[i]))].iloc[0] >= 0.5:
        if j > i: #if there are several boxes to join, calculate new metadata for the box
            data.loc[i,'area_x1'] = np.min(sub_data['area_x1'])
            data.loc[i, 'area_x2'] = np.max(sub_data['area_x2'])
            data.loc[i, 'area_y1'] = np.min(sub_data['area_y1'])
            data.loc[i, 'area_y2'] = np.max(sub_data['area_y2'])
            data.loc[i, 'overlaps_with_other_species'] = sub_data['overlaps_with_other_species'].any()
            data.loc[i, 'occurrence'] = np.max(sub_data['occurrence'])
            rec_data_other_sps = rec_data.loc[rec_data['species_code']!=data['species_code'].iloc[i]] # Check if the resulting box overlaps with boxes of other species
            for sub_i in range(len(rec_data_other_sps)):
                if (((data.loc[i, 'area_x2']> rec_data_other_sps['area_x1'].iloc[sub_i]) & (data.loc[i, 'area_y2']> rec_data_other_sps['area_y1'].iloc[sub_i])) & ((data.loc[i, 'area_x1']< rec_data_other_sps['area_x2'].iloc[sub_i]) & (data.loc[i, 'area_y1']< rec_data_other_sps['area_y2'].iloc[sub_i]))):
                    data.loc[i, 'overlaps_with_other_species'] = True
        row_idxs.append(i)
    i = j+1
joint_boxes_data=data.iloc[row_idxs]
joint_boxes_data.reset_index(drop=True, inplace=True)
joint_boxes_data=pd.merge(joint_boxes_data, recordings, on="recording_id", how='left')
joint_boxes_data.to_csv(processed_box_data_out, index=False)
print("Box data preprocessing complete.")

###############################
# Save clips and their labels #
###############################
bsg_metadata = pd.DataFrame()
bsg_labels = pd.DataFrame()

print("Saving clips...")
running_idx = 1

# Empty and non-bird clips
file_list = os.listdir(path_info)
rec_info_data = pd.DataFrame()
for i in range(len(file_list)):
    temp_data = pd.read_csv(path_info + file_list[i], sep='\t')
    rec_info_data = pd.concat([rec_info_data, temp_data])
rec_info_data=pd.merge(rec_info_data, recordings, on="recording_id", how='right')

print("    Processing empty clips...")
# Choose only recordings, where all agree that no birds occur: sort by does_not_contain_birds and drop duplicates
empty_data = rec_info_data.sort_values('does_not_contain_birds', ascending=True)
empty_data.reset_index(drop=True, inplace=True)
empty_data.drop_duplicates(subset=['recording_id'], keep='first', inplace=True)
empty_data = empty_data.loc[((empty_data['does_not_contain_birds']==True)&(empty_data['contains_unknown_birds']==False))] # filter by both attributes just in case
empty_data = empty_data[~empty_data['recording_id'].isin(joint_boxes_data['recording_id'])] # filter out those cases where somebody has still drawn boxes for birds
empty_data.reset_index(drop=True, inplace=True)
mp3_buffers_path = 'BSG_data/mp3_20s.txt'
recording_splits = pd.read_csv(mp3_buffers_path, sep = ' ', header = None, names = ['file_name', 'start_buff', 'sample', 'end_buff'])
empty_data = empty_data.merge(recording_splits, on='file_name', how='left')

for i in range(len(empty_data)): # Clip to 4s frames and save
    buffer = empty_data['start_buff'].iloc[i]
    sample_len = empty_data['sample'].iloc[i]
    x1 = buffer
    while(x1+4 < buffer+sample_len):
        with io.capture_output() as captured:
            sig, sr = librosa.load(path_to_soundscapes  + empty_data['file_name'].iloc[i], sr = 24000, offset = x1, duration = 4)
        file_out = str(empty_data['recording_id'].iloc[i]) + '_' + str(round(x1,1)) + '_' + str(running_idx) + '.wav'
        sf.write(path_to_birdsound_library + file_out, sig, 24000)
        running_idx = running_idx+1  
        bsg_metadata = pd.concat([bsg_metadata, pd.DataFrame({"file_name":file_out, "site_id":[empty_data['site_id'].iloc[i]], 
                                                                   "cleaned":False, "no_other_species":True, "noise":True})])
        bsg_labels = pd.concat([bsg_labels, pd.DataFrame({"file_name":file_out, "site_id":[empty_data['site_id'].iloc[i]], 
                                                        "species":"nobird", "occurrence":[1]})])
        x1=x1+3
        if i % 100 == 0:
            print(f"    Processed {i}/{len(empty_data)} ({round(i/len(empty_data)*100, 2)} %) clips   ", end = "\r")
      
                                                          
print("    Processing non-bird sounds...                             ")
non_bird_data = rec_info_data.dropna(subset=['non_bird_area_x1'])
non_bird_data.reset_index(drop=True, inplace=True)

for i in range(len(non_bird_data)): # loop through non-bird boxes
    file_name = non_bird_data['file_name'].iloc[i]
    full_rec_length = recording_splits[['start_buff', 'sample', 'end_buff']].loc[recording_splits['file_name']==file_name].sum(1).iloc[0]
    x1 = non_bird_data['non_bird_area_x1'].iloc[i]
    x2 = non_bird_data['non_bird_area_x2'].iloc[i]
    if x2-x1 > 4:
        start = x1
        stop = x2
    else:
        mean_x = np.mean([x1, x2])
        start = mean_x -2
        stop = mean_x + 2
    if start < 0:
        start = 0
        stop = np.max([stop, 3])
    if stop > full_rec_length:
        stop = full_rec_length
        start = np.min([start, full_rec_length-3])
    with io.capture_output() as captured:
        sig, sr = librosa.load(path_to_soundscapes + file_name, sr = 48000, offset = start, duration = stop-start)
    sig = pad(sig, x1-start, x2-start, snr=1, target_len=len(sig), sr=48000) # time- and frequency cropped version
    sig = butter_bandpass_filter(sig, [non_bird_data['non_bird_area_y1'].iloc[i],non_bird_data['non_bird_area_y2'].iloc[i]], 48000, 12) 
    sig = librosa.resample(sig, orig_sr=48000, target_sr=24000)
    file_out = str(non_bird_data['recording_id'].iloc[i]) + '_' + str(round(start,1)) + '_' + str(running_idx) + '.wav'
    sf.write(path_to_birdsound_library + file_out, sig, 24000)
    running_idx = running_idx+1  
    bsg_metadata = pd.concat([bsg_metadata, pd.DataFrame({"file_name":file_out, "site_id":[non_bird_data['site_id'].iloc[i]], 
                                                                   "cleaned":True, "no_other_species":True, "noise":True})])
    bsg_labels = pd.concat([bsg_labels, pd.DataFrame({"file_name":file_out, "site_id":[non_bird_data['site_id'].iloc[i]], 
                                                        "species":"nobird", "occurrence":[1]})])
    if i % 100 == 0:
        print(f"    Processed {i}/{len(non_bird_data)} ({round(i/len(non_bird_data)*100, 2)} %) clips   ", end = "\r")
                                                      

print("    Processing bird vocalizations...             ")
prev_rec=0
for i in range(len(joint_boxes_data)):
    if joint_boxes_data['recording_id'].iloc[i] != prev_rec: # select all boxes from same recording for overlapping comparison
        prev_rec = joint_boxes_data['recording_id'].iloc[i]
        rec_data = joint_boxes_data.loc[joint_boxes_data['recording_id'] == prev_rec]
    # Define start and stop for the clip
    file_name = joint_boxes_data['file_name'].iloc[i]
    full_rec_length = recording_splits[['start_buff', 'sample', 'end_buff']].loc[recording_splits['file_name']==file_name].sum(1).iloc[0]
    x1 = joint_boxes_data['area_x1'].iloc[i]
    x2 = joint_boxes_data['area_x2'].iloc[i]
    if x2-x1 > 3:
        start = x1
        stop = x2
    else:
        mean_x = np.mean([x1, x2])
        start = mean_x -2
        stop = mean_x + 2
    if start < 0:
        start = 0
        stop = np.max([stop, 3])
    if stop > full_rec_length:
        stop = full_rec_length
        start = np.min([start, full_rec_length-3])
    with io.capture_output() as captured:
        sig, sr = librosa.load(path_to_soundscapes + file_name, sr = 48000, offset = start, duration = stop-start)
    # collect information of all species occurring in the clip
    sps = rec_data.loc[((rec_data['area_x1'] < stop-1) & (rec_data['area_x2'] > start+1))]
    sps = sps.sort_values('occurrence', ascending=False)
    sps.reset_index(drop=True, inplace=True)
    sps.drop_duplicates(subset=['species_code'], keep='first', inplace=True)
    # load signal
    sig1 = librosa.resample(sig, orig_sr=48000, target_sr=24000)
    file_out = str(joint_boxes_data['recording_id'].iloc[i]) + '_' + str(round(start,1)) + '_' + str(running_idx) + '.wav'
    sf.write(path_to_birdsound_library + file_out, sig1, 24000)
    running_idx = running_idx+1 
    bsg_metadata = pd.concat([bsg_metadata, pd.DataFrame({"file_name":file_out, "site_id":[joint_boxes_data['site_id'].iloc[i]], 
                                                        "cleaned":False, "no_other_species":joint_boxes_data['has_boxes_for_all_bird_sounds'].iloc[i], 
                                                          "noise":False})])
    bsg_labels = pd.concat([bsg_labels, pd.DataFrame({"file_name":file_out, "site_id":sps['site_id'], 
                                                        "species":sps['species_code'], "occurrence":sps['occurrence']})])    
    # if box is clean, save cleaned clip
    if(joint_boxes_data['overlaps_with_other_species'].iloc[i] == False):
        sig2 = pad(sig, x1-start, x2-start, snr=10, target_len=len(sig), sr=48000) # time- and frequency cropped version
        sig2 = butter_bandpass_filter(sig2, [joint_boxes_data['area_y1'].iloc[i],joint_boxes_data['area_y2'].iloc[i]], 48000, 12) 
        sig2 = librosa.resample(sig2, orig_sr=48000, target_sr=24000)
        file_out = str(joint_boxes_data['recording_id'].iloc[i]) + '_' + str(round(start,1)) + '_' + str(running_idx) + '_cleaned.wav'
        sf.write(path_to_birdsound_library + file_out, sig2, 24000)
        running_idx = running_idx+1  
        bsg_metadata = pd.concat([bsg_metadata, pd.DataFrame({"file_name":file_out, "site_id":[joint_boxes_data['site_id'].iloc[i]], 
                                                        "cleaned":True, "no_other_species":True, "noise":False})])
        bsg_labels = pd.concat([bsg_labels, pd.DataFrame({"file_name":file_out, "site_id":[joint_boxes_data['site_id'].iloc[i]], 
                                                        "species":joint_boxes_data['species_code'].iloc[i], "occurrence":[joint_boxes_data['occurrence'].iloc[i]]})])
    if i % 100 == 0:
        print(f"    Processed {i}/{len(joint_boxes_data)} ({round(i/len(joint_boxes_data)*100, 2)} %) clips   ", end = "\r")
            
bsg_metadata.reset_index(drop=True, inplace=True)
bsg_metadata.to_csv(bsg_metadata_out+ 'BSG_soundscapes.csv', index=False)
                                                      
bsg_labels.reset_index(drop=True, inplace=True)
bsg_labels.to_csv(bsg_metadata_out+ 'BSG_labels.csv', index=False)
                                                      
print("Clips complete.                                                               ")

# check metadata and files
audio_clips = os.listdir(path_to_birdsound_library) 
print(f"Clips saved: {len(audio_clips)}")
print(f"Files in metadata: {len(bsg_metadata)}")
print("Difference between sets:")
print(list(set(bsg_metadata['file_name']) - set(audio_clips)))
print(list(set(audio_clips) - set(bsg_metadata['file_name'])))