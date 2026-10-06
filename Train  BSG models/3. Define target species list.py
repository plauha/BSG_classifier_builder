import pandas as pd
import numpy as np
import os

# Create initial species list based on BSG annotations

model_path = 'models/Spain_Portugal/' # name of the model
os.mkdir(model_path)

# define the BSG sites that will be included in the model
site_list = [622, 718, 655, 750, 643] #spain & portugal

np.save(model_path + 'site_list.npy', site_list)

annotations = pd.read_csv("BSG_results/10s_annotations.csv")
bsg_species = pd.read_csv("BSG_results/bsg_species.csv")

annotations = annotations[annotations['site_id'].isin(site_list)]
species_list = np.unique(annotations['species_code'])
species_list = np.delete(species_list, np.where(species_list == 'other'))
species_list = bsg_species[bsg_species['species_code'].isin(species_list)]
species_list.to_csv(model_path + 'initial_sp_list.csv', index=False) # save initial list for manual check 
species_list

# The species list should be manually checked in case of erroneous annotations. Additional species can also be included, if necessary.

# Check and save final species list

species_list = pd.read_csv(model_path + 'initial_species_list.csv')
# Check if information of species names exists for all species
# print species with missing names
species_list.loc[species_list['xc_scientific_name'].isna()]

# Check if all species included in the species list still exist in xeno-canto archives (due to eg. taxonomic changes, changes in species names etc.)

# print species_list with non-matching species names
# xeno-canto species list updated from https://xeno-canto.org/collection/species/all on 11.6.2024
xc_species = pd.read_csv("BSG_results/xc_species.csv")
species_list.loc[~species_list['xc_scientific_name'].isin(xc_species['Scientific name'])]

# if needed, manually fix the updated names to  bsg_results/bsg_species.csv and to the current species list model_path/species_list.csv