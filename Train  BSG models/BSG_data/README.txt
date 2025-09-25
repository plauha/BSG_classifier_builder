CONTENTS

FOLDER bsg_identification results:
  Results from BSG Annotate soundscapes -section, downloaded from BSG.
    annotations:
      recording-level annotations in user-specific tsv-files. Contains annotation_id, recording_id and background information of the recording (if recording contains human speech, if recording contains unidentified bird vocalizations, if recording does not contain any birds, if recording contains species not included in BSG species list, which species is that, if all vocalizations of the recording have been marked, x and y coordinates of a non-bird signal (x in seconds, y in Hz), time of creation and editing).
    species_annotation_boxes:
      vocalization-level annotations in user-specific tsv-files. Contains species_annotation_id, x-coordinates (start and stop) of box in seconds, y-coordinates (low, high) of box in Hz, if box overlaps with vocalizations from other species.
    species_annotations:
      vocalization-level annotations in user-specific tsv-files. Maps species_annotation_id to recording_id and annotation_id, additionally contains species name and certainty of occurrence (1=certain, 2=uncertain)
    recordings:
      maps recording_id to site_id and name of audio file

FOLDER Original_metadata_for_BSG_templates_audio_files:
  Original metadata from Macaulay Library and xeno-canto for the recordings from which template candidates of BSG, validate species templates -section have been selected.

bsg_identification_users.tsv:
  Background information for BSG users, downloaded from BSG. Contains user_ids and self-reported birdwatching activity level and skill levels for taxa of different continents 1 being the lowest and 4 the highest.
  MA.birdwatchingActivityLevelEnum1 -- I’m not a birdwatcher
  MA.birdwatchingActivityLevelEnum2 -- I watch birds occasionally
  MA.birdwatchingActivityLevelEnum3 -- I watch birds actively and regularly
  MA.birdwatchingActivityLevelEnum4 -- I am bird researcher or professional birdwatcher
  MA.birdSongRecognitionSkillLevelEnum1 -- I can hardly identify any bird species
  MA.birdSongRecognitionSkillLevelEnum2 -- I can identify some of the species (regionally)
  MA.birdSongRecognitionSkillLevelEnum3 -- I can identify majority of the species regionally
  MA.birdSongRecognitionSkillLevelEnum4 -- I can identify majority of the species of the continent

bsg_results.tsv:
  Results from BSG Validate species templates -section, downloaded from BSG. Contains species name, recording url, x-coordinates (start and stop) of template in seconds, y-coordinates (low, high) of template in Hz, name of audio file.  

BSG_template_files_metadata.csv:
  Combined metadata for all files that are included in BSG Validate species templates -section. Created based on files in Original_metadata_for_BSG_templates_audio_files.

mp3_20s.txt:
  Buffer and clip lengths for all clips included in BSG, annotate soundscapes -section. Must be updated by hand after each BSG-update.
 
