from google_drive_downloader import GoogleDriveDownloader as gdd
from pathlib import Path
from tqdm import tqdm
import pandas as pd

""" 
This script downloads from Google Drive the Real World Masked Face Dataset.

Folders labelled as mask images are labelled with a 1, non-mask images are
labelled with a 0.

It is then appended to the Dataframe maskDF and saved into a pickle within
the data folder.
"""

# Link where the Dataset is hosted
# https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset
dataPath = Path('data/mask.zip')
gdd.download_file_from_google_drive(file_id='1UlOk6EtiaXTHylRUx2mySgvJX9ycoeBp',
                                    dest_path=str(dataPath), 
                                    unzip=True)
# Remove zip file
dataPath.unlink()

dataPath = Path('data/self-built-masked-face-recognition-dataset')
maskPath = dataPath/'AFDB_masked_face_dataset'
nonMaskPath = dataPath/'AFDB_face_dataset'
maskDF = pd.DataFrame()

for directory in tqdm(list(maskPath.iterdir()), desc='mask photos'):
    for imgPath in directory.iterdir():
        maskDF = maskDF.append({
            'image': str(imgPath),
            'mask': 1
        }, ignore_index=True)

for directory in tqdm(list(nonMaskPath.iterdir()), desc='non mask photos'):
    for imgPath in directory.iterdir():
        maskDF = maskDF.append({
            'image': str(imgPath),
            'mask': 0
        }, ignore_index=True)

mask_pickle = 'data/mask_DF.pickle'
print(f'Saving to: {mask_pickle}')
maskDF.to_pickle(mask_pickle)