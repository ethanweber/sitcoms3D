"""
This script is used to download the data for the project.

We referenced this site when creating this script:
    https://discuss.dizzycoding.com/python-download-files-from-google-drive-using-url/
"""

import os
import zipfile
import gdown

data_filename_to_file_id = [
    ("./data/human_data.zip", "1zk12qxnZbReKqdvy-JYp1jQn2QA3oJzz"),
    ("./data/human_pairs.zip", "1927rnmJ3mWcjsj0afqawSto2IK2lLwmA"),
    ("./data/sparse_reconstruction_and_nerf_data.zip", "1RmwDUAp1T4RkwZg1S_7L9JAViWtBZfAG")
]

disk_data_filename_to_file_id = [
    ("./data/sparse_reconstruction_and_nerf_data/ELR-apartment-disk.zip", "1YcTwMv5PP0uqXYdN-Lp8k0WnXUdHBXrY"),
    ("./data/sparse_reconstruction_and_nerf_data/Frasier-apartment-disk.zip", "1vPqplDv5rrFu5LrxTHbET0FiePTGi8zE"),
    ("./data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment-disk.zip", "1yNpU4M44gIuWEaC5tSAOKy-A6KX8wTm8"),
    ("./data/sparse_reconstruction_and_nerf_data/HIMYM-red_apartment-disk.zip", "1m76i46YYlpDKvlx-Jn7qDDAvkOVVVmqF"),
    ("./data/sparse_reconstruction_and_nerf_data/Seinfeld-jerry_living_room-disk.zip", "1YFoV3Gd9asKsRarZvzRTYtri1755rBPx"),
    ("./data/sparse_reconstruction_and_nerf_data/TAAHM-kitchen-disk.zip", "1QpF8kVGfgkqm_cDY5sY_E82BT8VOWL38"),
    ("./data/sparse_reconstruction_and_nerf_data/TBBT-big_living_room-disk.zip", "175BjkpxMAOt75ZVIjYpArxWpO8uRexgJ")
]

# !!! Large files !!! Only uncomment if you need the disk correspondences. 
# data_filename_to_file_id += disk_data_filename_to_file_id

if __name__ == "__main__":

    for data_filename, file_id in data_filename_to_file_id:
        # Download the files
        print("Downloading {}...".format(data_filename))
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, data_filename, quiet=False)
        print("Downloaded {}".format(data_filename))

        # Unzip the files into folders
        print("Unzipping {}...".format(data_filename))
        with zipfile.ZipFile(data_filename, 'r') as zip_ref:
            zip_ref.extractall(data_filename.replace('.zip','').replace('-disk',''))
        print("Unzipped {}".format(data_filename))

        # Delete zip files
        os.remove(data_filename)
