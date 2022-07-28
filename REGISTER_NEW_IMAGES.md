# Register new images

This document explains how to register new images to our existing sparse COLMAP reconstructions. This could be useful if you want to download images from YouTube, for instance, and obtain camera poses.

Currently, due to the large size of the related files, we only provide the keypoint correspondences for Monica's apartment. If you are interested in the correspondences for the other environments, please contact us via email.

First, you need to prepare the images you want to register in the existing reconstruction. This includes a folder with the raw images and a folder where you store the masks where you mask out the dynamic objects (humans). This can be easily done with a Mask-RCNN network.

```
|- data/new_frames/
  |- images/
  |- masks/
```

Then you can run our customized ```disk``` code to establish correspondences with the existing images.

```
cd external/disk
export DATA_DIRECTORY=../../data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment
export OUTPUT_DIRECTORY=../../data/new_frames
python detect.py --height 720 --width 1280 --n 2048 new_images/h5 new_images/images --image-extension jpg
python match_to_existing_keypoints.py --rt 0.95 --save-threshold 10 $DATA_DIRECTORY/h5 $OUTPUT_DIRECTORY/h5
python colmap/h5_to_db_update_existing.py --existing-database-path $DATA_DIRECTORY/database.db 
                                          --database-path $OUTPUT_DIRECTORY/database.db --camera-model simple-pinhole --image-extension .jpg
                                          $DATA_DIRECTORY/h5 $OUTPUT_DIRECTORY/h5 $OUTPUT_DIRECTORY/images $DATA_DIRECTORY/masks $OUTPUT_DIRECTORY/masks
```

Finally, you can run the corresponding COLMAP command to register the images in the existing reconsruction. This will return the camera extrinsics and extrinsics in the colmap format.

```
mkdir $OUTPUT_DIRECTORY/cameras
colmap image_registrator --database_path $OUTPUT_DIRECTORY/database.db --input_path $DATA_DIRECTORY/colmap --output_path $OUTPUT_DIRECTORY/cameras
```
