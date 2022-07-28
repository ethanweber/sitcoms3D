# Metadata

To download the relevant files you can run the following script (enabled by the [gdown pip package](https://github.com/wkentaro/gdown)). All unzipped folders should live in the [data/](data/) folder after running the script.

```bash
pip install gdown
python download_data.py
```

Our data uses a convention of `<sitcom>-<location>` for seven sitcoms, which can be seen in the NeRF-W panoramic images below:

1. TBBT-big_living_room
2. Frasier-apartment
3. ELR-apartment
4. Friends-monica_apartment
5. TAAHM-kitchen
6. Seinfeld-jerry_living_room
7. HIMYM-red_apartment

![Panoramic NeRF-W renderings of the sitcom locations](/media/nerf_rooms@4x.png)

#### Environments: sparse reconstruction and NeRF-W data

This section concerns the data of the COLMAP sparse reconstructions and images used to train NeRF-W. These live in the folder `sparse_reconstruction_and_nerf_data/<sitcom>-<location>/`. Among others, you can find:

- `cameras.json` is a processed version of the `colmap/` sparse reconstruction and the `threejs.json` file. The keys include `{"bbox", "point_cloud_transform", "scale_factor", "frames"}`. The `"frames"` are processed camera poses (NeRF cameras) where `NeRF cameras = (point_cloud_transform @ COLMAP cameras) / scale_factor`. See [notebooks/data_demo.ipynb](notebooks/data_demo.ipynb) for an explanation.

- `panoptic_classes.json` and `segmentations/` have been created with panoptic segmentation from [detectron2](https://github.com/facebookresearch/detectron2). `panoptic_classes` are ordered and correspond to the pixel values inside `segmentations` for the `stuff` and `thing` classes, respectively. _We only use the `thing` `person` class in our work. However, we are including all information to encourage future work on incoorportating semantics into the scene + human reconstruction pipeline. For example, [Semantic-NeRF](https://github.com/Harry-Zhi/semantic_nerf) could be used with this data._

- `threejs.json` is a file that can be visualized with this online three.js editor [https://threejs.org/editor/](https://threejs.org/editor/). This file will show the COLMAP sparse point cloud and the bounding box used to define regions where the NeRF-W field is valid. `point_cloud_transform` was created in this interface, where we rotated and translated the point cloud in the three.js editor to obtain an axis-aligned bounding box (AABB). This allowed for efficient ray near/far bounds sampling when using with NeRF.

If you are interested in all the images, and you have access to the videos for the episodes, you can also take a look at the `process_videos.py` script.

<hr>

#### Humans: SMPL parameters and human-pair data

Here we give an overview of the contents in each of the files relevant for human reconstruction.

```text
human_data.zip
human_data/<sitcom>-<location>.json
# Contains the "openpose_keypoints" for all humans and the "smpl" parameters where they exist.
# The "smpl" parameters only exist when we could use our method ("calibrated multi-shot") to optimize across the shot change.
{
  "<image_name>": [
    { # human_idx_0 for this image_name
      "openpose_keypoints": ...,
      "smpl": {
        "camera_translation": ...,
        "betas": ...,
        "global_orient": ...,
        "body_pose": ...,
        "colmap_rescale": ...
      }
    },
    { # human_idx_1 for this image_name
      ...
    }
  ]
}

human_pairs.zip
human_pairs/<sitcom>-<location>.json
# The image idx, human_idx pairs for which humans were optimized together after solving the Hungarian matching problem.
# This is where our method ("calibrated multi-shot") was used to create the "smpl" parameters as described above.
[
  [image_name_a, human_idx_a, image_name_b, human_idx_b],
  ...
]
```

#### 2D DISK features

Registering new images into the same coordinate frame as our COLMAP reconstructions requires having 2D DISK features to match to. The ZIP files have filenames `<sitcom>-<location>-disk.zip` and can be downloaded by uncommenting line 29 in `download_data.py`. These files are quite large, but you can unzip them and put their contents in the `data/sparse_reconstruction_and_nerf_data/<sitcom>-<location>/` folder.
