# EPL445 Traffic Light Recognition

UCY EPL445 Project - Traffic light recognition using a YOLO model trained on the BSTLD dataset

# Dependencies

- [PyYAML](https://pyyaml.org)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

```sh
pip install ultralytics PyYAML
```

# Usage

## Label Processing

### Find all unique classes

```
python3 label-process.py find-all-classes <bstld_labels_file.yaml>
```

Finds all unique classes in a BSTLD-formatted label file and prints them.

### Convert to YOLO format

```
python3 label-process.py convert [options...] <output_path>
```

Coverts dataset from Bosch format to YOLO format.

`--train <path/to/train.yaml>`  
Training set labels

`--test <path/to/test.yaml>`  
Testing set labels

`--hardlink`  
Creates hard links in the converted dataset, which point to the images from the original dataset. This saves disk space and is significantly faster than copying. Both datasets MUST be on the same filesystem (drive or partition). This is the default behavior.

`--symlink-rel`  
Creates relative symbolic links in the converted dataset, which point to the images from the original dataset. This saves disk space and is significantly faster than copying. The two datasets can be on different filesystems and they can be moved together, as long as the relative position to each other remains unchanged.

`--symlink-abs`  
Creates absolute symbolic links in the converted dataset, which point to the images from the original dataset. This saves disk space and is significantly faster than copying. The two datasets can be on different filesystems and the converted dataset can be freely moved, but the original dataset CANNOT be moved.

`--copy`  
Copies images to the converted dataset, instead of creating links. This has none of the limitations of linking, but it's slower and it will use significantly more disk space.

`--merge-color-variants`  
Merge direction variants of color classes into one. For example, 'Red', 'RedLeft', 'RedRight', etc. will be merged to 'Red'.

`--start-id <id>`  
When choosing class IDs, start counting from this one. Useful for adding new classes onto a pre-trained model.

`--replace`  
Replace existing output destination without asking.