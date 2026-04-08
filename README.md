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

`--copy`  
Copy images to converted dataset, instead of creating links.

`--merge-color-variants`  
Merge direction variants of color classes into one. For example, 'Red', 'RedLeft', 'RedRight', etc. will be merged to 'Red'.

`--start-id <id>`  
When choosing class IDs, start counting from this one. Useful for adding new classes onto a pre-trained model.

`--replace`  
Replace existing output destination without asking.