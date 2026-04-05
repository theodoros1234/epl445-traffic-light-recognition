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
./label-process.py find-all-classes <bstld_labels_file.yaml>
```

Finds all unique classes in a BSTLD-formatted label file and prints them.
