# EPL445 Traffic Light Recognition

UCY EPL445 Project - Traffic light recognition using a YOLO model trained on the [Bosch Small Traffic Lights Dataset](https://zenodo.org/records/12706046)

# Dependencies

- [PyYAML](https://pyyaml.org)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Streamlit (Optional)](https://streamlit.io/)

```sh
pip install ultralytics PyYAML streamlit
```

# Usage

## GUI (`gui.py`)

A Streamlit-based interface for the dataset pipeline tools.

##### Run with:

```bash
pip install streamlit pandas
streamlit run gui.py
```

The sidebar selects between five sections:

##### BSTLD Stats
Shows class counts and null frame statistics for a BSTLD-formatted `.yaml` labels file.

##### YOLO Stats
Shows a per-split breakdown of class instance counts and null frames for a YOLO-formatted dataset directory.

##### Convert BSTLD → YOLO
Converts a Bosch-format dataset to YOLO format, with options for image transfer method, color variant merging, and class ID offset.

##### Re-split Dataset
Merges and re-splits an existing YOLO dataset, either randomly or with stratified balancing to preserve class distribution across splits.

##### View Results
A lightweight file browser for navigating dataset directories. Displays `.png` / `.jpg` images inline and opens `.csv` files as interactive tables. Includes a breadcrumb trail for easy navigation.

---

## Label Processing

### Print BSTLD Stats

```
python3 label-process.py stats <bstld_labels_file.yaml>
```


### Print YOLO Stats

```
python3 label-process.py yolo-stats <bstld_labels_file.yaml>
```

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

## Inference

### Preview

```
yolo predict model=/path/to/model.pt source=/path/to/image_or_video.mp4 save=False show
```

Runs inference on a chosen image or video clip using the selected model. Output is shown in a window as it's being processed.

### Save to file

```
yolo predict model=/path/to/model.pt source=/path/to/image_or_video.mp4
```

Runs inference on a chosen image or video clip using the selected model. Output is saved to a file in your Ultralytics runs directory.
