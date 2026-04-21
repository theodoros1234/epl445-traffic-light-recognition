#!/bin/python3

import os, sys, shutil, re
try:
  import yaml
except ModuleNotFoundError:
  print("""Missing required library: PyYAML

Try installing with:
pip install PyYAML""")
  exit()

# Gives statistics about a BSTLD-formatted dataset:
# unique classes, instances of each class, null frame count
# output can be 'print' or 'return'
def stats(labels, output):
  if type(labels) == str:     # file given
    with open(labels, 'r') as f:
      data = yaml.safe_load(f)
  elif type(labels) == list:  # parsed data given
    data = labels
  else:
    raise TypeError("'labels' must be a path string to load from, or a list of parsed data")
  classes = dict()
  null_frames = 0

  # scan data for classes
  for entry in data:
    for box in entry['boxes']:
      c = box['label']
      if c in classes:
        classes[c] += 1
      else:
        classes[c] = 1

    if len(entry['boxes']) == 0:
      null_frames += 1

  if output == 'print':
    if type(labels) == str:   # file path given, print that first
      print("Stats for %s:\n" % repr(labels))

    print("Classes:")
    for c in classes:
      print("% 5dx %s" % (classes[c], c))
    print("\nNull frames: %d of %d (%.1f%%)" % (null_frames, len(data), 100 * null_frames / len(data)))
  elif output == 'return':
    return {
      "classes": classes,
      "null_frames": null_frames,
      "total_frames": len(data)
    }
  else:
    raise ValueError("invalid output type %s" % repr(output))




import yaml
from pathlib import Path
from collections import defaultdict

def yolo_stats(dataset_dir, output='print', splits=("train", "val")):
    dataset_dir = Path(dataset_dir)
    config_path = dataset_dir / "config.yaml"
    class_names = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        names = config.get("names", {})
        if isinstance(names, list):
            class_names = {i: name for i, name in enumerate(names)}
        elif isinstance(names, dict):
            class_names = names

    classes = defaultdict(int)                          # class_id -> total instance count
    classes_per_split = defaultdict(lambda: defaultdict(int))  # split -> class_id -> count
    null_frames = defaultdict(int)
    total_frames = defaultdict(int)

    for split in splits:
        lbl_dir = dataset_dir / "labels" / split
        if not lbl_dir.exists():
            continue
        for lbl_path in lbl_dir.glob("*.txt"):
            total_frames[split] += 1
            lines = [l.strip() for l in lbl_path.read_text().splitlines() if l.strip()]
            if not lines:
                null_frames[split] += 1
                continue
            for line in lines:
                parts = line.split()
                if parts:
                    cls_id = int(parts[0])
                    classes[cls_id] += 1
                    classes_per_split[split][cls_id] += 1

    total = sum(total_frames.values())
    total_null = sum(null_frames.values())
    all_cls_ids = sorted(classes)

    if output == 'print':
        print(f"Stats for: {dataset_dir}\n")

        # Frames table
        print(f"{'Split':<10} {'Frames':>8} {'Null':>8} {'Null %':>8}")
        print("-" * 38)
        for split in splits:
            n = total_frames.get(split, 0)
            nf = null_frames.get(split, 0)
            pct = 100 * nf / n if n else 0
            print(f"{split:<10} {n:>8} {nf:>8} {pct:>7.1f}%")
        print("-" * 38)
        pct = 100 * total_null / total if total else 0
        print(f"{'TOTAL':<10} {total:>8} {total_null:>8} {pct:>7.1f}%")

        # Classes table with per-split columns
        print("\nClasses:")
        col_w = 10
        header = f"  {'ID':<6} {'Name':<20} {'TOTAL':>{col_w}}"
        for split in splits:
            header += f"  {split:>{col_w}}"
        print(header)
        print("  " + "-" * (28 + (col_w + 2) * (1 + len(splits))))
        for cls_id in all_cls_ids:
            name = class_names.get(cls_id, f"class_{cls_id}")
            row = f"  {cls_id:<6} {name:<20} {classes[cls_id]:>{col_w}}"
            for split in splits:
                row += f"  {classes_per_split[split].get(cls_id, 0):>{col_w}}"
            print(row)

    elif output == 'return':
        return {
            "classes": {
                cls_id: {
                    "name": class_names.get(cls_id, f"class_{cls_id}"),
                    "count": classes[cls_id],
                    "count_per_split": {
                        split: classes_per_split[split].get(cls_id, 0)
                        for split in splits
                    },
                }
                for cls_id in all_cls_ids
            },
            "null_frames": dict(null_frames),
            "total_frames": dict(total_frames),
            "null_frames_total": total_null,
            "total_frames_total": total,
        }
    else:
        raise ValueError("invalid output type %s" % repr(output))



# Converts a BSTLD formatted dataset into the YOLO format
# TODO: explain parameters
def convert_to_yolo(output, train_labels = None, test_labels = None, image_transfer = 'hardlink', merge_color_variants = False, start_id = 0, replace = False):
  WIDTH, HEIGHT = 1280, 720   # image resolution for label conversion

  # check param validity
  if not image_transfer in ['hardlink', 'symlink-rel', 'symlink-abs', 'copy']:
    raise ValueError("'image_transfer' must be 'hardlink', 'symlink-rel', 'symlink-abs' or 'copy'")

  output = os.path.normpath(output)
  # print("output:", output)
  # check if output path already exists and ask user whether to replace it
  if os.path.exists(output):
    if replace:
      answer = 'y'
    else:
      answer = input("%s already exists. Would you like to replace it? [y/N] " % repr(output)).strip()

    if answer == 'y' or answer == 'Y':
      # delete output path
      if os.path.isdir(output):
        shutil.rmtree(output)
      else:
        os.remove(output)
    else:
      print("Aborting")
      return

  # create output dir
  os.mkdir(output)

  # define paths and options for each given subset (training, testing)
  subsets = []
  if train_labels != None:
    labels = os.path.normpath(train_labels)
    subsets.append({
      "type": "train",
      "labels": labels,
      "labels_file": open(labels, 'r'), # opened here to catch I/O errors sooner
    })

  if test_labels != None:
    labels = os.path.normpath(test_labels)
    subsets.append({
      "type": "val",
      "labels": labels,
      "labels_file": open(labels, 'r'), # opened here to catch I/O errors sooner
    })

  # make sure at least something was given
  if len(subsets) == 0:
    print("Nothing to do.")
    return

  classes = set()
  class_ids = dict()
  classes_for_config = dict()

  for subset in subsets:
    # import label data
    print("Importing data from %s" % repr(subset['labels']))
    data = yaml.safe_load(subset['labels_file'])
    subset['data'] = data
    # close original label file
    subset['labels_file'].close()
    subset.pop('labels_file')
    # find classes in dataset
    classes.update(stats(data, 'return')['classes'])

  # assign IDs to classes
  print("Assigning class IDs")
  first_word_finder = re.compile(r"^[A-Za-z][a-z]*")
  current_id = start_id

  for c in classes:
    if merge_color_variants:
      # merge variants (e.g. GreenLeft, GreenRight, etc.) into one (e.g. Green)

      # extract 'first word' in class name by looking at upper/lowercase letters
      match = first_word_finder.match(c)
      if match == None:
        # fallback if regex match fails
        c_prefix = c
      else:
        c_prefix = match.group()

      if not c_prefix in class_ids:
        # prefix doesn't have an ID, create it now
        class_ids[c_prefix] = current_id
        classes_for_config[current_id] = c_prefix
        current_id += 1

      # reuse prefix's ID to merge them
      class_ids[c] = class_ids[c_prefix]
    else:
      # merging not wanted, just assign IDs normally
      class_ids[c] = current_id
      classes_for_config[current_id] = c
      current_id += 1

  # for c in class_ids:
  #   print("%s: %d" % (c, class_ids[c]))

  # create yolo config
  print("Creating dataset config file")
  config = {"names": classes_for_config}
  # paths to subset dirs
  for subset in subsets:
    config[subset['type']] = os.path.join("images", subset['type'])
  # save to file
  with open(os.path.join(output, "config.yaml"), 'w') as f:
    f.write(yaml.safe_dump(config))

  # convert frame annotations
  frames_processed = 0
  frames_total = sum([len(subset['data']) for subset in subsets])
  for subset in subsets:
    source_dir = os.path.dirname(subset['labels'])
    target_image_dir = os.path.join(output, "images", subset['type'])
    target_label_dir = os.path.join(output, "labels", subset['type'])

    # create target directories
    os.makedirs(target_image_dir)
    os.makedirs(target_label_dir)

    # go through every frame in subset
    for frame in subset['data']:
      # show status (using ANSI escape codes to reuse same line)
      print("\r\033[KConverting frame annotations: %d/%d" % (frames_processed, frames_total), end='')

      # fix path issue in BSTLD's test.yaml file
      path = frame['path'].replace('/net/pal-soc1.us.bosch.com/ifs/data/Shared_Exports/deep_learning_data/traffic_lights/university_run1/', './rgb/test/')
      # extract filename
      filename = os.path.basename(path)
      filename_noext = filename[:filename.rfind('.')]

      # write annotation file
      with open(os.path.join(target_label_dir, filename_noext + ".txt"), 'w') as f:
        for box in frame['boxes']:
          # transform bounding box coordinates to YOLO format:
          # value range: 0 to 1
          # params: class_id, x_center, y_center, width, height
          x_min = min(max(box['x_min'], 0), WIDTH)
          x_max = min(max(box['x_max'], 0), WIDTH)
          y_min = min(max(box['y_min'], 0), HEIGHT)
          y_max = min(max(box['y_max'], 0), HEIGHT)

          class_id = class_ids[box['label']]
          x_center = (x_max + x_min) / (WIDTH * 2)
          y_center = (y_max + y_min) / (HEIGHT * 2)
          width = (x_max - x_min) / WIDTH
          height = (y_max - y_min) / HEIGHT
          f.write("%d %f %f %f %f\n" % (class_id, x_center, y_center, width, height))

      # transfer image
      src = os.path.join(source_dir, path)
      dst = os.path.join(target_image_dir, filename)

      if image_transfer == 'hardlink':
        # create a hard link to the original
        os.link(src, dst)
      elif image_transfer == 'symlink-rel':
        # create a relative symlink to the original
        symlink = os.path.relpath(src, target_image_dir)
        os.symlink(symlink, dst)
      elif image_transfer == 'symlink-abs':
        # create an absolute symlink to the original
        symlink = os.path.abspath(src)
        os.symlink(symlink, dst)
      elif image_transfer == 'copy':
        # just copy it over
        shutil.copy(src, dst)

      frames_processed += 1
  print("\r\033[KConverting frame annotations: DONE")

  # reminder
  if start_id != 0:
    print("\nDon't forget to add your old classes to the config file!")


# Extracts wanted command line options
def extract_cmd_args(action, args, wanted_options, other_count = None):
  extracted = dict()
  other = list()

  # go through args
  i = 0
  while i < len(args):
    arg = args[i]
    if arg.startswith('-'):                     # is an option
      if arg in wanted_options:                 # is an option we want
        if wanted_options[arg]['gets_param']:   # option gets param
          if i + 1 < len(args):
            extracted[arg] = args[i + 1]
            i += 1
          else:                                 # missing param
            print("Missing parameter for option %s.\n" % repr(arg))
            usage(action)
            return
        else:                                   # option doesn't get param
          extracted[arg] = True
      else:                                     # is NOT an option we want
        print("Unknown option %s.\n" % repr(arg))
        usage(action)
        return

    else: # not an option
      other.append(arg)

    i += 1

  # make sure required options were included, and set default values for the others
  for option_name in wanted_options:
    option = wanted_options[option_name]
    if not option_name in extracted:
      if option['required']:
        print("Missing required option %s\n" % repr(option_name))
        usage(action)
        return
      else:
        extracted[option_name] = None if option['gets_param'] else False

  # make sure other args count is correct (if needed)
  if other_count != None:
    if other_count < len(other):
      print("Too many arguments.\n")
      usage(action)
      return
    elif other_count > len(other):
      print("Too few arguments.\n")
      usage(action)
      return

  return (extracted, other)


# Prints command line usage for all actions or for a specific action
def usage(action=None):
  known_actions = {
    'stats': """python3 %s stats <bstld_labels_file.yaml>
Prints the following statistics about a BSTLD-formatted labels file:
- unique classes
- instances of each class
- null frame count and percentage
""",
    'convert': """python3 %s convert [options...] <output_path>
Coverts dataset from Bosch format to YOLO format.

  --train <path/to/train.yaml>
    Training set labels

  --test <path/to/test.yaml>
    Testing set labels

  --hardlink
    Creates hard links in the converted dataset, which point to the images from
    the original dataset. This saves disk space and is significantly faster than
    copying. Both datasets MUST be on the same filesystem (drive or partition).
    This is the default behavior.

  --symlink-rel
    Creates relative symbolic links in the converted dataset, which point to the
    images from the original dataset. This saves disk space and is significantly
    faster than copying. The two datasets can be on different filesystems and
    they can be moved together, as long as the relative position to each other
    remains unchanged.

  --symlink-abs
    Creates absolute symbolic links in the converted dataset, which point to the
    images from the original dataset. This saves disk space and is significantly
    faster than copying. The two datasets can be on different filesystems and
    the converted dataset can be freely moved, but the original dataset CANNOT
    be moved.

  --copy
    Copies images to the converted dataset, instead of creating links. This has
    none of the limitations of linking, but it's slower and it will use
    significantly more disk space.

  --merge-color-variants
    Merge direction variants of color classes into one.
    For example, 'Red', 'RedLeft', 'RedRight', etc. will be merged to 'Red'.

  --start-id <id>
    When choosing class IDs, start counting from this one. Useful for adding
    new classes onto a pre-trained model.

  --replace
    Replace existing output destination without asking.
"""
  }

  if action != None and not action in known_actions:
  # if user is trying to use an unknown action, print usage for all actions
    print("Unknown action %s\n" % repr(action))
    action = None

  if action == None:
    # no action specified (or it's unknown), print everything
    print("Usage: python3 %s <action> [params...]\n" % sys.argv[0])

    for a in known_actions:
      print(known_actions[a] % sys.argv[0])

  else:
    # print usage for specific action
    print("Usage for action '%s':" % action)
    print(known_actions[action] % sys.argv[0])


if __name__ == "__main__":
  # parse command line arguments
  argv_len = len(sys.argv)
  if argv_len == 0:
    # should never happen
    pass
  elif argv_len == 1:
    # show usage
    usage()
  else:
    # run action or show usage for bad params
    action = sys.argv[1]

    if action == 'stats':
      if argv_len == 3:
        labels_file = sys.argv[2]
        stats(labels_file, 'print')
      else:
        usage(action)
    
    elif action == 'yolo_stats':
      if argv_len == 3:
        labels_file = sys.argv[2]
        yolo_stats(labels_file, 'print')
      else:
        usage(action)
          
    elif action == 'convert':
      args = extract_cmd_args(action, sys.argv[2:], {
        "--train": {"gets_param": True, "required": False},
        "--test": {"gets_param": True, "required": False},
        "--hardlink": {"gets_param": False, "required": False},
        "--symlink-rel": {"gets_param": False, "required": False},
        "--symlink-abs": {"gets_param": False, "required": False},
        "--copy": {"gets_param": False, "required": False},
        "--merge-color-variants": {"gets_param": False, "required": False},
        "--start-id": {"gets_param": True, "required": False},
        "--replace": {"gets_param": False, "required": False}
      }, other_count=1)

      # exit if arg parse fails
      if args == None:
        exit()

      # parse start-id int
      try:
        start_id = int(args[0]['--start-id'])
      except TypeError:   # option not specified
        start_id = 0
      except ValueError:  # not a number
        print("Start ID must be an number.\n")
        usage(action)
        exit()

      # decide image transfer method
      image_transfer = 'hardlink'
      image_transfer_options = []
      for option in ('--hardlink', '--symlink-rel', '--symlink-abs', '--copy'):
        if args[0][option]:
          image_transfer_options.append(option)

      if len(image_transfer_options) > 1:   # conflicting options
        print("Conflicting options:", ', '.join(image_transfer_options), end='\n\n')
        usage(action)
        exit()
      elif len(image_transfer_options) == 1:
        image_transfer = image_transfer_options[0][2:] # strip out '--'

      convert_to_yolo(
        args[1][0],
        train_labels = args[0]['--train'],
        test_labels = args[0]['--test'],
        image_transfer = image_transfer,
        merge_color_variants = args[0]['--merge-color-variants'],
        start_id = start_id,
        replace = args[0]['--replace']
      )

    else:
      # unknown action
      usage(action)
