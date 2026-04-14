import os
import random
import shutil
from pathlib import Path
from collections import defaultdict
import hashlib

def make_id(path):
    return hashlib.md5(str(path).encode()).hexdigest()[:12]


def merge_and_resplit_dataset(
    input_dir,
    output_dir,
    split_ratio=0.8,
    seed=42,
    link_type="hardlink"  # or "copy"
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Collect all image-label pairs
    all_files = []

    for split in ["train", "val"]:
        img_dir = input_dir / "images" / split
        lbl_dir = input_dir / "labels" / split

        for img_path in img_dir.glob("*"):
            lbl_path = lbl_dir / (img_path.stem + ".txt")

            if lbl_path.exists():
                all_files.append((img_path, lbl_path))

    print(f"Total samples found: {len(all_files)}")

    # Shuffle
    random.seed(seed)
    random.shuffle(all_files)

    # Split
    split_idx = int(len(all_files) * split_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Create output structure
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Helper for transfer
    def transfer(src, dst):
        if link_type == "hardlink":
            os.link(src, dst)
        elif link_type == "symlink":
            os.symlink(src, dst)
        else:
            shutil.copy(src, dst)

    # Write files
    for split_name, files in [("train", train_files), ("val", val_files)]:
        for img_path, lbl_path in files:
            dst_img = output_dir / "images" / split_name / img_path.name
            dst_lbl = output_dir / "labels" / split_name / lbl_path.name

            transfer(img_path, dst_img)
            transfer(lbl_path, dst_lbl)

    # Copy config.yaml
    shutil.copy(input_dir / "config.yaml", output_dir / "config.yaml")

    print("Done!")



def stratified_split_dataset(
    input_dir,
    output_dir,
    split_ratio=0.8,
    seed=42,
    link_type="hardlink"
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    random.seed(seed)

    # -----------------------------
    # 1. Collect all image-label pairs
    # -----------------------------
    all_samples = []

    for split in ["train", "val"]:
        img_dir = input_dir / "images" / split
        lbl_dir = input_dir / "labels" / split

        if not img_dir.exists():
            continue

        for img_path in img_dir.glob("*"):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                all_samples.append((img_path, lbl_path))

    print(f"Total samples: {len(all_samples)}")

    # -----------------------------
    # 2. Extract class sets per image
    # -----------------------------
    samples_with_classes = []
    class_counts = defaultdict(int)

    for img, lbl in all_samples:
        classes = set()

        with open(lbl, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue
                cls = int(parts[0])
                classes.add(cls)

        samples_with_classes.append((img, lbl, classes))

        for c in classes:
            class_counts[c] += 1

    # -----------------------------
    # 3. Separate null frames and annotated frames
    # -----------------------------
    non_null = [(img, lbl, classes) for img, lbl, classes in samples_with_classes if classes]
    null_samples = [(img, lbl) for img, lbl, classes in samples_with_classes if not classes]

    print(f"Annotated: {len(non_null)}, Null frames: {len(null_samples)}")

    # Split nulls randomly at the same ratio
    random.shuffle(null_samples)
    null_split_idx = int(len(null_samples) * split_ratio)
    null_train = null_samples[:null_split_idx]
    null_val   = null_samples[null_split_idx:]

    # -----------------------------
    # 4. Sort: rare-class images first (annotated only)
    # -----------------------------
    non_null.sort(
        key=lambda x: sum(class_counts[c] for c in x[2])
    )

    # -----------------------------
    # 5. Greedy stratified split (annotated only)
    # -----------------------------
    train, val = [], []

    train_class_counts = defaultdict(int)
    val_class_counts = defaultdict(int)

    target_train_size = int(len(non_null) * split_ratio)

    for img, lbl, classes in non_null:
        train_score = 0.0
        val_score = 0.0

        for c in classes:
            train_score += train_class_counts[c] / (class_counts[c] + 1e-6)
            val_score += val_class_counts[c] / (class_counts[c] + 1e-6)

        if len(train) < target_train_size and train_score <= val_score:
            train.append((img, lbl))
            for c in classes:
                train_class_counts[c] += 1
        else:
            val.append((img, lbl))
            for c in classes:
                val_class_counts[c] += 1

    # Merge nulls into their respective splits
    train += null_train
    val   += null_val

    print(f"Train: {len(train)} ({len(null_train)} null), Val: {len(val)} ({len(null_val)} null)")

    # -----------------------------
    # 6. Create output structure
    # -----------------------------
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 7. File transfer helper
    # -----------------------------
    def transfer(src, dst):
        if link_type == "hardlink":
            os.link(src, dst)
        elif link_type == "symlink":
            os.symlink(src, dst)
        else:
            import shutil
            shutil.copy(src, dst)

    # -----------------------------
    # 8. Write dataset
    # -----------------------------
    def write_split(data, split_name):
        for img, lbl in data:
            uid = make_id(img)
            dst_img = output_dir / "images" / split_name / f"{uid}.png"
            dst_lbl = output_dir / "labels" / split_name / f"{uid}.txt"

            transfer(img, dst_img)
            transfer(lbl, dst_lbl)

    write_split(train, "train")
    write_split(val, "val")

    # -----------------------------
    # 9. Done
    # -----------------------------
    shutil.copy(input_dir / "config.yaml", output_dir / "config.yaml")
    print("Stratified split completed.")




##### Usage: python3 split_dataset.py --input=<input_folder> --output=<new_folder> 
### This should be ran AFTER the label-process.py script, with the output folder of the label-process.py to be the input of the split_dataset.py

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge and re-split YOLO dataset")

    parser.add_argument("--input", required=True, help="Input dataset directory")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--link-type", choices=["hardlink", "symlink", "copy"], default="hardlink")
    
    parser.add_argument("--mode", choices=["random", "balanced"], default="balanced")

    args = parser.parse_args()

    if args.mode == "random":
        merge_and_resplit_dataset(
            input_dir=args.input,
            output_dir=args.output,
            split_ratio=args.split_ratio,
            seed=args.seed,
            link_type=args.link_type
        )

    else:
        stratified_split_dataset(
            input_dir=args.input,
            output_dir=args.output,
            split_ratio=args.split_ratio,
            seed=args.seed,
            link_type=args.link_type
        )

