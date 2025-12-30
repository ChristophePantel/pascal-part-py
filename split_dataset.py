import os
import shutil
import random
from pathlib import Path

def split_dataset(
    dataset_dir, 
    output_dir,
    train_ratio=0.7, 
    val_ratio=0.2, 
    test_ratio=0.1, 
    seed=42
):
    random.seed(seed)

    image_dir = Path(dataset_dir) / "Images"
    annot_dir = Path(dataset_dir) / "YOLO_Annotations_Part"

    image_files = sorted([f for f in image_dir.glob("*") if f.is_file()])
    base_names = [f.stem for f in image_files]

    random.shuffle(base_names)

    total = len(base_names)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": base_names[:train_end],
        "val": base_names[train_end:val_end],
        "test": base_names[val_end:]
    }

    for split, names in splits.items():
        split_img_dir = Path(output_dir) / split / "images"
        split_annot_dir = Path(output_dir) / split / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_annot_dir.mkdir(parents=True, exist_ok=True)

        for name in names:
            img_file = image_dir / f"{name}.jpg"  # or .png, modify if needed
            annot_file = annot_dir / f"{name}.txt"  # or .txt, modify if needed

            if img_file.exists():
                shutil.copy2(img_file, split_img_dir / img_file.name)
            if annot_file.exists():
                shutil.copy2(annot_file, split_annot_dir / annot_file.name)

    print(f"Dataset split complete. {total} files split into:")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val:   {len(splits['val'])}")
    print(f"  Test:  {len(splits['test'])}")

# Usage
split_dataset(
    dataset_dir="/data/christophe/hierarchical/OriginalPascalPart",
    output_dir="/data/christophe/hierarchical/OriginalPascalPart/YOLO",
    train_ratio=0.8,
    val_ratio=0.2,
    test_ratio=0.0
)