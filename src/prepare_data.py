import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = "/Users/simar/Documents/pneumonia_covid_detection/covid19-pneumonia-normal-chest-xraypa-dataset"
DATASET_DIR = os.path.join(BASE_DIR, "COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset")
CSV_PATH = os.path.join(BASE_DIR, "metadata.csv")
OUTPUT_DIR = "data"

for subset in ["train", "val", "test"]:
    subset_path = os.path.join(OUTPUT_DIR, subset)
    if os.path.exists(subset_path):
        shutil.rmtree(subset_path)


df = pd.read_csv(CSV_PATH)
print("Metadata preview:")
print(df.head())

class_map = {
    0: "normal",
    1: "covid",
    2: "pneumonia"
}
df['label'] = df['class'].map(class_map)


train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df['label'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
)

def copy_images(subset_df, subset_name):
    for _, row in subset_df.iterrows():
        rel_path = row['directory']   # e.g. covid/COVID19(308).jpg
        label = row['label']
        src_path = os.path.join(DATASET_DIR, rel_path)
        dest_dir = os.path.join(OUTPUT_DIR, subset_name, label)
        os.makedirs(dest_dir, exist_ok=True)
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_dir)
        else:
            # Skip missing files silently
            continue


copy_images(train_df, "train")
copy_images(val_df, "val")
copy_images(test_df, "test")

print(" Dataset organized into train/val/test folders successfully!")

# verification step: count images per class in each split
for subset in ["train", "val", "test"]:
    print(f"\n📊 Counts for {subset}:")
    subset_path = os.path.join(OUTPUT_DIR, subset)
    for label in class_map.values():
        label_path = os.path.join(subset_path, label)
        count = len(os.listdir(label_path)) if os.path.exists(label_path) else 0
        print(f"  {label}: {count} images")
