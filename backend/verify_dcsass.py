import os
import pandas as pd

# ==========================================
# DCSASS DATASET VERIFICATION
# ==========================================

DATASET_DIR = "../data/datasets"
LABELS_DIR = os.path.join(DATASET_DIR, "Labels")

CLASSES = [
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism"
]

total_rows = 0
total_found = 0
total_missing = 0
total_zero = 0
total_one = 0

print("=" * 70)
print("DCSASS DATASET VERIFICATION")
print("=" * 70)

for activity in CLASSES:

    csv_path = os.path.join(LABELS_DIR, activity + ".csv")

    print(f"\n[{activity}]")

    if not os.path.exists(csv_path):
        print("  ERROR: CSV file not found")
        continue

    # DCSASS CSV has no header
    df = pd.read_csv(csv_path, header=None)

    if df.shape[1] < 3:
        print("  ERROR: CSV does not contain 3 columns")
        continue

    activity_rows = len(df)
    activity_found = 0
    activity_missing = 0

    label_counts = df.iloc[:, 2].value_counts()

    zero_count = int(label_counts.get(0, 0))
    one_count = int(label_counts.get(1, 0))

    total_rows += activity_rows
    total_zero += zero_count
    total_one += one_count

    for _, row in df.iterrows():

        clip_id = str(row.iloc[0]).strip()

        # Example:
        # Fighting002_x264_3
        #
        # Actual file:
        # Fighting/Fighting002_x264.mp4/Fighting002_x264_3.mp4

        parts = clip_id.rsplit("_", 1)

        if len(parts) != 2:
            activity_missing += 1
            continue

        parent_video = parts[0] + ".mp4"
        clip_file = clip_id + ".mp4"

        clip_path = os.path.join(
            DATASET_DIR,
            activity,
            parent_video,
            clip_file
        )

        if os.path.isfile(clip_path):
            activity_found += 1
        else:
            activity_missing += 1

    total_found += activity_found
    total_missing += activity_missing

    print(f"  CSV samples       : {activity_rows}")
    print(f"  Clips found       : {activity_found}")
    print(f"  Clips missing     : {activity_missing}")
    print(f"  Label 0           : {zero_count}")
    print(f"  Label 1           : {one_count}")

print("\n" + "=" * 70)
print("OVERALL RESULT")
print("=" * 70)

print(f"Total CSV samples   : {total_rows}")
print(f"Total clips found   : {total_found}")
print(f"Total clips missing : {total_missing}")
print(f"Total label 0       : {total_zero}")
print(f"Total label 1       : {total_one}")

print("\nVerification completed.")