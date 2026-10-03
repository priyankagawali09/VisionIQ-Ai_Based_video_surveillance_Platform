import os
import random
import pandas as pd

# ============================================================
# DCSASS METADATA PREPARATION
# ============================================================

DATASET_DIR = "../data/datasets"
LABELS_DIR = os.path.join(DATASET_DIR, "Labels")

OUTPUT_DIR = "../data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dcsass_metadata.csv")

# Currently usable classes
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
    "Shoplifting"
]

# Reproducible split
random.seed(42)

records = []

print("=" * 70)
print("PREPARING DCSASS METADATA")
print("=" * 70)

# ============================================================
# READ CSV + FIND MATCHING VIDEO CLIPS
# ============================================================

for activity in CLASSES:

    csv_path = os.path.join(
        LABELS_DIR,
        activity + ".csv"
    )

    if not os.path.isfile(csv_path):
        print(f"\n{activity}: CSV NOT FOUND")
        continue

    df = pd.read_csv(
        csv_path,
        header=None
    )

    found = 0
    missing = 0
    invalid = 0

    # Process every row in this activity CSV
    for _, row in df.iterrows():

        # Skip incomplete rows
        if (
            pd.isna(row.iloc[0])
            or pd.isna(row.iloc[1])
            or pd.isna(row.iloc[2])
        ):
            invalid += 1
            continue

        clip_id = str(row.iloc[0]).strip()
        activity_name = str(row.iloc[1]).strip()

        # Convert suspicious label safely
        try:
            suspicious = int(row.iloc[2])
        except (ValueError, TypeError):
            invalid += 1
            continue

        # Only accept 0 or 1
        if suspicious not in [0, 1]:
            invalid += 1
            continue

        # Example:
        # Fighting002_x264_3
        #
        # Parent video:
        # Fighting002_x264.mp4
        #
        # Clip:
        # Fighting002_x264_3.mp4

        parts = clip_id.rsplit("_", 1)

        if len(parts) != 2:
            missing += 1
            continue

        parent_video = parts[0] + ".mp4"
        clip_file = clip_id + ".mp4"

        clip_path = os.path.join(
            DATASET_DIR,
            activity,
            parent_video,
            clip_file
        )

        # Check actual clip
        if not os.path.isfile(clip_path):
            missing += 1
            continue

        records.append({
            "clip_path": os.path.normpath(clip_path),
            "video_id": parent_video,
            "activity": activity_name,
            "suspicious": suspicious
        })

        found += 1

    print(
        f"{activity:<16} "
        f"Found: {found:<5} "
        f"Missing: {missing:<5} "
        f"Invalid: {invalid}"
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)

if df.empty:
    print("\nERROR: No valid clips found.")
    exit()

print("\n" + "=" * 70)
print("CREATING VIDEO-LEVEL SPLIT")
print("=" * 70)


# ============================================================
# GET UNIQUE PARENT VIDEOS
# ============================================================

videos = df[
    ["activity", "video_id"]
].drop_duplicates()

train_videos = []
val_videos = []
test_videos = []


# ============================================================
# SPLIT BY ORIGINAL VIDEO
# ============================================================

for activity in sorted(videos["activity"].unique()):

    activity_videos = videos[
        videos["activity"] == activity
    ]["video_id"].tolist()

    random.shuffle(activity_videos)

    total = len(activity_videos)

    if total < 3:

        print(
            f"{activity:<16} "
            f"Videos: {total:<4} "
            f"All → train"
        )

        train_videos.extend(
            [(activity, v) for v in activity_videos]
        )

        continue

    train_count = max(
        1,
        int(total * 0.70)
    )

    val_count = max(
        1,
        int(total * 0.15)
    )

    train_part = activity_videos[:train_count]

    val_part = activity_videos[
        train_count:
        train_count + val_count
    ]

    test_part = activity_videos[
        train_count + val_count:
    ]

    train_videos.extend(
        [(activity, v) for v in train_part]
    )

    val_videos.extend(
        [(activity, v) for v in val_part]
    )

    test_videos.extend(
        [(activity, v) for v in test_part]
    )

    print(
        f"{activity:<16} "
        f"Videos: {total:<4} "
        f"Train: {len(train_part):<3} "
        f"Val: {len(val_part):<3} "
        f"Test: {len(test_part):<3}"
    )


# ============================================================
# CREATE LOOKUP SETS
# ============================================================

train_set = set(train_videos)
val_set = set(val_videos)
test_set = set(test_videos)


def get_split(row):

    key = (
        row["activity"],
        row["video_id"]
    )

    if key in train_set:
        return "train"

    if key in val_set:
        return "validation"

    if key in test_set:
        return "test"

    return "unknown"


df["split"] = df.apply(
    get_split,
    axis=1
)


# ============================================================
# REMOVE UNKNOWN
# ============================================================

df = df[
    df["split"] != "unknown"
].copy()


# ============================================================
# SAVE METADATA
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

print(f"Total clips : {len(df)}")

print("\nSplit:")
print(
    df["split"].value_counts()
)

print("\nActivity:")
print(
    df["activity"].value_counts()
)

print("\nActivity + Split:")
print(
    pd.crosstab(
        df["activity"],
        df["split"]
    )
)

print("\nSuspicious labels:")
print(
    df["suspicious"].value_counts()
)

print("\nSaved to:")
print(
    os.path.abspath(OUTPUT_FILE)
)

print("\nMetadata preparation completed successfully.")