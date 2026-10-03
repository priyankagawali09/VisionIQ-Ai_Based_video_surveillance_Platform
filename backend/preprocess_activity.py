import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# DCSASS ACTIVITY RECOGNITION PREPROCESSING
# ============================================================


# ============================================================
# PROJECT ROOT
# ============================================================

# This file is located at:
#
# E:\CCTV\intelligent-surveillance\backend\preprocess_activity.py
#
# Therefore:
# __file__          -> backend\preprocess_activity.py
# dirname           -> backend
# dirname again     -> intelligent-surveillance
#
# So BASE_DIR becomes:
#
# E:\CCTV\intelligent-surveillance
#

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# DATASET / METADATA PATH
# ============================================================

METADATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "dcsass_metadata.csv"
)


# ============================================================
# OUTPUT PATH
# ============================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "activity_sequences"
)


# ============================================================
# DISPLAY PATHS FOR VERIFICATION
# ============================================================

print("=" * 70)
print("DCSASS ACTIVITY PREPROCESSING")
print("=" * 70)

print("\nPROJECT BASE DIRECTORY:")
print(BASE_DIR)

print("\nMETADATA FILE:")
print(METADATA_FILE)

print("\nOUTPUT DIRECTORY:")
print(OUTPUT_DIR)


# ============================================================
# CHECK METADATA FILE
# ============================================================

if not os.path.isfile(METADATA_FILE):

    print("\nERROR: Metadata file not found!")
    print(METADATA_FILE)
    raise SystemExit


# ============================================================
# SELECTED ACTIVITIES
# ============================================================

SELECTED_CLASSES = [
    "Fighting",
    "Robbery",
    "Shoplifting",
    "Shooting",
    "RoadAccidents"
]


# ============================================================
# IMAGE / SEQUENCE SETTINGS
# ============================================================

IMG_SIZE = 224

SEQUENCE_LENGTH = 16

JPEG_QUALITY = 90


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD METADATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DCSASS METADATA")
print("=" * 70)

df = pd.read_csv(
    METADATA_FILE
)


print(
    f"Total clips in metadata : {len(df)}"
)


print("\nSelected activities:")

for activity in SELECTED_CLASSES:

    print(
        f"  - {activity}"
    )


# ============================================================
# FILTER SELECTED ACTIVITIES
# ============================================================

df = df[
    df["activity"].isin(
        SELECTED_CLASSES
    )
].copy()


print(
    f"\nTotal selected clips : {len(df)}"
)


print(
    "\nSelected activity distribution:"
)


print(
    df["activity"].value_counts()
)


# ============================================================
# PROCESS TRAIN / VALIDATION / TEST
# ============================================================

for split in [
    "train",
    "validation",
    "test"
]:

    print("\n" + "=" * 70)

    print(
        f"PROCESSING {split.upper()} DATA"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Filter current split
    # --------------------------------------------------------

    split_df = df[
        df["split"] == split
    ].copy()


    print(
        f"Clips in {split}: {len(split_df)}"
    )


    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    split_output = os.path.join(
        OUTPUT_DIR,
        split
    )


    os.makedirs(
        split_output,
        exist_ok=True
    )


    processed = 0

    skipped = 0


    # ========================================================
    # PROCESS EACH VIDEO
    # ========================================================

    for index, row in tqdm(
        split_df.iterrows(),
        total=len(split_df),
        desc=split
    ):

        # ----------------------------------------------------
        # Read metadata
        # ----------------------------------------------------

        clip_path = str(
            row["clip_path"]
        ).strip()


        activity = str(
            row["activity"]
        ).strip()


        suspicious = int(
            row["suspicious"]
        )


        video_id = str(
            row["video_id"]
        ).strip()


        # ====================================================
        # FIX VIDEO PATH
        # ====================================================

        # Example metadata path:
        #
        # ../data/datasets/Abuse/Abuse001_x264.mp4/Abuse001_x264_0.mp4
        #
        # We remove ../ because BASE_DIR already points to:
        #
        # E:\CCTV\intelligent-surveillance
        #

        clip_path = clip_path.replace(
            "\\",
            "/"
        )


        while clip_path.startswith("../"):

            clip_path = clip_path[3:]


        # ----------------------------------------------------
        # Build absolute video path
        # ----------------------------------------------------

        absolute_path = os.path.normpath(
            os.path.join(
                BASE_DIR,
                clip_path
            )
        )


        # ----------------------------------------------------
        # Check video exists
        # ----------------------------------------------------

        if not os.path.isfile(
            absolute_path
        ):

            skipped += 1

            continue


        # ====================================================
        # CREATE ACTIVITY FOLDER
        # ====================================================

        activity_folder = os.path.join(
            split_output,
            activity
        )


        os.makedirs(
            activity_folder,
            exist_ok=True
        )


        # ====================================================
        # CREATE UNIQUE SEQUENCE FOLDER
        # ====================================================

        clip_name = os.path.splitext(
            os.path.basename(
                clip_path
            )
        )[0]


        output_folder = os.path.join(
            activity_folder,
            f"{video_id}_{clip_name}"
        )


        # IMPORTANT:
        # Do NOT replace "\" or "/" here.
        #
        # They are valid path separators.
        #
        # The old code converted them into "_",
        # which could create an incorrect path.
        #


        # ====================================================
        # CHECK IF ALREADY PROCESSED
        # ====================================================

        if os.path.isdir(
            output_folder
        ):

            existing_frames = [
                filename
                for filename in os.listdir(
                    output_folder
                )
                if filename.lower().endswith(".jpg")
            ]


            if len(existing_frames) == SEQUENCE_LENGTH:

                processed += 1

                continue


            # Remove incomplete old sequence

            for filename in os.listdir(
                output_folder
            ):

                old_file = os.path.join(
                    output_folder,
                    filename
                )


                if os.path.isfile(
                    old_file
                ):

                    try:

                        os.remove(
                            old_file
                        )

                    except OSError:

                        pass


        else:

            os.makedirs(
                output_folder,
                exist_ok=True
            )


        # ====================================================
        # OPEN VIDEO
        # ====================================================

        cap = cv2.VideoCapture(
            absolute_path
        )


        if not cap.isOpened():

            skipped += 1

            continue


        # ====================================================
        # GET TOTAL FRAMES
        # ====================================================

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        if total_frames <= 0:

            cap.release()

            skipped += 1

            continue


        # ====================================================
        # SELECT TEMPORALLY DISTRIBUTED FRAMES
        # ====================================================

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            SEQUENCE_LENGTH,
            dtype=int
        )


        success_count = 0


        # ====================================================
        # EXTRACT FRAMES
        # ====================================================

        for frame_number, frame_index in enumerate(
            frame_indices
        ):

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index)
            )


            ret, frame = cap.read()


            if not ret:

                continue


            # ------------------------------------------------
            # Resize to 224 x 224
            # ------------------------------------------------

            frame = cv2.resize(
                frame,
                (
                    IMG_SIZE,
                    IMG_SIZE
                )
            )


            # ------------------------------------------------
            # Save frame
            # ------------------------------------------------

            frame_filename = os.path.join(
                output_folder,
                f"frame_{frame_number:02d}.jpg"
            )


            saved = cv2.imwrite(
                frame_filename,
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY
                ]
            )


            if saved:

                success_count += 1


        cap.release()


        # ====================================================
        # CHECK SEQUENCE
        # ====================================================

        if success_count == SEQUENCE_LENGTH:

            # ------------------------------------------------
            # Save label file
            # ------------------------------------------------

            label_file = os.path.join(
                output_folder,
                "label.txt"
            )


            with open(
                label_file,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    f"activity={activity}\n"
                )

                f.write(
                    f"suspicious={suspicious}\n"
                )

                f.write(
                    f"video_id={video_id}\n"
                )

                f.write(
                    f"source_video={absolute_path}\n"
                )


            processed += 1


        else:

            # ------------------------------------------------
            # Remove incomplete sequence
            # ------------------------------------------------

            if os.path.isdir(
                output_folder
            ):

                for filename in os.listdir(
                    output_folder
                ):

                    file_path = os.path.join(
                        output_folder,
                        filename
                    )


                    if os.path.isfile(
                        file_path
                    ):

                        try:

                            os.remove(
                                file_path
                            )

                        except OSError:

                            pass


            skipped += 1


    # ========================================================
    # SPLIT SUMMARY
    # ========================================================

    print(
        f"\n{split.upper()} COMPLETED"
    )

    print(
        f"Processed : {processed}"
    )

    print(
        f"Skipped   : {skipped}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print(
    "ACTIVITY PREPROCESSING COMPLETED"
)

print("=" * 70)


print(
    f"\nSelected classes : {len(SELECTED_CLASSES)}"
)


print("\nClasses:")

for activity in SELECTED_CLASSES:

    print(
        f"  - {activity}"
    )


print(
    f"\nFrames per sequence : {SEQUENCE_LENGTH}"
)


print(
    f"Frame size          : "
    f"{IMG_SIZE} x {IMG_SIZE}"
)


print(
    "\nOutput directory:"
)


print(
    OUTPUT_DIR
)


print(
    "\nReady for CNN-LSTM / Video Transformer."
)