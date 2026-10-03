import os

dataset_path = "../data/datasets"

print("\nDCSASS DATASET CHECK")
print("=" * 40)

if not os.path.exists(dataset_path):
    print("Dataset folder not found!")
    exit()

classes = []

for item in os.listdir(dataset_path):

    full_path = os.path.join(dataset_path, item)

    if os.path.isdir(full_path):
        classes.append(item)

print("Number of folders/classes:", len(classes))
print()

for class_name in sorted(classes):

    class_path = os.path.join(dataset_path, class_name)

    videos = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    print(f"{class_name:<20} {len(videos)} videos")

print()
print("Dataset check completed.")