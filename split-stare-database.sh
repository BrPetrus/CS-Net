#!/bin/bash

# Define the directories
eyes_dir="eyes"
labels_dir="labels-ah"
train_images_dir="training/images"
train_labels_dir="training/labels-ah"
test_images_dir="testing/images"
test_labels_dir="testing/labels-ah"

# Create directories if they don't exist
mkdir -p "$train_images_dir" "$train_labels_dir" "$test_images_dir" "$test_labels_dir"

# Step 1: Get the list of images that have ground truth (matching files in eyes/ and labels-ah/)
# Assuming the filenames match, i.e., im1.ppm in eyes corresponds to im1.ppm in labels-ah
images_with_labels=()
for img in "$eyes_dir"/im*.ppm; do
    img_name=$(basename "$img" .ppm)
    label_file="$labels_dir/$img_name.ah.ppm"
    if [ -f "$label_file" ]; then
        images_with_labels+=("$img_name")
    fi
done

# Step 2: Shuffle and split into training and testing sets (80%/20%)
num_images=${#images_with_labels[@]}
num_train=$((num_images * 80 / 100))
num_test=$((num_images - num_train))

# Shuffle the array
shuffled_images=($(shuf -e "${images_with_labels[@]}"))

# Split into training and testing
train_images=("${shuffled_images[@]:0:$num_train}")
test_images=("${shuffled_images[@]:$num_train:$num_test}")

# Step 3: Move images and labels to appropriate directories
for img_name in "${train_images[@]}"; do
    # Move original image
    cp "$eyes_dir/$img_name.ppm" "$train_images_dir/"
    # Move corresponding label
    cp "$labels_dir/$img_name.ah.ppm" "$train_labels_dir/"
done

for img_name in "${test_images[@]}"; do
    # Move original image
    cp "$eyes_dir/$img_name.ppm" "$test_images_dir/"
    # Move corresponding label
    cp "$labels_dir/$img_name.ah.ppm" "$test_labels_dir/"
done

echo "Data has been split into training and testing sets."

