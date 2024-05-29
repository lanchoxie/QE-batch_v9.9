#!/bin/bash

# Check if user provided an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <source_directory_name>"
    exit 1
fi

# Source directory from user input
SRC_DIR="$1"

# Remove trailing slash if present
SRC_DIR="${SRC_DIR%/}"

# Define the destination directory
DEST_DIR="${SRC_DIR}-extractor"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Loop through each subdirectory in the source directory
for subdir in "$SRC_DIR"/*; do
    # Only process if it's a directory
    if [ -d "$subdir" ]; then
        # Get the base name of the subdirectory
        base_subdir=$(basename "$subdir")

        # Create corresponding subdirectory in the destination directory
        mkdir -p "$DEST_DIR/$base_subdir"

        # Check if the current subdirectory is "QE-batch"
        if [ "$base_subdir" == "QE-batch" ]; then
            cp -r "$subdir"/* "$DEST_DIR/$base_subdir/"
        else
            # Copy the specified files from source to destination
            cp "$subdir/out_relax_$base_subdir" "$DEST_DIR/$base_subdir/" 2>/dev/null
            cp "$subdir/in_relax_$base_subdir" "$DEST_DIR/$base_subdir/" 2>/dev/null
            cp "$subdir"/*.UPF "$DEST_DIR/$base_subdir/" 2>/dev/null
            cp "$subdir"/xty_test "$DEST_DIR/$base_subdir/" 2>/dev/null
        fi
    fi
    cp "$SRC_DIR/${base_subdir}.vasp" "$DEST_DIR/" 2>/dev/null
done

echo "Files have been copied to $DEST_DIR."

