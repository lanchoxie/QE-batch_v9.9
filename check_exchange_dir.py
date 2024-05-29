import os
import sys

def find_missing_directories(current_dir, target_dir):
    # List to store the names of directories in the current directory
    current_dir_subdirs = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]
    
    # List to store the names of directories in the target directory
    target_dir_subdirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
    
    # List to store directories from current directory not found in target directory
    missing_dirs = []

    # Check each directory in the current directory
    for dir in current_dir_subdirs:
        found = False
        # Check if a directory in the target directory contains the current directory's name
        for target_dir_sub in target_dir_subdirs:
            if dir in target_dir_sub:
                found = True
                break
        # If not found, add to the missing list
        if not found:
            missing_dirs.append(dir)

    return missing_dirs

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_target_directory>")
        sys.exit(1)

    target_dir = sys.argv[1]
    current_dir = os.getcwd()

    missing_dirs = find_missing_directories(current_dir, target_dir)
    if missing_dirs:
        print("Directories in the current directory not found in the target directory (by inclusion):")
        for d in missing_dirs:
            print(d)
    else:
        print("All directories in the current directory are represented in the target directory.")

