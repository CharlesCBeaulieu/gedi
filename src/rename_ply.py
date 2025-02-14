#!/usr/bin/env python3
import os
import glob
import re


def rename_files():
    # Define the folder containing the files.
    folder = "gedi_data/gen_scan"
    # Create a glob pattern for files of interest.
    pattern = os.path.join(folder, "part*_pcd_view*.ply")
    # Get a list of matching files.
    files = glob.glob(pattern)

    for file in files:
        # Extract the filename (without path)
        base = os.path.basename(file)
        # Use a regex to capture what is between "part" and "_pcd"
        match = re.search(r"part(.*?)_pcd", base)
        if match:
            extracted = match.group(1)
            new_name = f"{extracted}.ply"
            new_path = os.path.join(folder, new_name)
            print(f"Renaming '{file}' -> '{new_path}'")
            os.rename(file, new_path)
        else:
            print(f"Pattern not found in: {file}")


if __name__ == "__main__":
    rename_files()
