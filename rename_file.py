#!/usr/bin/env python3
import os
import re


def rename_files_in_folder(folder_path):
    # This pattern matches filenames that start with 'part', followed by digits,
    # and then any characters until the '.ply' extension.
    pattern = re.compile(r"^part(\d+).*\.ply$", re.IGNORECASE)

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            number = match.group(1)
            new_name = f"{number}.ply"
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            print(f"Renaming: {old_path} -> {new_path}")
            os.rename(old_path, new_path)
        else:
            print(f"Skipping file: {filename} (pattern not matched)")


if __name__ == "__main__":
    # Folder containing the files to rename.
    folder = "gedi_data/working_data/scan/point_cloud"
    rename_files_in_folder(folder)
