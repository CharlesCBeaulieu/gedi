import os

def rename_files_in_folder(folder):
    """
    Iterates through every file in the specified folder and renames files that contain an underscore.
    The new filename will consist of only the part before the underscore, keeping the original extension.
    
    For example:
        "770066_2.ply" -> "770066.ply"
    
    Parameters:
        folder (str): Path to the folder containing the files.
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        # Only process files, not directories.
        if os.path.isfile(file_path) and "_" in filename:
            name_part, ext = os.path.splitext(filename)
            # Keep only the part before the underscore.
            new_name = name_part.split("_")[0] + ext
            new_file_path = os.path.join(folder, new_name)
            print(f"Renaming '{filename}' to '{new_name}'")
            os.rename(file_path, new_file_path)

# Example usage:
rename_files_in_folder("../bindmount/data/generated_scan/median")
