import os

def read_files_from_folder(folder_path):
    """
    Reads all files from a given folder and returns their content as a dictionary.
    """
    file_contents = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                file_contents[file] = f.read()
    return file_contents