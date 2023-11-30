import gzip
import shutil
import os

def unzip_all(zip_dir, unzip_dir, redo_all=False):
    for f in os.listdir(zip_dir):
        dest = unzip_dir + f.replace('.gz', '')
        if redo_all | (not os.path.exists(dest)):
            decompress_gz_file(zip_dir + f, dest)
            print('Unzipped', f,flush=True)
        else:
            print('Previously done', f,flush=True)

def decompress_gz_file(file_path, output_path):
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


import zipfile
import os

def unzip_file(zip_path, extract_to):
    """
    Unzip a file to a specified directory.

    :param zip_path: The path to the .zip file.
    :param extract_to: The directory where files will be extracted.
    """
    # Check if the zip file exists
    if not os.path.exists(zip_path):
        print("The zip file does not exist: " + zip_path)
        return

    # Check if the destination directory exists, if not create it
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Unzipping the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Files extracted to {extract_to}")

