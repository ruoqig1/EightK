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

