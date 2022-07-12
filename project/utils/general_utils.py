from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)


import os
import pickle
import shutil

def create_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def dump_to_pickle(name, file):
    with open(name + '.pkl', 'wb') as f:
       pickle.dump(file, f)


def load_pickle(name):
    file = None
    with open(name, 'rb') as f:
        file = pickle.load(f)
    return file


def rename_files(current_dir, name, extension):
  idx = 0
  files = sorted(os.listdir(current_dir))
  no_files = len(files)
  pad = len(str(no_files))
  for file in sorted(os.listdir(current_dir)):
    old_name = os.path.join(current_dir, file)
    new_name = os.path.join(current_dir, name + str(idx).zfill(pad) + extension)
    os.rename(old_name, new_name)
    idx += 1


def split_move_files(old_pos, new_pos, train, valid, test, file_names=None):
  files = os.listdir(old_pos) if file_names is None else file_names
  for idx, file in enumerate(sorted(files)):
    if idx in train:
      shutil.move(os.path.join(old_pos, file), new_pos[0])
    elif idx in valid:
      shutil.move(os.path.join(old_pos, file), new_pos[1])
    elif idx in test:
      shutil.move(os.path.join(old_pos, file), new_pos[2])


def copy_dir(src, dst):
    shutil.copytree(src, dst)