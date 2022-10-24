from pathlib import Path
import json
import os
import os.path as osp
import re
import shutil


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def may_create_folder(folder_path):
    if not osp.exists(folder_path):
        oldmask = os.umask(000)
        os.makedirs(folder_path, mode=0o777)
        os.umask(oldmask)
        return True
    return False


def make_clean_folder(folder_path):
    success = may_create_folder(folder_path)
    if not success:
        shutil.rmtree(folder_path)
        may_create_folder(folder_path)


def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) if len(c) > 0]
    return sorted(file_list_ordered, key=alphanum_key)


def list_files(folder_path, name_filter, alphanum_sort=False):
    file_list = [p.name for p in list(Path(folder_path).glob(name_filter))]
    if alphanum_sort:
        return sorted_alphanum(file_list)
    else:
        return sorted(file_list)


def list_folders(folder_path, name_filter=None, alphanum_sort=False):
    folders = list()
    for subfolder in Path(folder_path).iterdir():
        if subfolder.is_dir() and not subfolder.name.startswith('.'):
            folder_name = subfolder.name
            if name_filter is not None:
                if name_filter in folder_name:
                    folders.append(folder_name)
            else:
                folders.append(folder_name)
    if alphanum_sort:
        return sorted_alphanum(folders)
    else:
        return sorted(folders)


def read_lines(file_path):
    with open(file_path, 'r') as fin:
        lines = [line.strip() for line in fin.readlines() if len(line.strip()) > 0]
    return lines


def read_strings(file_path):
    with open(file_path, 'r') as fin:
        ret = fin.readlines()
    return ''.join(ret)


def read_json(filepath):
    with open(filepath, 'r') as fh:
        ret = json.load(fh)
    return ret
