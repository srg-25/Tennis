"""
This module contains IO utilities
"""
import os
import pathlib
import glob
import shutil
import sys

def get_dir(file_dir, full_path: bool = False) -> list:
    """
    It returns list of subdirectories
    :param file_dir is a folder to look sub-folders in
    :param full_path sub-folder full path will be return if it is True
    """
    full_path_list = [os.path.join(file_dir, d)
                      for d in os.listdir(file_dir)
                      if os.path.isdir(os.path.join(file_dir, d))]
    if full_path:
        return full_path_list

    subdir_list = [os.path.basename(d) for d in full_path_list]
    return subdir_list


def get_files(file_dir, file_pattern) -> list:
    """
    It returns list of file names in subdirectory
    :param file_dir: a folder to get file names from.
    :param file_pattern: '*.tiff' for example
    return list full file names.
    """
    f_names = glob.glob(os.path.join(file_dir, file_pattern))
    return f_names


def copy_tree(source_dir, destination_dir, symlinks=False, file_extension='.py', ignore_dirs=['code']):
    """
    Copy sub-folders which have files with specific extension
    Params:
        source_dir (str):  A root folder to copy sub-folders tree from
        destination_dir (str): A destination folder to copy  sub-folders tree to. It created by this function.
        symlinks (bool): See description  of shutil.copytree method
        file_extension (str): extension of files to copy
        ignore_dirs (list of strings): list of folders to not consider at all
    """
    def ignore_files_without_extensions(dirname, filenames):
        ignore_names = [name for name in filenames if not name.endswith(file_extension)]
        return ignore_names

    def ignore_folders(path_str):
        for ignore_folder in ignore_dirs:
            if ignore_folder in path_str:
                return True
        return False

    for item in os.listdir(source_dir):
        if ignore_folders(item):
            continue
        s = os.path.join(source_dir, item)
        d = os.path.join(destination_dir, item)

        # It works under Windows
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore_files_without_extensions)
            # Remove empty folders
            with os.scandir(d) as it:
                if not any(it):
                    os.rmdir(d)
        elif d.endswith(file_extension):
            shutil.copy2(s, d)


"""
-------------- Check -----------
"""


def check_get_dir():
    dir_to_check = '..'
    empty_list = get_dir(dir_to_check)
    if len(empty_list) != 0:
        print(f'check_get_dir(): The subdir list is not empty. Look path: \n\'{dir_to_check}\'')
    dir_to_check = '../Utils'
    subdir_list = get_dir(dir_to_check)
    if len(subdir_list) == 0:
        print(f'check_get_dir(): The folder has not sub-folders. Look path: \n\'{dir_to_check}\'')
    else:
        print(f'check_get_dir() -> Folder {dir_to_check} contains following sub-folders:\n {[f for f in dir_to_check]}')
    dir_to_check = '..'
    subdir_list = get_dir(dir_to_check, full_path=False)
    if len(subdir_list) == 0:
        print(f'check_get_dir(): The folder has not sub-folders. Look path: \n\'{dir_to_check}\'')
    else:
        print(f'check_get_dir() -> Folder {dir_to_check} contains following sub-folders:\n {[f for f in subdir_list]}')


def check_get_files():
    dir_to_check = '.'
    file_name_pattern = '*.py'
    names_list = get_files(dir_to_check, file_name_pattern)
    if len(names_list) == 0:
        print(f'check_get_files(): Folder {dir_to_check} has not any python files')
    else:
        print(f'check_get_files(): Folder {dir_to_check} contains following python files:\n {[f for f in names_list]}')


def check_ignore_names():
    def ignore_files_without_extensions(dirname, filenames):
        ignore_names = [name for name in filenames if not name.endswith('.py')]
        return ignore_names
    fnames = ['a.py', 'b.pdf', 'c.cpp', 'd.py']
    print(f'File names are \n\t{fnames}')
    fnames_ignore = ignore_files_without_extensions(dirname='', filenames=fnames)
    print(f'File names to ignore \n\t{fnames_ignore}')


def check_copy_tree():
    src_dir = '../'
    fext = '.py'
    dest_dir = './debug/2/code/'

    copy_tree(source_dir=src_dir, destination_dir=dest_dir, file_extension=fext, ignore_dirs=['code'])


if __name__ == '__main__':
    # check_get_dir()
    # check_get_files()
    # check_ignore_names()
    check_copy_tree()
