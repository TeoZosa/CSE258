
from os import walk, path
from fnmatch import filter as fnmatch_filter


def find_files(file_path, extension):
    """'
     Recursively find files at path with extension; pulled from StackOverflow
    ''"""#
    for root, dirs, files in walk(file_path):
        for file in fnmatch_filter(files, extension):
            yield path.join(root, file)