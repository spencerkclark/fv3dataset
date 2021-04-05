import collections
import glob
import os


TILES = 6


def history_path(root):
    """Return the path to the 'history' directory of a run if it exists.

    Parameters
    ----------
    root : str
        Root directory of the simulation.  Does not include the "history"
        stub.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If the 'history' directory cannot be found at the given root.
    """
    history = os.path.join(root, "history")
    if os.path.isdir(history):
        return history
    else:
        raise ValueError(f"Directory {root} is not a valid root directory.")


def segment_paths(root):
    """Return the absolute paths of all completed segments of the run.

    Parameters
    ----------
    root : str
        Root directory of the simulation.  Does not include the "history"
        stub.

    Returns
    -------
    list
    """
    directories = []
    history = history_path(root)
    for d in os.listdir(history):
        path = os.path.join(history, d)
        if os.path.isdir(path):
            directories.append(path)
    return sorted(directories)


def filenames(root, combined=None):
    """Return a list of files that are present in all segment directories.

    This means that if only some segments have outputs combined via
    mppnccombine, they will not show up in the list.

    Parameters
    ----------
    root : str
        Root directory of simulation.
    combined : bool (optional)
        Whether to only look for combined files.

    Returns
    -------
    list
    """
    segments = segment_paths(root)
    globname = "*.nc*" if combined is None else "*.nc"
    files = []
    for segment in segments:
        globstring = os.path.join(segment, globname)
        segment_files = sorted(glob.glob(globstring))
        segment_files = [os.path.basename(file) for file in segment_files]
        files.append(set(segment_files))
    return list(set.intersection(*files))


def tapes(root, combined=None):
    return list(tapes_to_files(root, combined=combined).keys())


def tapes_to_files(root, combined=None):
    names = filenames(root, combined=combined)
    tapes = collections.defaultdict(list)
    for name in names:
        if ".tile" in name:
            tape = name[: name.find(".tile")]
            tapes[tape].append(name)
    return tapes


def tapes_to_tiles_to_files(root, combined=None):
    _tapes_to_files = tapes_to_files(root, combined=combined)
    result = collections.defaultdict(dict)
    for tape, files in _tapes_to_files.items():
        for tile in range(1, TILES + 1):
            tile_files = [f for f in files if f"tile{tile}" in f]
            result[tape][tile] = tile_files
    return result


def subtile_properties(root, combined=None):
    _tapes_to_files = tapes_to_files(root, combined=combined)
    properties = collections.defaultdict(list)
    for tape, files in _tapes_to_files.items():
        n = len(files)
        subtiles = n // TILES
        properties[subtiles].append(tape)
    return properties
