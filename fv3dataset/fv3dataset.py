import os

from functools import cached_property

from . import history
from .history_dataset import HistoryDataset


def _subtiles(layout):
    x, y = layout
    subtiles = x * y
    if subtiles == 1:
        return None
    else:
        return subtiles


HISTORY_PATTERN = "{tape}.tile{tile}.nc"
HISTORY_PATTERN_SUBTILES = "{tape}.tile{tile}.nc.{subtile:04d}"
TILES = range(1, 7)


def _history_filenames(subtiles, tape):
    files = []
    if subtiles is None:
        for tile in TILES:
            file = HISTORY_PATTERN.format(tape=tape, tile=tile)
            files.append(file)
    else:
        for tile in TILES:
            for subtile in range(subtiles):
                file = HISTORY_PATTERN_SUBTILES.format(
                    tape=tape, tile=tile, subtile=subtile
                )
                files.append(file)
    return files


class FV3Dataset:
    def __init__(self, root, target_chunk_size="128Mi"):
        self.root = root
        self.target_chunk_size = target_chunk_size
        self.tiles = TILES

    @property
    def tapes(self):
        return history.tapes(self.root)

    @property
    def combined_tapes(self):
        return history.tapes(self.root, combined=True)

    @property
    def history_segments(self):
        paths = history.segment_paths(self.root)
        return [os.path.basename(path) for path in paths]

    @property
    def history_files(self):
        return history.filenames(self.root)

    @property
    def tapes_to_tiles_to_files(self):
        return history.tapes_to_tiles_to_files(self.root)

    @property
    def fine_subtiles(self):
        return _subtiles(self.fine_io_layout)

    @property
    def coarse_subtiles(self):
        return _subtiles(self.coarse_io_layout)

    def _raw_coarse_history_files(self, tape):
        subtiles = self.coarse_subtiles
        return _history_filenames(subtiles, tape)

    def _raw_fine_history_files(self, tape):
        subtiles = self.fine_subtiles
        return _history_filenames(subtiles, tape)

    @cached_property
    def datasets(self):
        datasets = {}
        for tape in self.combined_tapes:
            datasets[tape] = self._to_dask(tape, self.root)
        return datasets

    def tape_to_dask(self, tape):
        return self._to_dask(tape, self.root)

    def _to_dask(self, tape, root):
        """Lazily open a combined dataset for a given tape.

        Note all data must be combined across subtiles and be accessible
        through the root directory first.
        """
        paths = history.segment_paths(root)
        return HistoryDataset(
            tape, paths, target_chunk_size=self.target_chunk_size
        ).to_dask()
