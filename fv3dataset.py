import logging
import os

import history

from functools import cached_property
from xpartition import HistoryDataset


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
                    tape=tape,
                    tile=tile,
                    subtile=subtile
                )
                files.append(file)
    return files


class FV3Dataset:
    def __init__(
        self,
        root,
        post_processing_root,
        fine_tapes=None,
        coarse_tapes=None,
        fine_io_layout=(1, 1),
        coarse_io_layout=(1, 1),
        coarse_tile_chunks_2d=6,
        coarse_tile_chunks_3d=1,
        fine_tile_chunks_2d=1,
        fine_tile_chunks_3d=1,
        coarse_time_chunks_2d=12,
        coarse_time_chunks_3d=1,
        fine_time_chunks_2d=1,
        fine_time_chunks_3d=1,
    ):
        self.root = root
        self.post_processing_root = post_processing_root
        if fine_tapes is None:
            self.fine_tapes = []
        else:
            self.fine_tapes = fine_tapes
        if coarse_tapes is None:
            self.coarse_tapes = []
        else:
            self.coarse_tapes = coarse_tapes
        self.fine_io_layout = fine_io_layout
        self.coarse_io_layout = coarse_io_layout
        self.tiles = TILES

        self.coarse_tile_chunks_2d = coarse_tile_chunks_2d
        self.coarse_tile_chunks_3d = coarse_tile_chunks_3d
        self.fine_tile_chunks_2d = fine_tile_chunks_2d
        self.fine_tile_chunks_3d = fine_tile_chunks_3d
        self.coarse_time_chunks_2d = coarse_time_chunks_2d
        self.coarse_time_chunks_3d = coarse_time_chunks_3d
        self.fine_time_chunks_2d = fine_time_chunks_2d
        self.fine_time_chunks_3d = fine_time_chunks_3d

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

    def symlink_history(self):
        history = os.path.join(self.post_processing_root, "history")
        try:
            os.makedirs(history, exist_ok=False)
            for segment in self.history_segments:
                segment_symlink = os.path.join(history, segment)
                os.makedirs(segment_symlink, exist_ok=False)
                for file in self.history_files:
                    source = os.path.join(self.root, "history", segment, file)
                    symlink = os.path.join(segment_symlink, file)
                    logging.info(f"Symlinking {source} to {symlink}.")
                    os.symlink(source, symlink)
        except FileExistsError:
            logging.info(
                f"History symlinks already exist at "
                f"{self.post_processing_root}. Skipping."
            )

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

    @property
    def mppnccombine_history_inputs(self):
        """Returns a function to determine inputs from snakemake wildcards."""
        def func(wildcards):
            segment = wildcards["segment"]
            tape = wildcards["tape"]
            tile = int(wildcards["tile"])
            files = self.tapes_to_tiles_to_files[tape][tile]
            root = os.path.join(
                self.post_processing_root,
                "history",
                segment
            )
            paths = []
            for file in files:
                path = os.path.join(root, file)
                paths.append(path)
            return paths
        return func

    @property
    def mppnccombine_history_output(self):
        """Returns a single filename for the mppnccombine rule output."""
        return os.path.join(
            self.post_processing_root,
            "history",
            "{segment}",
            "{tape}.tile{tile}.nc"
        )

    @property
    def zarr_history_input(self):
        paths = []
        for segment in self.history_segments:
            for tile in self.tiles:
                path = os.path.join(
                    self.post_processing_root,
                    "history",
                    segment,
                    f"{{tape}}.tile{tile}.nc"
                )
                paths.append(path)
        return paths

    @property
    def zarr_history_root(self):
        return os.path.join(
            self.post_processing_root,
            "zarr-history"
        )

    @property
    def zarr_history_output(self):
        return os.path.join(self.zarr_history_root, "{tape}.zarr")

    @cached_property
    def datasets(self):
        datasets = {}
        for tape in self.combined_tapes:
            target_chunks_2d = self.target_chunks_2d(tape)
            target_chunks_3d = self.target_chunks_3d(tape)
            try:
                datasets[tape] = self._to_dask(
                    tape, self.root, target_chunks_2d, target_chunks_3d
                )
            except ValueError:
                print(f"Could not convert {tape} to dask.")
        return datasets

    def _to_dask(self, tape, root, target_chunks_2d, target_chunks_3d):
        """Lazily open a combined dataset for a given tape.

        Note all data must be combined across subtiles and be accessible
        through the post_processing_root directory first.
        """
        paths = history.segment_paths(root)
        hd = HistoryDataset(
            tape,
            paths,
            target_chunks_2d=target_chunks_2d,
            target_chunks_3d=target_chunks_3d
        )
        return hd.to_dask()

    @property
    def tile_chunks_2d(self):
        def func(wildcards):
            tape = wildcards["tape"]
            if tape in self.coarse_tapes:
                return self.coarse_tile_chunks_2d
            else:
                return self.fine_tile_chunks_2d
        return func

    @property
    def time_chunks_2d(self):
        def func(wildcards):
            tape = wildcards["tape"]
            if tape in self.coarse_tapes:
                return self.coarse_time_chunks_2d
            else:
                return self.fine_time_chunks_2d
        return func

    @property
    def tile_chunks_3d(self):
        def func(wildcards):
            tape = wildcards["tape"]
            if tape in self.coarse_tapes:
                return self.coarse_tile_chunks_3d
            else:
                return self.fine_tile_chunks_3d
        return func

    @property
    def time_chunks_3d(self):
        def func(wildcards):
            tape = wildcards["tape"]
            if tape in self.coarse_tapes:
                return self.coarse_time_chunks_3d
            else:
                return self.fine_time_chunks_3d
        return func

    def target_chunks_2d(self, tape):
        if tape in self.coarse_tapes:
            time_chunks = self.coarse_time_chunks_2d
            tile_chunks = self.coarse_tile_chunks_2d
        else:
            time_chunks = self.fine_time_chunks_2d
            tile_chunks = self.fine_tile_chunks_2d
        return {"time": time_chunks, "tile": tile_chunks}

    def target_chunks_3d(self, tape):
        if tape in self.coarse_tapes:
            time_chunks = self.coarse_time_chunks_3d
            tile_chunks = self.coarse_tile_chunks_3d
        else:
            time_chunks = self.fine_time_chunks_3d
            tile_chunks = self.fine_tile_chunks_3d
        return {"time": time_chunks, "tile": tile_chunks}

    @property
    def pp_history_root(self):
        return os.path.join(self.post_processing_root, "history")
