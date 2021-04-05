import dataclasses
import functools
import os
import typing

import numpy as np
import xarray as xr

import xpartition

PATTERN = "{tape}.tile{tile}.nc"
TILES = range(1, 7)

def expand_dims_for_certain_variables(ds, variables, dims):
    return ds.assign({variable: ds[variable].expand_dims(dims) for variable in variables})

def rechunk_da(da, target_chunks):
    chunks = {dim: chunks for dim, chunks in target_chunks.items() if dim in da.dims}
    return da.chunk(chunks)

def shares_dims(da, dims):
    return bool(set(da.dims).intersection(set(dims)))

def has_dims(da, dims):
    if isinstance(dims, str):
        dims = [dims]
    return all(dim in da.dims for dim in dims)

def is_dimension_coordinate(v):
    return v.name in v.dims

def targeted_open_mfdataset(filenames, keep_variables, **kwargs):
    preprocess = lambda ds: ds[keep_variables]
    return xr.open_mfdataset(filenames, preprocess=preprocess, **kwargs)

def full_region(ds):
    return {dim: slice(None, None) for dim in ds.dims}

def total_chunks(da):
    return np.product([len(chunks) for chunks in da.chunks])

def factorize(num):
    return [n for n in range(1, num + 1) if num % n == 0]

def closest_factors(num):
    factors = factorize(num)
    a = factors[np.argmin(np.abs(np.array(factors) - np.sqrt(num)))]
    b = num // a
    return a, b

def safe_concat(datasets, **kwargs):
    common = set.intersection(*[set(dataset.variables) for dataset in datasets])
    datasets = [ds[list(common)] for ds in datasets]
    return xr.concat(datasets, **kwargs)


@dataclasses.dataclass
class HistoryDataset:
    tape: str
    directories: typing.Sequence[str]
    target_chunks_2d: typing.Dict[typing.Hashable, int]
    target_chunks_3d: typing.Dict[typing.Hashable, int]
    time_dim: typing.Hashable = "time"
    tile_dim: typing.Hashable = "tile"
    x_dim: typing.Hashable = "grid_xt_coarse"
    y_dim: typing.Hashable = "grid_yt_coarse"
    z_dim: typing.Hashable = "pfull"
    horizontal_dims: typing.Sequence[typing.Hashable] = ("grid_xt_coarse", "grid_yt_coarse", "grid_x_coarse", "grid_y_coarse", "grid_xt", "grid_yt", "grid_x", "grid_y")
    vertical_dims: typing.Sequence[typing.Hashable] = ("pfull", "phalf", "plev")
    tiles: int = 6

    @property
    def segments(self):
        return len(self.directories)

    @property
    def files(self):
        files = []
        for segment in self.directories:
            pattern = os.path.join(segment, PATTERN)
            files.append([pattern.format(tape=self.tape, tile=tile) for tile in TILES])
        return xr.DataArray(files, dims=[self.time_dim, self.tile_dim])

    @property
    def _sample_file(self):
        indexers = {self.time_dim: -1, self.tile_dim: 0}
        return self.files.isel(indexers).item()

    @functools.cached_property
    def _sample_dataset(self):
        return xr.open_dataset(self._sample_file, chunks={})

    @functools.cached_property
    def sample_dataset(self):
        return expand_dims_for_certain_variables(
            self._sample_dataset,
            self.tile_varying_variables,
            self.tile_dim
        )

    @property
    def dims(self):
        return set(self.sample_dataset.dims)

    @property
    def combining_dims(self):
        return {self.time_dim, self.tile_dim}

    @property
    def static_dims(self):
        return self.dims - self.combining_dims

    @property
    def netcdf_sizes(self):
        return self.sample_dataset.sizes

    @property
    def total_sizes(self):
        sizes = {dim: size for dim, size in self.netcdf_sizes.items() if dim in self.static_dims}
        sizes[self.tile_dim] = self.tiles
        sizes[self.time_dim] = self.segments * self.netcdf_sizes[self.time_dim]
        return sizes

    @property
    def netcdf_chunks(self):
        chunks = {dim: size for dim, size in self.netcdf_sizes.items() if dim in self.static_dims}
        chunks[self.tile_dim] = 1
        chunks[self.time_dim] = 1
        return chunks

    @property
    def sample_data_arrays(self):
        data_vars = [da for da in self.sample_dataset.data_vars.values()]
        coords = [da for da in self.sample_dataset.coords.values()]
        return data_vars + coords

    @property
    def _sample_data_arrays(self):
        data_vars = [da for da in self._sample_dataset.data_vars.values()]
        coords = [da for da in self._sample_dataset.coords.values()]
        return data_vars + coords

    def is_tile_varying(self, da):
        return shares_dims(da, self.horizontal_dims) and not is_dimension_coordinate(da)

    @functools.cached_property
    def tile_varying_variables(self):
        return [da.name for da in self._sample_data_arrays if self.is_tile_varying(da)]

    @functools.cached_property
    def chunked_variables(self):
        return [da.name for da in self.sample_data_arrays if has_dims(da, self.combining_dims)]

    @functools.cached_property
    def unchunked_variables(self):
        return [da.name for da in self.sample_data_arrays if da.name not in self.chunked_variables]

    @functools.cached_property
    def chunked_variables_2d(self):
        return [name for name in self.chunked_variables if not shares_dims(self.sample_dataset[name], self.vertical_dims)]

    @functools.cached_property
    def chunked_variables_3d(self):
        return [name for name in self.chunked_variables if shares_dims(self.sample_dataset[name], self.vertical_dims)]

    @functools.cached_property
    def non_chunked_dimension_coordinates(self):
        return [dim for dim in self._sample_dataset.dims if dim != self.time_dim]

    @functools.cached_property
    def time_varying_coordinates(self):
        return [name for name in self.unchunked_variables if has_dims(self.sample_dataset[name], self.time_dim)]

    @functools.cached_property
    def tile_varying_coordinates(self):
        return [name for name in self.unchunked_variables if has_dims(self.sample_dataset[name], self.tile_dim)]

    def rechunk_variables(self, ds, variables, chunks):
        rechunked = {}
        for variable in variables:
            rechunked[variable] = rechunk_da(ds[variable], chunks)
        return ds.assign(rechunked)

    def rechunk(self, ds):
        ds = self.rechunk_variables(ds, self.chunked_variables_2d, self.target_chunks_2d)
        ds = self.rechunk_variables(ds, self.chunked_variables_3d, self.target_chunks_3d)
        return ds

    @functools.cached_property
    def combined_template_dataset(self):
        tiled = xr.concat(
            self.tiles * [self.sample_dataset],
            dim=self.tile_dim,
            data_vars="minimal"
        )
        segmented = xr.concat(
            self.segments * [tiled],
            dim=self.time_dim,
            data_vars="minimal"
        )
        return self.rechunk(segmented)

    def _validate_chunks(self, ranks):
        assert ranks % self.tiles == 0

        time_size = self.netcdf_sizes[self.time_dim]
        time_chunks_2d = self.target_chunks_2d[self.time_dim]
        time_chunks_3d = self.target_chunks_3d[self.time_dim]
        assert time_size % time_chunks_2d == 0
        assert time_size % time_chunks_3d == 0

        time_partitions = ranks // self.tiles
        assert (time_size // time_chunks_2d) % time_partitions == 0
        assert (time_size // time_chunks_3d) % time_partitions == 0

    def _open_dataset_2d(self, filename):
        chunks = {self.time_dim: self.target_chunks_2d[self.time_dim]}
        ds = xr.open_dataset(filename, chunks=chunks)
        return ds[self.chunked_variables_2d].expand_dims(self.tile_dim)

    def _open_dataset_3d(self, filename):
        chunks = {self.time_dim: self.target_chunks_3d[self.time_dim]}
        ds = xr.open_dataset(filename, chunks=chunks)
        return ds[self.chunked_variables_3d].expand_dims(self.tile_dim)

    def _open_chunked_variables(self, filename):
        ds_2d = self._open_dataset_2d(filename)
        ds_3d = self._open_dataset_3d(filename)
        ds = xr.merge([ds_2d, ds_3d])
        return ds.drop([dim for dim in ds.dims if dim in ds.coords])

    def _open_all_chunked_variables(self):
        segments = []
        for segment in range(self.files.sizes[self.time_dim]):
            tiles = []
            for tile in range(self.files.sizes[self.tile_dim]):
                filename = self.files.isel(time=segment, tile=tile).item()
                tiles.append(self._open_chunked_variables(filename))
            tiles = safe_concat(tiles, dim=self.tile_dim, data_vars="minimal")
            segments.append(tiles)
        return safe_concat(segments, dim=self.time_dim, data_vars="minimal")

    def _open_time_varying_coordinates(self):
        filenames = self.files.isel({self.tile_dim: 0}).values.tolist()
        return targeted_open_mfdataset(
            filenames,
            self.time_varying_coordinates,
            concat_dim=[self.time_dim],
            combine="by_coords"
        ).load()

    def _open_tile_varying_coordinates(self):
        filenames = self.files.isel({self.time_dim: 0}).values.tolist()
        return targeted_open_mfdataset(
            filenames,
            self.tile_varying_coordinates,
            concat_dim=[self.tile_dim],
            combine="nested"
        ).assign({self.tile_dim: range(6)}).load()

    def initialize_store(self, store):
        ds = self.to_dask()
        ds.to_zarr(store, compute=False)

    def to_dask(self):
        ds = xr.merge([
            self._open_all_chunked_variables(),
            self._open_time_varying_coordinates(),
            self._open_tile_varying_coordinates(),
            self._sample_dataset[self.non_chunked_dimension_coordinates]
        ])
        return self.rechunk(ds)

    def _write_partition(self, ranks, rank, variables, store):
        if variables:
            ds = self.to_dask()[variables]
            sample = variables[0]
            partition = ds[sample].partition.indexers(ranks, rank, [self.tile_dim, self.time_dim])
            if partition is None:
                print(f"No work to be done on rank {rank} for {', '.join(variables)}.")
            else:
                print(f"Writing {', '.join(variables)} to {partition} on rank {rank}.")
                ds.isel(partition).to_zarr(store, region=partition)

    def write_partition(self, ranks, rank, store):
        self._write_partition(ranks, rank, self.chunked_variables_2d, store)
        self._write_partition(ranks, rank, self.chunked_variables_3d, store)

    def print_expected_chunk_sizes(self):
        x_size = self.total_sizes[self.x_dim]
        y_size = self.total_sizes[self.y_dim]
        z_size = self.total_sizes[self.z_dim]
        tile_size_2d = self.target_chunks_2d[self.tile_dim]
        tile_size_3d = self.target_chunks_3d[self.tile_dim]
        t_size_2d = self.target_chunks_2d[self.time_dim]
        t_size_3d = self.target_chunks_3d[self.time_dim]
        face_size = x_size * y_size
        chunk_sizes_2d = tile_size_2d * t_size_2d * face_size * 4.0 / 1.0e6
        chunk_sizes_3d = tile_size_3d * t_size_3d * face_size * z_size * 4.0 / 1.0e6
        print(f"Chunk size for 2D variables is {chunk_sizes_2d} MB; "
              f"chunk size for 3D variables is {chunk_sizes_3d} MB.")

    def print_possible_chunk_sizes(self, ranks):
        assert ranks % self.tiles == 0
        time_partitions = ranks // self.tiles

        time_size = self.netcdf_sizes[self.time_dim]
        factors = factorize(time_size)
        return [factor for factor in factors if (time_size // factor) % time_partitions == 0]

    def print_possible_ranks(self):
        factors = factorize(self._total_dask_blocks)
        print([factor for factor in factors if factor % self.tiles == 0])
