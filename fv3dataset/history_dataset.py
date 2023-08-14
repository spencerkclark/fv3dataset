import dataclasses
import os
import typing

import dask
import xarray as xr


try:
    from functools import cached_property
except ImportError:
    cached_property = property


HORIZONTAL_DIMS = (
    "grid_xt_coarse",
    "grid_yt_coarse",
    "grid_x_coarse",
    "grid_y_coarse",
    "grid_xt",
    "grid_yt",
    "grid_x",
    "grid_y",
)
PATTERN = "{tape}.tile{tile}.nc"
TILES = range(1, 7)
VERTICAL_DIMS = ("pfull", "phalf", "plev")


def expand_dims_for_certain_variables(ds, variables, dims):
    return ds.assign(
        {variable: ds[variable].expand_dims(dims) for variable in variables}
    )


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


def safe_concat(datasets, **kwargs):
    """Concatenate datasets together, automaticalally dropping
    any variables that are not common to all provided datasets.
    """
    common = set.intersection(*[set(dataset.variables) for dataset in datasets])
    datasets = [ds[list(common)] for ds in datasets]
    return xr.concat(datasets, **kwargs)


@dataclasses.dataclass
class HistoryDataset:
    tape: str
    directories: typing.Sequence[str]
    target_chunk_size: str = "128Mi"
    auto_chunk_dims: typing.Sequence[typing.Hashable] = ("time", "tile")
    time_dim: typing.Hashable = "time"
    tile_dim: typing.Hashable = "tile"
    x_dim: typing.Hashable = "grid_xt_coarse"
    y_dim: typing.Hashable = "grid_yt_coarse"
    z_dim: typing.Hashable = "pfull"
    horizontal_dims: typing.Sequence[typing.Hashable] = HORIZONTAL_DIMS
    vertical_dims: typing.Sequence[typing.Hashable] = VERTICAL_DIMS
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

    @cached_property
    def _sample_dataset(self):
        return xr.open_dataset(self._sample_file, chunks={})

    @cached_property
    def sample_dataset(self):
        return expand_dims_for_certain_variables(
            self._sample_dataset, self.tile_varying_variables, self.tile_dim
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

    @cached_property
    def tile_varying_variables(self):
        return [da.name for da in self._sample_data_arrays if self.is_tile_varying(da)]

    @cached_property
    def chunked_variables(self):
        return [
            da.name
            for da in self.sample_data_arrays
            if has_dims(da, self.combining_dims)
        ]

    @cached_property
    def unchunked_variables(self):
        return [
            da.name
            for da in self.sample_data_arrays
            if da.name not in self.chunked_variables
        ]

    @cached_property
    def chunked_variables_2d(self):
        return [
            name
            for name in self.chunked_variables
            if not shares_dims(self.sample_dataset[name], self.vertical_dims)
        ]

    @cached_property
    def chunked_variables_3d(self):
        return [
            name
            for name in self.chunked_variables
            if shares_dims(self.sample_dataset[name], self.vertical_dims)
        ]

    @cached_property
    def non_chunked_dimension_coordinates(self):
        return [dim for dim in self._sample_dataset.dims if dim != self.time_dim]

    @cached_property
    def time_varying_coordinates(self):
        return [
            name
            for name in self.unchunked_variables
            if has_dims(self.sample_dataset[name], self.time_dim)
        ]

    @cached_property
    def tile_varying_coordinates(self):
        return [
            name
            for name in self.unchunked_variables
            if has_dims(self.sample_dataset[name], self.tile_dim)
        ]

    def _rechunk_da(self, da, auto_chunk_dim, already_autochunked_dims):
        chunks = {}
        for dim in da.dims:
            if dim == auto_chunk_dim:
                chunks[dim] = "auto"
            elif dim in already_autochunked_dims:
                chunks[dim] = None
            else:
                chunks[dim] = -1
        with dask.config.set({"array.chunk-size": self.target_chunk_size}):
            return da.chunk(chunks)

    def rechunk_da(self, da):
        # Iteratively rechunk DataArray over one dimension at a time to ensure
        # that rechunking occurs preferentially in the following order:
        # 1. Over dimensions specified in auto_rechunk_dims.
        # 2. Over the remaining dimensions, in the order that they appear in
        #    the DataArray.
        rechunking_order = []
        for dim in self.auto_chunk_dims:
            if dim in da.dims:
                rechunking_order.append(dim)
        for dim in da.dims:
            if dim not in self.auto_chunk_dims:
                rechunking_order.append(dim)

        already_autochunked_dims = []
        for dim in rechunking_order:
            da = self._rechunk_da(da, dim, already_autochunked_dims)
            already_autochunked_dims.append(dim)
        return da

    def rechunk_variables(self, ds, variables):
        rechunked = {}
        for variable in variables:
            rechunked[variable] = self.rechunk_da(ds[variable])
        return ds.assign(rechunked)

    def rechunk(self, ds):
        ds = self.rechunk_variables(ds, self.chunked_variables_2d)
        ds = self.rechunk_variables(ds, self.chunked_variables_3d)
        return ds

    def _open_dataset_2d(self, filename):
        chunks = {self.time_dim: "auto"}
        with dask.config.set({"array.chunk-size": self.target_chunk_size}):
            ds = xr.open_dataset(filename, chunks=chunks)
        return ds[self.chunked_variables_2d].expand_dims(self.tile_dim)

    def _open_dataset_3d(self, filename):
        chunks = {self.time_dim: "auto"}
        with dask.config.set({"array.chunk-size": self.target_chunk_size}):
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
        if not self.chunked_variables_2d and not self.chunked_variables_3d:
            return xr.Dataset()
        else:
            return targeted_open_mfdataset(
                filenames,
                self.time_varying_coordinates,
                concat_dim=[self.time_dim],
                combine="by_coords",
            ).load()

    def _open_tile_varying_coordinates(self):
        filenames = self.files.isel({self.time_dim: 0}).values.tolist()
        return (
            targeted_open_mfdataset(
                filenames,
                self.tile_varying_coordinates,
                concat_dim=[self.tile_dim],
                combine="nested",
            )
            .assign({self.tile_dim: range(6)})
            .load()
        )

    def initialize_store(self, store):
        ds = self.to_dask()
        ds.to_zarr(store, compute=False)

    def to_dask(self):
        ds = xr.merge(
            [
                self._open_all_chunked_variables(),
                self._open_time_varying_coordinates(),
                self._open_tile_varying_coordinates(),
                self._sample_dataset[self.non_chunked_dimension_coordinates],
            ]
        )
        return self.rechunk(ds)

    def _write_partition(self, ranks, rank, variables, store):
        try:
            import xpartition  # noqa
        except ImportError:
            raise ImportError("Using _write_partition requires xpartition be installed.")
        if variables:
            ds = self.to_dask()[variables]
            sample = variables[0]
            partition = ds[sample].partition.indexers(
                ranks, rank, [self.tile_dim, self.time_dim]
            )
            if partition is None:
                print(f"No work to be done on rank {rank} for {', '.join(variables)}.")
            else:
                print(f"Writing {', '.join(variables)} to {partition} on rank {rank}.")
                ds.isel(partition).to_zarr(store, region=partition)

    def write_partition(self, ranks, rank, store):
        self._write_partition(ranks, rank, self.chunked_variables_2d, store)
        self._write_partition(ranks, rank, self.chunked_variables_3d, store)
