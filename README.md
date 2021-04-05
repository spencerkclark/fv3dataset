# fv3dataset

This is a library meant for interacting with data produced by running SHiELD.
SHiELD outputs can sometimes be complicated to deal with, because outputs for
specific variables are distributed across multiple tile and/or subtile files,
and across multiple time segments.  Therefore accessing the full spatial time
series of a variable involves accessing many separate files.

Xarray is capable of lazily opening all of these files -- for their metadata
only -- and combining them into a single coherent data structure that makes
things look like you opened them from a single netCDF file.  Here we make this
especially convenient for outputs from SHiELD.

## Example

The basic usage of this package is the following.  All you need to provide to
create an `fv3dataset.FV3Dataset` object is a path to the root directory of the
output of a simulation:

```
>>> import fv3dataset
>>> root = "/lustre/f2/scratch/Spencer.Clark/SHiELD/20200120.00Z.C48.L79z12.2021-04-04"
>>> fv3ds = fv3dataset.FV3Dataset(root)
```

If there are any sets of history files that have been
`mppnccombine`'d for all segments of the run, `fv3dataset` will recognize it,
and combine those sets into lazy datasets:

```
>>> datasets = fv3ds.datasets
>>> list(datasets.keys())
['demo_coarse_inst', 'grid_spec_coarse', 'demo_ave', 'demo_inst', 'demo_coarse_ave']
>>> datasets["demo_ave"]
<xarray.Dataset>
Dimensions:     (grid_xt: 48, grid_yt: 48, nv: 2, pfull: 79, phalf: 80, tile: 6, time: 4)
Coordinates:
  * time        (time) object 2020-01-20 06:00:00 ... 2020-01-21 18:00:00
  * tile        (tile) int64 0 1 2 3 4 5
  * grid_xt     (grid_xt) float64 1.0 2.0 3.0 4.0 5.0 ... 45.0 46.0 47.0 48.0
  * grid_yt     (grid_yt) float64 1.0 2.0 3.0 4.0 5.0 ... 45.0 46.0 47.0 48.0
  * pfull       (pfull) float64 4.514 8.301 12.45 16.74 ... 989.5 994.3 998.3
  * nv          (nv) float64 1.0 2.0
  * phalf       (phalf) float64 3.0 6.467 10.45 14.69 ... 992.2 996.5 1e+03
Data variables:
    z200        (tile, time, grid_yt, grid_xt) float32 dask.array<chunksize=(6, 4, 48, 48), meta=np.ndarray>
    ucomp       (tile, time, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(6, 4, 79, 48, 48), meta=np.ndarray>
    average_T1  (time) datetime64[ns] 2020-01-20 ... 2020-01-21T12:00:00
    average_T2  (time) datetime64[ns] 2020-01-20T12:00:00 ... 2020-01-22
    average_DT  (time) timedelta64[ns] 12:00:00 12:00:00 12:00:00 12:00:00
    time_bnds   (time, nv) timedelta64[ns] 0 days 00:00:00 ... 2 days 00:00:00
```

Note the first call to `fv3ds.datasets` may take a couple seconds -- it takes
some time to go through and open all the files -- but the result will be
cached, so future accesses will be fast.  For analysis on Gaea or PP/AN, this
view of the data may be enough; however, if you would like to convert the
dataset to a zarr store, one way to do so would be to simply use xarray's built
in `to_zarr` method:

```
>>> datasets["demo_ave"].to_zarr("/path/to/store/demo_ave.zarr")
```

Note for large datasets there are more efficient ways to do this in a
distributed fashion.  For that, see
[`xpartition`](https://github.com/spencerkclark/xpartition).

On PP/AN, with the tape archive, you may not want to access every tape in a
dataset at a time.  Instead you might want to read in data from a single tape.
For this you can use the `FV3Dataset.tape_to_dask` method:

```
>>> fv3ds.tape_to_dask("demo_ave")
<xarray.Dataset>
Dimensions:     (grid_xt: 48, grid_yt: 48, nv: 2, pfull: 79, phalf: 80, tile: 6, time: 4)
Coordinates:
  * time        (time) object 2020-01-20 06:00:00 ... 2020-01-21 18:00:00
  * tile        (tile) int64 0 1 2 3 4 5
  * grid_xt     (grid_xt) float64 1.0 2.0 3.0 4.0 5.0 ... 45.0 46.0 47.0 48.0
  * grid_yt     (grid_yt) float64 1.0 2.0 3.0 4.0 5.0 ... 45.0 46.0 47.0 48.0
  * pfull       (pfull) float64 4.514 8.301 12.45 16.74 ... 989.5 994.3 998.3
  * nv          (nv) float64 1.0 2.0
  * phalf       (phalf) float64 3.0 6.467 10.45 14.69 ... 992.2 996.5 1e+03
Data variables:
    z200        (tile, time, grid_yt, grid_xt) float32 dask.array<chunksize=(6, 4, 48, 48), meta=np.ndarray>
    ucomp       (tile, time, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(6, 4, 79, 48, 48), meta=np.ndarray>
    average_T1  (time) datetime64[ns] 2020-01-20 ... 2020-01-21T12:00:00
    average_T2  (time) datetime64[ns] 2020-01-20T12:00:00 ... 2020-01-22
    average_DT  (time) timedelta64[ns] 12:00:00 12:00:00 12:00:00 12:00:00
    time_bnds   (time, nv) timedelta64[ns] 0 days 00:00:00 ... 2 days 00:00:00
```

## Chunk sizes

By default, datasets will be chunked with a target chunk size of 128 MB each.
This can be configured in the `FV3Dataset` constructor using the
`target_chunk_size` argument:

```
>>> fv3ds = fv3dataset.FV3Dataset(root, "10Mi")
>>> fv3ds.tape_to_dask("demo_ave")
<xarray.Dataset>
Dimensions:     (grid_xt: 48, grid_yt: 48, nv: 2, pfull: 79, phalf: 80, tile: 6, time: 4)
Coordinates:
  * time        (time) object 2020-01-20 06:00:00 ... 2020-01-21 18:00:00
  * tile        (tile) int64 0 1 2 3 4 5
  * grid_xt     (grid_xt) float64 1.0 2.0 3.0 4.0 5.0 ... 45.0 46.0 47.0 48.0
  * grid_yt     (grid_yt) float64 1.0 2.0 3.0 4.0 5.0 ... 45.0 46.0 47.0 48.0
  * pfull       (pfull) float64 4.514 8.301 12.45 16.74 ... 989.5 994.3 998.3
  * nv          (nv) float64 1.0 2.0
  * phalf       (phalf) float64 3.0 6.467 10.45 14.69 ... 992.2 996.5 1e+03
Data variables:
    z200        (tile, time, grid_yt, grid_xt) float32 dask.array<chunksize=(6, 4, 48, 48), meta=np.ndarray>
    ucomp       (tile, time, pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(3, 4, 79, 48, 48), meta=np.ndarray>
    average_T1  (time) datetime64[ns] 2020-01-20 ... 2020-01-21T12:00:00
    average_T2  (time) datetime64[ns] 2020-01-20T12:00:00 ... 2020-01-22
    average_DT  (time) timedelta64[ns] 12:00:00 12:00:00 12:00:00 12:00:00
    time_bnds   (time, nv) timedelta64[ns] 0 days 00:00:00 ... 2 days 00:00:00
```

Here we can see that with a target chunk size of 10 MB, the chunk size of
`ucomp` along the tile dimension was cut in half.

## Installation

To install `fv3dataset`, checkout the source code from GitHub and use `pip`:

```
$ git clone https://github.com/spencerkclark/fv3dataset.git
$ cd fv3dataset
$ pip install -e .
```
