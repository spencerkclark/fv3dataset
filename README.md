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
>>> datasets["demo_ave"]
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

## Installation

To install `fv3dataset`, checkout the source code from GitHub and use `pip`:

```
$ git clone https://github.com/spencerkclark/fv3dataset.git
$ cd fv3dataset
$ pip install -e .
```
