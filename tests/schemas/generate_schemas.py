import os

import synth
import numpy as np
import pandas as pd
import xarray as xr


BOUNDS_DIM = "nv"
X_DIM_FINE = "grid_xt"
Y_DIM_FINE = "grid_yt"
X_DIM_COARSE = "grid_xt_coarse"
Y_DIM_COARSE = "grid_yt_coarse"
PFULL_DIM = "pfull"
PHALF_DIM = "phalf"
PLEV_DIM = "plev"
TILES = range(1, 7)
TIME_DIM = "time"
TIME_FORMAT = "%Y%m%d%H"

SIZE_COARSE = 6
SIZE_FINE = 2 * SIZE_COARSE
SIZE_TIME = 8
FREQ_TIME = "6H"

VERTICAL_SIZES = {PFULL_DIM: 8, PHALF_DIM: 9, PLEV_DIM: 5}


def time_dimension_coordinate(size, freq, segment=0):
    result = xr.cftime_range(
        start="2000-01-01", periods=size, freq=freq, calendar="julian"
    )
    segment_length = size * pd.Timedelta(freq).to_pytimedelta()
    return result + segment * segment_length


def time_bounds_coordinate(size, freq, segment=0):
    left = time_dimension_coordinate(size, freq, segment)
    right = time_dimension_coordinate(size, freq, segment)
    right = right + pd.Timedelta(freq).to_pytimedelta()
    return np.vstack([left, right])


def horizontal_dimension_coordinate(size):
    return np.arange(1.0, size + 1.0)


def vertical_dimension_coordinate(size):
    return np.linspace(0.0, 1000.0, size)


def two_dimensional_field(name, coarse=False, time_varying=True, segment=0):
    if coarse:
        x_dim = X_DIM_COARSE
        y_dim = Y_DIM_COARSE
        horizontal_size = SIZE_COARSE
    else:
        x_dim = X_DIM_FINE
        y_dim = Y_DIM_FINE
        horizontal_size = SIZE_FINE

    if time_varying:
        sizes = (SIZE_TIME, horizontal_size, horizontal_size)
        dims = [TIME_DIM, x_dim, y_dim]
        x = horizontal_dimension_coordinate(horizontal_size)
        y = horizontal_dimension_coordinate(horizontal_size)
        time = time_dimension_coordinate(SIZE_TIME, FREQ_TIME, segment)
        coords = [time, x, y]
    else:
        sizes = (horizontal_size, horizontal_size)
        dims = [x_dim, y_dim]
        x = horizontal_dimension_coordinate(horizontal_size)
        y = horizontal_dimension_coordinate(horizontal_size)
        coords = [x, y]

    arr = np.ones(sizes)
    attrs = {"units": f"units of {name}", "long_name": f"long_name of {name}"}
    return xr.DataArray(arr, dims=dims, coords=coords, name=name, attrs=attrs)


def three_dimensional_field(
    name, vertical_dim, coarse=False, time_varying=True, segment=0
):
    dataarrays = []
    size = VERTICAL_SIZES[vertical_dim]
    for _ in range(size):
        dataarray = two_dimensional_field(name, coarse, time_varying, segment)
        dataarrays.append(dataarray)
    vertical_coord = vertical_dimension_coordinate(size)
    dim = pd.Index(vertical_coord, name=vertical_dim)
    return xr.concat(dataarrays, dim=dim)


def time_bounds_field(name, segment=0):
    time = time_dimension_coordinate(SIZE_TIME, FREQ_TIME, segment)
    time_bounds = time_bounds_coordinate(SIZE_TIME, FREQ_TIME, segment)
    bounds = [1.0, 2.0]
    dims = [BOUNDS_DIM, TIME_DIM]
    return xr.DataArray(time_bounds, dims=dims, coords=[bounds, time], name=name)


tape_fine = {
    "static_2d": {"dimension": 2, "coarse": False, "time_varying": False},
    "time_bounds": {"dimension": "time_bounds"},
    "time_varying_2d": {"dimension": 2, "coarse": False, "time_varying": True},
    "static_pfull_3d": {
        "dimension": 3,
        "coarse": False,
        "time_varying": False,
        "vertical_dim": PFULL_DIM,
    },
    "static_phalf_3d": {
        "dimension": 3,
        "coarse": False,
        "time_varying": False,
        "vertical_dim": PHALF_DIM,
    },
    "static_plev_3d": {
        "dimension": 3,
        "coarse": False,
        "time_varying": False,
        "vertical_dim": PLEV_DIM,
    },
    "time_varying_pfull_3d": {
        "dimension": 3,
        "coarse": False,
        "time_varying": True,
        "vertical_dim": PFULL_DIM,
    },
    "time_varying_phalf_3d": {
        "dimension": 3,
        "coarse": False,
        "time_varying": True,
        "vertical_dim": PHALF_DIM,
    },
    "time_varying_plev_3d": {
        "dimension": 3,
        "coarse": False,
        "time_varying": True,
        "vertical_dim": PLEV_DIM,
    },
}

tape_coarse = {
    "static_2d": {"dimension": 2, "coarse": True, "time_varying": False},
    "time_bounds": {"dimension": "time_bounds"},
    "time_varying_2d": {"dimension": 2, "coarse": True, "time_varying": True},
    "static_pfull_3d": {
        "dimension": 3,
        "coarse": True,
        "time_varying": False,
        "vertical_dim": PFULL_DIM,
    },
    "static_phalf_3d": {
        "dimension": 3,
        "coarse": True,
        "time_varying": False,
        "vertical_dim": PHALF_DIM,
    },
    "static_plev_3d": {
        "dimension": 3,
        "coarse": True,
        "time_varying": False,
        "vertical_dim": PLEV_DIM,
    },
    "time_varying_pfull_3d": {
        "dimension": 3,
        "coarse": True,
        "time_varying": True,
        "vertical_dim": PFULL_DIM,
    },
    "time_varying_phalf_3d": {
        "dimension": 3,
        "coarse": True,
        "time_varying": True,
        "vertical_dim": PHALF_DIM,
    },
    "time_varying_plev_3d": {
        "dimension": 3,
        "coarse": True,
        "time_varying": True,
        "vertical_dim": PLEV_DIM,
    },
}

tape_static = {
    "static_2d": {"dimension": 2, "coarse": True, "time_varying": False},
    "static_pfull_3d": {
        "dimension": 3,
        "coarse": True,
        "time_varying": False,
        "vertical_dim": PFULL_DIM,
    },
    "static_phalf_3d": {
        "dimension": 3,
        "coarse": True,
        "time_varying": False,
        "vertical_dim": PHALF_DIM,
    },
    "static_plev_3d": {
        "dimension": 3,
        "coarse": True,
        "time_varying": False,
        "vertical_dim": PLEV_DIM,
    },
}


def construct_dataset_template(parameters, segment=0):
    dataarrays = []
    for name, specifications in parameters.items():
        if specifications["dimension"] == 2:
            coarse = specifications["coarse"]
            time_varying = specifications["time_varying"]
            dataarray = two_dimensional_field(name, coarse, time_varying, segment)
        elif specifications["dimension"] == 3:
            coarse = specifications["coarse"]
            time_varying = specifications["time_varying"]
            vertical_dim = specifications["vertical_dim"]
            dataarray = three_dimensional_field(
                name, vertical_dim, coarse, time_varying, segment
            )
        elif specifications["dimension"] == "time_bounds":
            dataarray = time_bounds_field(name, segment)
        dataarrays.append(dataarray)
    ds = xr.merge(dataarrays)
    ds.attrs = {}
    return ds.assign_attrs(foo="bar")


parameter_sets = {"fine": tape_fine, "coarse": tape_coarse, "static": tape_static}
datasets = {}
for tape, parameters in parameter_sets.items():
    datasets[tape] = {}
    for segment in range(2):
        time = time_dimension_coordinate(SIZE_TIME, FREQ_TIME, segment)
        timestamp = time.strftime(TIME_FORMAT)[0]
        ds = construct_dataset_template(parameters, segment=segment)
        datasets[tape][timestamp] = ds

schemas = {}
for tape in datasets:
    schemas[tape] = {}
    for timestamp in datasets[tape]:
        ds = datasets[tape][timestamp]
        schemas[tape][timestamp] = synth.read_schema_from_dataset(ds)


for tape in schemas:
    for timestamp in schemas[tape]:
        schema = schemas[tape][timestamp]
        root = os.path.join("history", timestamp)
        os.makedirs(root, exist_ok=True)
        for tile in TILES:
            filename = os.path.join(root, f"{tape}.tile{tile}.nc.json")
            with open(filename, "w") as file:
                synth.dump(schema, file)
