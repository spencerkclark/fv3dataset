import os

import pytest
import synth

from pathlib import Path

from fv3dataset import HistoryDataset


TIME_DIM = "time"
LOCAL_DIRECTORY = Path(os.path.dirname(__file__))
HISTORY_SCHEMAS = LOCAL_DIRECTORY / "schemas" / "history"
HISTORY_TIMESTAMPS = ["2000010100", "2000010300"]
TAPES = ["fine", "coarse", "static"]


def generate_synthetic_history_dataset(root):
    root = Path(root)
    for timestamp in os.listdir(HISTORY_SCHEMAS):
        path = HISTORY_SCHEMAS / timestamp
        directory_schema = synth.load_directory_schema(path)
        destination = root / "history" / timestamp
        os.makedirs(destination, exist_ok=True)
        synth.write_directory_schema(destination, directory_schema)
    return root / "history"


@pytest.fixture(scope="module")
def synthetic_history_dataset(tmpdir_factory):
    root = tmpdir_factory.mktemp("history_dataset")
    return generate_synthetic_history_dataset(root)


@pytest.fixture(params=TAPES, scope="module")
def tape(request):
    return request.param


@pytest.fixture(scope="module")
def history_dataset(tape, synthetic_history_dataset):
    directories = []
    for timestamp in HISTORY_TIMESTAMPS:
        directory = synthetic_history_dataset / timestamp
        directories.append(directory)
    return HistoryDataset(tape, directories, target_chunk_size="0.01Mi")


def test_history_dataset_tape(tape, history_dataset):
    assert history_dataset.tape == tape


def test_history_dataset_dims(history_dataset):
    if history_dataset.tape == "fine":
        expected = {
            "grid_xt",
            "grid_yt",
            "tile",
            "pfull",
            "phalf",
            "plev",
            "time",
            "nv",
        }
    elif history_dataset.tape == "coarse":
        expected = {
            "grid_xt_coarse",
            "grid_yt_coarse",
            "tile",
            "pfull",
            "phalf",
            "plev",
            "time",
            "nv",
        }
    elif history_dataset.tape == "static":
        expected = {
            "grid_xt_coarse",
            "grid_yt_coarse",
            "tile",
            "pfull",
            "phalf",
            "plev",
        }
    result = history_dataset.dims
    assert result == expected


def test_history_dataset_combining_dims(history_dataset):
    expected = {"time", "tile"}
    result = history_dataset.combining_dims
    assert result == expected


def test_history_dataset_static_dims(history_dataset):
    if history_dataset.tape == "fine":
        expected = {"grid_xt", "grid_yt", "pfull", "phalf", "plev", "nv"}
    elif history_dataset.tape == "coarse":
        expected = {"grid_xt_coarse", "grid_yt_coarse", "pfull", "phalf", "plev", "nv"}
    elif history_dataset.tape == "static":
        expected = {"grid_xt_coarse", "grid_yt_coarse", "pfull", "phalf", "plev"}
    result = history_dataset.static_dims
    assert result == expected


def test_history_dataset_chunked_variables_2d(history_dataset):
    if history_dataset.tape in ["fine", "coarse"]:
        expected = ["time_varying_2d"]
    elif history_dataset.tape == "static":
        expected = []
    result = history_dataset.chunked_variables_2d
    assert result == expected


def test_history_dataset_chunked_variables_3d(history_dataset):
    if history_dataset.tape in ["fine", "coarse"]:
        expected = [
            "time_varying_pfull_3d",
            "time_varying_phalf_3d",
            "time_varying_plev_3d",
        ]
    elif history_dataset.tape == "static":
        expected = []
    result = history_dataset.chunked_variables_3d
    assert result == expected


def test_history_dataset_unchunked_variables(history_dataset):
    if history_dataset.tape == "fine":
        expected = [
            "static_2d",
            "time_bounds",
            "static_pfull_3d",
            "static_phalf_3d",
            "static_plev_3d",
            "grid_xt",
            "grid_yt",
            "nv",
            "time",
            "pfull",
            "phalf",
            "plev",
        ]
    elif history_dataset.tape == "coarse":
        expected = [
            "static_2d",
            "time_bounds",
            "static_pfull_3d",
            "static_phalf_3d",
            "static_plev_3d",
            "grid_xt_coarse",
            "grid_yt_coarse",
            "nv",
            "time",
            "pfull",
            "phalf",
            "plev",
        ]
    elif history_dataset.tape == "static":
        expected = [
            "static_2d",
            "static_pfull_3d",
            "static_phalf_3d",
            "static_plev_3d",
            "grid_xt_coarse",
            "grid_yt_coarse",
            "pfull",
            "phalf",
            "plev",
        ]
    result = history_dataset.unchunked_variables
    assert result == expected


def test_history_dataset_to_dask(history_dataset):
    ds = history_dataset.to_dask()

    if TIME_DIM in ds.dims:
        assert ds.sizes[TIME_DIM] == 16

    if history_dataset.tape == "static":
        assert TIME_DIM not in ds.dims

    for name, da in ds.data_vars.items():
        if name == "time_bounds":
            assert "tile" not in da.dims
        else:
            assert "tile" in da.dims

    # TODO: test the chunk sizes are as expected.
