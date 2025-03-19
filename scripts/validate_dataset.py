"""
A script that checks an input dataset (netCDF file) for adherence to ILAMB standards.
The netCDF can contain site data or gridded data.
"""

import sys
from typing import Literal

import cftime
import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, field_validator


def get_dim_name(
    dset: xr.Dataset | xr.DataArray,
    dim: Literal["time", "lat", "lon", "depth", "site"],
) -> str:
    dim_names = {
        "time": ["time"],
        "lat": ["lat", "latitude", "Latitude", "y", "lat_"],
        "lon": ["lon", "longitude", "Longitude", "x", "lon_"],
        "depth": ["depth"],
    }
    # Assumption: the 'site' dimension is what is left over after all others are removed
    if dim == "site":
        try:
            get_dim_name(dset, "lat")
            get_dim_name(dset, "lon")
            # raise NoSiteDimension("Dataset/dataarray is spatial")
        except KeyError:
            pass
        possible_names = list(
            set(dset.dims) - set([d for _, dims in dim_names.items() for d in dims])
        )
        if len(possible_names) == 1:
            return possible_names[0]
        msg = f"Ambiguity in locating a site dimension, found: {possible_names}"
        # raise NoSiteDimension(msg)
    possible_names = dim_names[dim]
    dim_name = set(dset.dims).intersection(possible_names)
    if len(dim_name) != 1:
        msg = f"{dim} dimension not found: {dset.dims} "
        msg += f"not in [{','.join(possible_names)}]"
        raise KeyError(msg)
    return str(dim_name.pop())


def is_spatial(da: xr.DataArray) -> bool:
    try:
        get_dim_name(da, "lat")
        get_dim_name(da, "lon")
        return True
    except KeyError:
        pass
    return False


# spatial validator
class ILAMBDataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    ds: xr.Dataset

    @field_validator("ds")
    @classmethod
    def check_vars(cls, ds: xr.Dataset) -> xr.Dataset:
        # Check that the dataset has at least one variable but not more than 2
        if not ds.data_vars:
            raise ValueError(
                "Dataset does not have any data variables. An example data variable is 'cSoil'."
            )
        if len(ds.data_vars) >= 3:
            raise ValueError(
                f"Dataset has too many data variables {ds.data_vars}. The measurement and the uncertainty are the only expected data variables. There should be one netCDF file per data variable if a dataset has multiple data variables."
            )
        return ds

    @field_validator("ds")
    @classmethod
    def global_attrs(cls, ds: xr.Dataset) -> xr.Dataset:
        # Check that the dataset has the required global attribute keys
        missing = set(
            [
                "title",
                "source_version_number",
                "institution",
                "source",
                "history",
                "references",
                "Conventions",
            ]
        ) - set(ds.attrs.keys())
        if missing:
            raise ValueError(
                f"Dataset does not properly encode global attributes, {missing=}"
            )
        return ds

    @field_validator("ds")
    @classmethod
    def time_dim(cls, ds: xr.Dataset) -> xr.Dataset:
        # Check that the dataset has a properly set-up time dimension
        dimensions = ds.dims
        time_dim_present = "time" in dimensions
        time_var = ds["time"]
        time_attrs = time_var.attrs
        if not time_dim_present:
            raise ValueError(f"Dataset does not have a time dimension, {dimensions=}")
        # Check if time values are decoded as datetime objects
        time_dtype = type(time_var.values[0])
        if not (
            np.issubdtype(time_var.values.dtype, np.datetime64)
            or isinstance(time_var.values[0], cftime.datetime)
        ):
            raise TypeError(
                f"Time values are not properly decoded as datetime objects: {time_dtype=}"
            )

        # Check for time attributes: axis, long_name
        missing = set(["axis", "long_name", "standard_name"]) - set(time_attrs)
        if missing:
            raise ValueError(
                f"Dataset is missing time-specific attributes, {missing=}. We recommend axis=T"
            )

        # Check that time units are encoded (and formatted correctly)
        time_encoding = time_var.encoding
        if "units" not in time_encoding or "since" not in time_encoding["units"]:
            raise ValueError(
                f"Time encoding is missing or incorrect, {time_encoding=}. Expected 'days since YYYY:MM:DD HH:MM'"
            )

        # Check if time calendar is encoded
        if "calendar" in time_encoding:
            valid_calendars = [
                "standard",
                "gregorian",
                "proleptic_gregorian",
                "noleap",
                "all_leap",
                "360_day",
                "julian",
            ]

            if time_encoding["calendar"] not in valid_calendars:
                # Check for explicitly defined calendar attributes
                if "month_lengths" in time_attrs:
                    # Validate month_lengths
                    month_lengths = time_attrs["month_lengths"]
                    if len(month_lengths) != 12 or not all(
                        isinstance(m, (int, np.integer)) for m in month_lengths
                    ):
                        raise ValueError(
                            "month_lengths must be a list of 12 integer values."
                        )

                    # Validate leap year settings if present
                    if "leap_year" in time_attrs:
                        leap_year = time_attrs["leap_year"]
                        if not isinstance(leap_year, (int, np.integer)):
                            raise ValueError("leap_year must be an integer.")

                        if "leap_month" in time_attrs:
                            leap_month = time_attrs["leap_month"]
                            if not (1 <= leap_month <= 12):
                                raise ValueError("leap_month must be between 1 and 12.")
                else:
                    raise ValueError(
                        f"Unrecognized calendar '{time_encoding['calendar']}' and no explicit month_lengths provided."
                    )
        else:
            raise ValueError("Calendar attribute is missing from the time encoding.")

        # Check if bounds are encoded
        time_bounds_name = time_attrs["bounds"]
        if time_bounds_name not in ds:
            raise ValueError(
                f"Time bounds variable '{time_bounds_name=}' is missing from dataset. Expected 'time_bounds'"
            )

        # Validate time_bounds structure
        time_bounds = ds[time_bounds_name]
        if len(time_bounds.dims) != 2 or time_bounds.dims[0] != "time":
            raise ValueError(
                f"Time bounds, '{time_bounds_name=}', has incorrect dimensions, {time_bounds.dims}."
                "Expected two dimensions: ('time', <second_dimension>)."
            )

        # Check that the second dimension length is 2 (indicating time bounds)
        if time_bounds.shape[1] != 2:
            raise ValueError(
                f"Time bounds '{time_bounds_name}' has incorrect shape {time_bounds.shape}. "
                "The second dimension should have length 2 to represent time bounds."
            )

        # Check for the correct 'long_name' attribute
        if (
            "long_name" not in time_bounds.attrs
            or time_bounds.attrs["long_name"] != "time_bounds"
        ):
            raise ValueError(
                f"Time bounds '{time_bounds_name}' is missing its 'long_name':'time_bounds' attribute."
            )

        return ds

    @field_validator("ds")
    @classmethod
    def lat_dim(cls, ds: xr.Dataset) -> xr.Dataset:
        # Check that the dataset has a properly set-up latitude dimension
        lat_names = {"lat", "latitude", "y"}
        dims = ds.dims
        dims_lower = {
            dim.lower(): dim for dim in dims
        }  # Map lowercased dims to original names
        lat_name = next(
            (
                dims_lower[name.lower()]
                for name in lat_names
                if name.lower() in dims_lower
            ),
            None,
        )

        # check that one of the accepted latitutde long_names exists
        if not lat_name:
            raise ValueError(
                f"Dataset does not have an accepted latitude dimension, {dims=}. Expected one of {lat_names} (case insensitive)."
            )

        # check that axis and long_name attributes are present
        else:
            lat_var = ds[lat_name]
            lat_attrs = lat_var.attrs
            missing = set(["axis", "long_name"]) - set(lat_attrs)
            if missing:
                raise ValueError(
                    f"Dataset is missing latitude-specific attributes, {missing=}"
                )

        return ds

    @field_validator("ds")
    @classmethod
    def lon_dim(cls, ds: xr.Dataset) -> xr.Dataset:
        # Check that the dataset has a properly set-up latitude dimension
        lon_names = {"lon", "longitude", "x"}
        dims = ds.dims
        dims_lower = {
            dim.lower(): dim for dim in dims
        }  # Map lowercased dims to original names
        lon_name = next(
            (
                dims_lower[name.lower()]
                for name in lon_names
                if name.lower() in dims_lower
            ),
            None,
        )

        # check that one of the accepted latitutde long_names exists
        if not lon_name:
            raise ValueError(
                f"Dataset does not have an accepted latitude dimension, {dims=}. Expected one of {lon_names} (case insensitive)."
            )

        # check that axis and long_name attributes are present
        else:
            lon_var = ds[lon_name]
            lon_attrs = lon_var.attrs
            missing = set(["axis", "long_name"]) - set(lon_attrs)
            if missing:
                raise ValueError(
                    f"Dataset is missing latitude-specific attributes, {missing=}"
                )

        return ds


if __name__ == "__main__":
    dset = xr.open_dataset(sys.argv[1])
    test = ILAMBDataset(ds=dset)
