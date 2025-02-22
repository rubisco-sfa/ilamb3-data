"""
A script that checks an input dataset (netCDF file) for adherence to ILAMB standards.
"""

import sys

import xarray as xr
from pydantic import BaseModel, ConfigDict, field_validator


class ILAMBDataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    ds: xr.Dataset

    @field_validator("ds")
    @classmethod
    def global_attrs(cls, ds: xr.Dataset) -> xr.Dataset:
        # Check that the dataset has at least the required attribute keys
        missing = set(
            ["title", "version", "institutions", "source", "history", "references"]
        ) - set(ds.attrs.keys())
        if missing:
            raise ValueError(
                f"Dataset does not properly encode global attributes, {missing=}"
            )
        return ds


if __name__ == "__main__":
    dset = xr.open_dataset(sys.argv[1])
    test = ILAMBDataset(ds=dset)
