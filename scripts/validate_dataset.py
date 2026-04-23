"""
A script that checks an input dataset (netCDF file) for adherence to ILAMB standards.
The dataset can contain site data or gridded data.
"""

import sys
from typing import Callable

import xarray as xr
from pydantic import BaseModel, ConfigDict, field_validator

########################################################################################
# First, we need to know if we can validate at all by reading the headers of the NetCDF
# and then checking the filename
########################################################################################


def check_netcdf(filename: str) -> list[str]:
    """Check that we can validate the dataset according to ILAMB or obs4MIPs standards."""
    # Basically, don't bother to actually open the whole file if we can't validate it

    # Split file name by underscores; if it starts with ILAMB or obs4MIPs, we can validate
    # Extract global attributes from the dataset and put them in a dict
    # attrs = {}
    # If ILAMB:
    # Ensure header attrs match what is in the file name
    # try:
    #    filename = "{activity_id}_{institution_id}_{source_id}_{frequency}_{variable_id}_{grid_label}_{version}.nc".format(**attrs)
    # except KeyError as e:
    #    raise ValueError(f"Missing required global attribute {e} for ILAMB filename validation.")
    # If obs4MIPs:
    # Ensure header attrs match what is in the file name
    # try:
    #    filename = "whatever the obs4MIPs filename convention is".format(**attrs)
    # except KeyError as e:
    #    raise ValueError(f"Missing required global attribute {e} for obs4MIPs filename validation.")
    # If neither, fail

    # Return grid_label from global attrs
    return list()


########################################################################################
# Then, we need to know if it is gridded or site data
########################################################################################


def is_gridded(grid_label: str) -> bool:
    """Check if the dataset is gridded by looking for lat and lon dimensions."""
    if grid_label != "site":
        return True
    else:
        return False


def is_site(grid_label: str) -> bool:
    """Check if the dataset is site by looking for a site dimension."""
    if grid_label == "site":
        return True
    else:
        return False


def check_attrs(cls, ds: xr.Dataset, unit_checker: Callable) -> xr.Dataset:
    """General attribute and encoding checker."""
    # we will require AT MINIMUM "long_name", "units", and "standard_name"
    # if time, lat, lon, or depth, also require "axis"
    #   Attr: units
    # if var is time, lat, lon, depth, or site:
    #   use unit_checker
    # if var is not time, lat, lon, depth, or site:
    #   The units attribute is required for all variables that represent dimensional/dimensionless quantities (except for boundary variables)
    #   The value of the units attribute has to be a string that can be recognized by cf-units>=3.3.0
    #   The canonical unit (see also Section 3.3, "Standard Name") for dimensionless quantities that represent fractions, or parts of a whole, is 1
    #   Otheriwse, "" is acceptable for dimensionless quantities that do not represent fractions, or parts of a whole
    # Attr: long_name
    #   descriptive free form text
    # Attr: standard_name
    #   starts with a letter, all lowercase, separated by underscores
    # Attr: ancillary_variables
    #   optional (check only if it exists)
    #   The attribute ancillary_variables is used to express when one data variable provides metadata about the individual values of another data variable. It is a string attribute whose value is a blank separated list of variable names
    #   An ancillary variable will often have the standard name of the data variable which points to it including a modifier (Appendix C, Standard Name Modifiers) to indicate the relationship
    #   The ancillary variables listed should also exist as variables
    #   the standard_name of the ancillary variables shoould be the standard_name of the parent variable with a " modifier" to indicate the relationship
    #   e.g., q.attrs = {"standard_name": "specific_humidity", "units": "g/g", "ancillary_variables": "q_error_limit"}
    #   q_error_limit.attrs = {"standard_name": "specific_humidity standard_error", "units": "g/g"}
    # Attr: flag_
    #   optional (check only if it exists)
    #   The flag_ prefix is used to indicate that an attr is a flag of some kind; if flag_ is present, at least flag_values and flag_meanings should be present, but other additional flag_ are fine
    #   The flag_values attribute is the same type as the variable to which it is attached, and contains a list of the possible flag values
    #   The flag_meanings attribute is a string whose value is a blank separated list of descriptive words or phrases, one for each flag value. Each word or phrase should consist of characters from the alphanumeric set and the following five: '_', '-', '.', '+', '@'. If multi-word phrases are used to describe the flag values, then the words within a phrase should be connected with underscores.
    # Attr: valid_range, valid_min, valid_max
    #   optional (check only if it exists)
    #   must be of the same data type as the packed data (numeric)
    #   valid_range: missing data is all values outside of the valid_range
    # Attr: scale_factor, add_offset
    #   optional (check only if it exists)
    #   When packed data is written, the scale_factor and add_offset attributes must be of the same type as the unpacked data, which must be either float or double
    # Encoding: _FillValue (or missing_value; replace with _FillValue because it's deprecated)
    #   optional (check only if it exists)
    #   must be of the same data type as the packed data
    #   The _FillValue attribute is used to identify a single value that is not a valid data value
    #   if dtype of variable is one of these keys, they _FillValue should be the correspoding value (NUG conventions): _FILL_VALUES = {np.dtype("S1"): np.bytes_(b"\x00"), np.int8: np.int8(-127), np.int16: np.int16(-32767), np.int32: np.int32(2147483647), np.float32: np.float32(1.0e20), np.float64: np.float64(1.0e20),}
    #   The _FillValue should be outside the range specified by valid_range (if used) for a variable
    # Any other attrs that haven't been validated should raise a warning that we can't validate them, but they should not cause the dataset to be rejected
    # Any other encoding that hasn't been validated should raise a warning that we can't validate it, but it should not cause the dataset to be rejected
    return ds


class ILAMBDataset(BaseModel):
    """A dataset in ILAMB that is either site or gridded."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    ds: xr.Dataset

    # # Check if we can validate
    # check_netcdf()

    # # Check if gridded or site (simple check)
    # is_gridded()
    # is_site()

    # See if gridded or site designation matches what is expected in the file structure
    class SiteDataset(BaseModel):
        """A dataset containing site data, with at least a site dimension."""

    # site dimension is an existing dim (with coordinates) that isn't time
    # lat/lon should not be dimensions, but should be variables with the site dimension and attributes that indicate they are lat/lon (e.g., standard_name, units, etc.)
    # the main var (has the site and optional time dimension)
    # should not have lat_bnds or lon_bnds because that doesn't make sense for point data

    model_config = ConfigDict(arbitrary_types_allowed=True)
    ds: xr.Dataset

    @field_validator("ds")
    @classmethod
    def time_dim(cls, ds: xr.Dataset) -> xr.Dataset:
        """If time dimension is present, check that attributes and encoding."""

        def _time_unit_checker(cls, ds: xr.Dataset) -> xr.Dataset:
            """Time attribute and encoding checker."""
            # The reference datetime string (appearing after the identifier since) is required. Its format is y-m-d [H:M:S[ T]], where […​] indicates an optional element, as follows: y is year, m month, d day, H hour and M minute, which are all integers of one or more digits, S is second, which may be integer or floating point, T is the time zone offset. T is not a time zone name or acronym; it is an interval of time.
            # The default for time zone offset T is zero, which may also be explicitly indicated in any of the numeric formats for T defined below, or by the letter Z, sometimes referred to as "Zulu Time".
            # A non-zero offset is not allowed in the utc and tai calendars
            # units = "days since 1990-1-1" means the same as units = "days since 1990-1-1 00:00:00Z"
            # Other than Z for zero offset, the time zone offset T must be in one of the following signed numeric formats, where ± stands for the positive sign + or the negative sign -, and n stands for a single digit. ±n or ±nn, the hour alone, of one or two digits e.g. -6, +2, +11. This format is sufficient for many time zones. Zero hours must have a positive sign e.g. +0, which means the same as Z. ±n:nn or ±nn:nn, the hour and minute, separated by a colon :, where the hour is one or two digits, and the minute is two digits e.g. -10:00, -3:30, +5:30, +12:45
            # Encoding: calendar
            # required if var is time
            return ds

        # Check if time dimension is present (can be written any way, so we will check a few we expect to see)
        ds = check_attrs(cls, ds, _time_unit_checker)
        # The interpretation of the numeric values as dates and times depends on both the units attribute and the calendar, specified by the calendar attribute. Given the units and calendar, a time coordinate value and its corresponding representation as a date and time are interconvertible
        # Calendars: standard (0001-01-01 to +inf), julian (0001-01-01 to +inf), proleptic_gregorian (-inf to +inf), no_leap/365_day (-inf to +inf), all_leap/366_day (-inf to +inf), 360_day (-inf to +inf), utc (1972-01-01 to today), tai (1958-01-01 to +inf)
        # The calendar attribute may be set to none in climate experiments that simulate a fixed time of year. The time of year is indicated by the date in the reference time of the units attribute
        # If calendar is none of those, it could be explicitly defined. So, check if month_lengths are present (and ensure it is a vector of size 12), and if leap years are included, check for leap_year and leap_month (A value in the range 1-12)
        return ds

    @field_validator("ds")
    @classmethod
    def lat_dim(cls, ds: xr.Dataset) -> xr.Dataset:
        """If latitude dimension is present, check that attributes and encoding."""

        def _lat_unit_checker(cls, ds: xr.Dataset) -> xr.Dataset:
            """Latitude attribute and encoding checker."""
            # unit options: degrees_north, degree_north, degree_N, degrees_N, degreeN, and degreesN
            # latitude with respect to a rotated pole should be given units of degrees
            return ds

        # Check if latitude dimension is present (can be written any way, so we will check a few we expect to see)
        ds = check_attrs(cls, ds, _lat_unit_checker)
        return ds

    @field_validator("ds")
    @classmethod
    def lon_dim(cls, ds: xr.Dataset) -> xr.Dataset:
        """If longitude dimension is present, check that attributes and encoding."""

        def _lon_unit_checker(cls, ds: xr.Dataset) -> xr.Dataset:
            """Longitude attribute and encoding checker."""
            # unit options: degrees_east, degree_east, degree_E, degrees_E, degreeE, and degreesE
            # the determination that a coordinate is a longitude type should be done via a string match between the given unit and one of the acceptable forms of degrees_east
            # Coordinates of longitude with respect to a rotated pole should be given units of degrees
            return ds

        # Check if longitude dimension is present (can be written any way, so we will check a few we expect to see)
        ds = check_attrs(cls, ds, _lon_unit_checker)
        return ds

    @field_validator("ds")
    @classmethod
    def depth_dim(cls, ds: xr.Dataset) -> xr.Dataset | None:
        """If depth dimension is present, check that attributes and encoding."""

        def _depth_unit_checker(cls, ds: xr.Dataset) -> xr.Dataset:
            """Depth attribute and encoding checker."""
            # Dimensional vertical coord: Variables representing dimensional vertical coordinates for depth or height must always explicitly include the units attribute. The acceptable units for a vertical (depth or height) coordinate variable must a UDUNITS [UDUNITS] representation of one of the following: units of pressure. For vertical axes the most commonly used of these include bar, millibar, decibar, atmosphere (atm), pascal (Pa), and hPa. units of length. For vertical axes the most commonly used of these include meter (metre, m), and kilometer (km). other units that may under certain circumstances reference vertical position such as units of density or temperature. Plural forms are also acceptable.
            # attribute positive as defined in the COARDS standard is required if the vertical axis units are not a valid unit of pressure (as determined by the UDUNITS package [UDUNITS]) — otherwise its inclusion is optional. The positive attribute may have the value up or down (case insensitive)
            # If, on the other hand, the depth of 1000 meters were represented as -1000 then the value of the positive attribute would have been up. If the units attribute value is a valid pressure unit the default value of the positive attribute is down.
            # If both positive and standard_name are provided, it is recommended that they should be consistent. For instance, if a depth of 1000 metres is represented by -1000 and positive is up, it would be inconsistent to give the standard_name as depth, whose definition (vertical distance below the surface) implies positive down. If an application detects such an inconsistency, the user should be warned, and the positive attribute should be used to determine the sign convention.
            return ds

        # Check if depth dimension is present (can be written any way, so we will check a few we expect to see)
        ds = check_attrs(cls, ds, _depth_unit_checker)
        return ds

    @field_validator("ds")
    @classmethod
    def site_dim(cls, ds: xr.Dataset) -> xr.Dataset | None:
        """If site dimension is present, check that attributes and encoding."""
        # Check if site dimension is present (can be written any way, so we will check a few we expect to see)

        def _site_unit_checker(cls, ds: xr.Dataset) -> xr.Dataset:
            """Site attribute and encoding checker."""
            # site should not have units
            return xr.Dataset()

        ds = check_attrs(cls, ds, _site_unit_checker)
        return ds

    @field_validator("ds")
    @classmethod
    def global_attrs_ods26(cls, ds: xr.Dataset) -> xr.Dataset:
        """If using ODS 2.6 global attributes, check that they are present and correct."""
        return ds

    @field_validator("ds")
    @classmethod
    def global_attrs_ilamb3(cls, ds: xr.Dataset) -> xr.Dataset:
        """If using ILAMB 3 global attributes, check that they are present and correct."""
        return ds

    @field_validator("ds")
    @classmethod
    def var(cls, ds: xr.Dataset) -> xr.Dataset:
        """Check that variables have the required attributes and encoding."""
        # Data variables must be one of the following data types: string, char, byte, unsigned byte, short, unsigned short, int, unsigned int, int64, unsigned int64, float or real, and double
        return ds


if __name__ == "__main__":
    dset = xr.open_dataset(sys.argv[1])
    test = ILAMBDataset(ds=dset)
