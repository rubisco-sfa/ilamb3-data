import re
import sys

import xarray as xr
from validate_dataset import ILAMBDataset

from ilamb3_data import create_registry

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No registry entries have changed.")

    # Parse out the lines in the registry that have been added
    content = " ".join(sys.argv[1:])
    match = re.search(r"registry/data.txt:\s\[([\d,\s]*)\]", content)
    if not match:
        if "data/" in content:
            raise ValueError(
                "It appears you are adding a new dataset, but we do not see a change in the registry. Did you add the output of `sha1sum FILE.nc` to `registry/data.txt`?"
            )
        print("No registry entries have changed.")
        sys.exit(0)
    lines = [int(i.strip()) - 1 for i in match.group(1).split(",")]

    # Open the registry and reduce by just the lines that were changed
    with open("registry/data.txt") as fin:
        reg_lines = fin.readlines()
    reg_lines = [reg_lines[i] for i in lines]

    # Download and validate these entries
    registry = create_registry("registry/data.txt")
    for line in reg_lines:
        path, _ = line.split()
        ds = xr.open_dataset(registry.fetch(path))
        ILAMBDataset(ds=ds)
