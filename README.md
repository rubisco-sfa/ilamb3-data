# ilamb3-data

This is a temporary space while we reorganize ourselves. This will eventually
move to replace the content in ILAMB-Data. For the moment we will use the README
to communicate about the planning of this repo.

## uv

I recommend you try using
[uv](https://docs.astral.sh/uv/getting-started/installation/) to manage
dependencies. You can keep conda around if you are using that. Just first issue
`deactivate` commands to back out of the conda environments. 

- Each code project I have now has its own environment that is stored (not
  committed) in a `.venv` folder. Once you clone this repo, you can navigate
  inside it and issue a `uv sync` to build the environment. 
- You don't have to activate the environment if you don't want to. You can issue
  for example `uv run python scripts/check` and `uv` will run your command with
  the environment. 
- You can however activate the environment if you wish with `source
  .venv/bin/activate` and deactivate the same as with conda.
- If you want to add a dependency to our project, then type `uv add
  PACKAGE_NAME` and it will automatically place it in the pyproject.toml file. 
- `uv` is 10-100x faster than pip


## Adding a reference dataset

This is how I propose adding a dataset would work:

- Add an appropriate directory to the `data` subdirectory, where the resulting
  new dataset will take the form `data/SOURCE/VARNAME.nc`.
    - If the data provider provides a name for the data (WECANN, Fluxnet2015,
      CRU4.02), then please use this as the value for `SOURCE`.
    - If the variable(s) directly correspond to a CMOR variable, please use that
      as `VARNAME`. 
    - If the variable(s) do not correspond, please select a descriptive name as
      `VARNAME`.
    - Encode `VARNAME` as the main variable in the file. 
- Create a script to encode the data and run it to generate the new datafile,
  `NEWFILE`.
    - We will need documentation for how to do this. Start by copying my
      tutorials over and then extending.
    - One key item is that downloading should be automatic if possible.
    - We also need to collect the data license. Some organizations need this to
      use ILAMB.
- Run `python scripts/check.py NEWFILE` until the validation no longer
  complains.
- Run `sha1sum NEWFILE` and add it to `registry/data.txt`.
    - We will need to settle on the paths
- Add an entry into `staging.cfg` detailing how you intend your `NEWFILE` to be
  run in ILAMB.
- Move `NEWFILE` to the appropriate place on www.ilamb.org if you have access.
    - If a user does not have access, we will have them ping a group of us in
      the PR (next step) and we will have to clone the PR and render the data
      and upload it ourselves.
- Submit a PR with (1) your script to generate the `NEWFILE` but not the file
  iteself (2) the addition to `staging.cfg` and (3) addition to the
  `registry/data.txt`.
    - Our CI would download your file using the addition to the registry and
      then ensure that `scripts/check.py` does in fact not complain. 

## CI actions to implement

- Given the PR, extract the registry entry being added and have the CI pull it
  down (automatically verifying the sha1sum) and then also running the check
  script to validate.
    - It should also make sure that something was added to staging.cfg or at
      least warn if it wasn't.
- Given the PR, find a way to regenerate website listing of data.
    - It may be that we go to the website repo and create an action to daily
      rebuild the list of datasets from items in the registry in this repo.