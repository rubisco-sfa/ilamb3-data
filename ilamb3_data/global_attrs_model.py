from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator
from typing import Optional


class Dataset(BaseModel):
    source: str
    source_id: str
    source_label: str
    source_version_number: str
    source_type: str
    source_data_url: HttpUrl
    realm: str
    product: str
    frequency: str
    grid: str
    grid_label: str
    nominal_resolution: str
    has_aux_unc: str
    region: str
    tracking_id: str
    variable_id: str
    variant_label: str = "ILAMB" 
    variant_info: str = "CMORized product prepared by ILAMB"
    institution: str
    institution_id: str
    contact: str

class Processing(BaseModel):
    processing_code_location: HttpUrl
    creation_date: str
    dataset_contributor: str
    history: str
    source_data_retrieval_date: str
    title: str
    version: str

class References(BaseModel):
    references: str

class License(BaseModel):
    license: str

class DOI(BaseModel):
    doi: HttpUrl | list[HttpUrl]

class DataGlobalAttrs(BaseModel):
    dataset: Dataset
    processing: Processing
    references: References
    license: License
    data_doi: DOI

    @field_validator("dataset")
    @classmethod
    def check_frequency(cls, v):
        if v.frequency not in {"mon", "day", "fx"}:
            raise ValueError("frequency must be CMIP-style (mon/day/fx)")
        return v


    @model_validator(mode="after")
    def _cache_flat(self):
        self._flat = {}
        for section in self.model_dump(mode="json").values():
            if isinstance(section, dict):
                self._flat.update(section)
        return self

    def flatten(self):
        return self._flat.copy()
