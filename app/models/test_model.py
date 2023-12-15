from typing import Optional, Literal
from pydantic import BaseModel, Field, ValidationError, validator


class StringifiedFormModel(BaseModel):
    testStr: str = Field(...) # ... Requires an input everytime. Offering default value can allow frontend to not fill out this field. 
    class Config:
        allow_population_by_field_name = True
        schema_extra = {}