from pydantic import BaseModel, Field, field_validator
from typing import Literal

EntityType = Literal["PERSON", "ORG", "LOCATION", "CONCEPT", "UNKNOWN"]


class Entity(BaseModel):
    name: str = Field(description="Canonical entity name")
    type: EntityType = Field(description="Entity category")

    model_config = {"extra": "ignore"}

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Entity name must be non-empty")
        return v.strip()


class Relation(BaseModel):
    source: str = Field(description="Source entity name, must match an entity name exactly")
    target: str = Field(description="Target entity name, must match an entity name exactly")
    relation: str = Field(description="Relation label, e.g., works_at, advises")

    model_config = {"extra": "ignore"}

    @field_validator("source", "target", "relation")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Relation fields must be non-empty")
        return v.strip()


class ExtractionResult(BaseModel):
    """Structured output for LLM extraction."""

    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

    model_config = {"extra": "forbid"}
