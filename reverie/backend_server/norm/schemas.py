"""
Pydantic v2 schemas for norm-related structured LLM outputs.
"""
from pydantic import BaseModel, Field
from typing import Literal


class NormEntry(BaseModel):
    id: str
    type: str
    content: str
    subject: str
    predicate: str
    object: str
    utility: int = Field(ge=1, le=100)
    activation_state: bool
    validity_state: bool


class NormDatabase(BaseModel):
    norm_1: NormEntry
    norm_2: NormEntry
    norm_3: NormEntry
    norm_4: NormEntry
    norm_5: NormEntry


class DefectionAssessment(BaseModel):
    decision: Literal["comply", "defect"]
    reasoning: str
    detection_risk: int = Field(ge=1, le=10)
    expected_benefit: int = Field(ge=1, le=10)


class ViolationCheck(BaseModel):
    violation: bool
    severity: int = Field(ge=1, le=10)
    certainty: int = Field(ge=1, le=10)
    response: Literal["confront", "gossip", "ignore"]


class ConflictDecision(BaseModel):
    decision: str
    reasoning: str


class NormUtilityRating(BaseModel):
    score: int = Field(ge=1, le=100)
    reason: str
