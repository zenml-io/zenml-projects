# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CreditScoringFeatures(BaseModel):
    """Credit scoring features for prediction."""

    # SK_ID_CURR is an identifier, not used for modeling
    # TARGET is what we're predicting, so not included in features

    NAME_CONTRACT_TYPE: str = Field(..., description="Type of contract")
    CODE_GENDER: str = Field(..., description="Gender of the client")
    FLAG_OWN_CAR: str = Field(..., description="Flag if client owns a car")
    FLAG_OWN_REALTY: str = Field(..., description="Flag if client owns real estate")
    CNT_CHILDREN: float = Field(..., description="Number of children")
    AMT_INCOME_TOTAL: float = Field(..., description="Income of the client")
    AMT_CREDIT: float = Field(..., description="Credit amount of the loan")
    AMT_ANNUITY: float = Field(..., description="Loan annuity")
    AMT_GOODS_PRICE: float = Field(..., description="Price of goods for the loan")
    NAME_TYPE_SUITE: str = Field(..., description="Who accompanied client when applying")
    NAME_INCOME_TYPE: str = Field(..., description="Income type of client")
    NAME_EDUCATION_TYPE: str = Field(..., description="Education level of client")
    NAME_FAMILY_STATUS: str = Field(..., description="Family status of client")
    NAME_HOUSING_TYPE: str = Field(..., description="Housing type of client")
    DAYS_BIRTH: float = Field(..., description="Days before application client was born")
    DAYS_EMPLOYED: float = Field(..., description="Days before application client was employed")
    OCCUPATION_TYPE: str = Field(..., description="Occupation of client")
    CNT_FAM_MEMBERS: float = Field(..., description="Count of family members")
    REGION_RATING_CLIENT: float = Field(..., description="Region rating of client")
    EXT_SOURCE_1: float = Field(..., description="External source 1 (normalized score)")
    EXT_SOURCE_2: float = Field(..., description="External source 2 (normalized score)")
    EXT_SOURCE_3: float = Field(..., description="External source 3 (normalized score)")


class PredictionResponse(BaseModel):
    """Response model for credit scoring prediction."""

    probabilities: List[float]
    model_version: str
    timestamp: str
    risk_assessment: Optional[Dict[str, float]] = None


class IncidentReport(BaseModel):
    """Model for reporting incidents."""

    severity: str = Field(..., description="Severity of the incident", example="medium")
    description: str = Field(..., description="Description of the incident")
    data: Optional[Dict] = Field(None, description="Additional data related to the incident")
    webhook_url: Optional[str] = Field(None, description="Optional webhook URL for notifications")


class IncidentResponse(BaseModel):
    """Response model for incident reports."""

    status: str
    incident_id: Optional[str] = None
    message: Optional[str] = None
    timestamp: str


class MonitorResponse(BaseModel):
    """Response model for drift monitoring."""

    timestamp: str
    drift_detected: bool
    status: Optional[str] = None
    message: Optional[str] = None


class ApiEndpoints(BaseModel):
    """URLs for the API endpoints."""

    root: str
    health: str
    predict: str
    monitor: str
    incident: str


class ApiInfo(BaseModel):
    """Response model for the root endpoint."""

    message: str
    endpoints: ApiEndpoints
    model: str
    timestamp: str
