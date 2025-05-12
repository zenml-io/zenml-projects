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

    borrow_block_number: float = Field(..., description="Block number when borrow occurred")
    borrow_timestamp: float = Field(..., description="Timestamp of borrow transaction")
    first_tx_timestamp: float = Field(..., description="Timestamp of first transaction")
    last_tx_timestamp: float = Field(..., description="Timestamp of last transaction")
    wallet_age: float = Field(..., description="Age of wallet in days")
    incoming_tx_count: float = Field(..., description="Count of incoming transactions")
    outgoing_tx_count: float = Field(..., description="Count of outgoing transactions")
    net_incoming_tx_count: float = Field(..., description="Net count of incoming transactions")
    total_gas_paid_eth: float = Field(..., description="Total gas paid in ETH")
    avg_gas_paid_per_tx_eth: float = Field(
        ..., description="Average gas paid per transaction in ETH"
    )
    risky_tx_count: float = Field(..., description="Count of transactions to risky contracts")
    risky_unique_contract_count: float = Field(
        ..., description="Count of unique risky contracts interacted with"
    )
    risky_first_tx_timestamp: float = Field(..., description="Timestamp of first risky transaction")
    risky_last_tx_timestamp: float = Field(..., description="Timestamp of last risky transaction")
    risky_first_last_tx_timestamp_diff: float = Field(
        ..., description="Time difference between first and last risky transaction"
    )
    risky_sum_outgoing_amount_eth: float = Field(
        ..., description="Sum of ETH sent to risky contracts"
    )
    outgoing_tx_sum_eth: float = Field(..., description="Sum of outgoing transactions in ETH")
    incoming_tx_sum_eth: float = Field(..., description="Sum of incoming transactions in ETH")
    outgoing_tx_avg_eth: float = Field(
        ..., description="Average outgoing transaction amount in ETH"
    )
    incoming_tx_avg_eth: float = Field(
        ..., description="Average incoming transaction amount in ETH"
    )
    max_eth_ever: float = Field(..., description="Maximum ETH balance ever held")
    min_eth_ever: float = Field(..., description="Minimum ETH balance ever held")
    total_balance_eth: float = Field(..., description="Current total balance in ETH")
    risk_factor: float = Field(..., description="Risk factor score")
    total_collateral_eth: float = Field(..., description="Total collateral in ETH")
    total_collateral_avg_eth: float = Field(..., description="Average collateral in ETH")
    total_available_borrows_eth: float = Field(..., description="Total available to borrow in ETH")
    total_available_borrows_avg_eth: float = Field(
        ..., description="Average available to borrow in ETH"
    )
    avg_weighted_risk_factor: float = Field(..., description="Weighted average risk factor")
    risk_factor_above_threshold_daily_count: float = Field(
        ..., description="Days with risk factor above threshold"
    )
    avg_risk_factor: float = Field(..., description="Average risk factor")
    max_risk_factor: float = Field(..., description="Maximum risk factor")
    borrow_amount_sum_eth: float = Field(..., description="Sum of borrow amounts in ETH")
    borrow_amount_avg_eth: float = Field(..., description="Average borrow amount in ETH")
    borrow_count: float = Field(..., description="Count of borrow transactions")
    repay_amount_sum_eth: float = Field(..., description="Sum of repay amounts in ETH")
    repay_amount_avg_eth: float = Field(..., description="Average repay amount in ETH")
    repay_count: float = Field(..., description="Count of repay transactions")
    borrow_repay_diff_eth: float = Field(
        ..., description="Difference between borrow and repay in ETH"
    )
    deposit_count: float = Field(..., description="Count of deposit transactions")
    deposit_amount_sum_eth: float = Field(..., description="Sum of deposit amounts in ETH")
    time_since_first_deposit: float = Field(..., description="Time since first deposit")
    withdraw_amount_sum_eth: float = Field(..., description="Sum of withdraw amounts in ETH")
    withdraw_deposit_diff_if_positive_eth: float = Field(
        ..., description="Positive difference between withdraw and deposit in ETH"
    )
    liquidation_count: float = Field(..., description="Count of liquidation events")
    time_since_last_liquidated: float = Field(..., description="Time since last liquidation")
    liquidation_amount_sum_eth: float = Field(..., description="Sum of liquidation amounts in ETH")
    market_adx: float = Field(..., description="Market ADX indicator")
    market_adxr: float = Field(..., description="Market ADXR indicator")
    market_apo: float = Field(..., description="Market APO indicator")
    market_aroonosc: float = Field(..., description="Market AROONOSC indicator")
    market_aroonup: float = Field(..., description="Market AROONUP indicator")
    market_atr: float = Field(..., description="Market ATR indicator")
    market_cci: float = Field(..., description="Market CCI indicator")
    market_cmo: float = Field(..., description="Market CMO indicator")
    market_correl: float = Field(..., description="Market correlation indicator")
    market_dx: float = Field(..., description="Market DX indicator")
    market_fastk: float = Field(..., description="Market Fast K indicator")
    market_fastd: float = Field(..., description="Market Fast D indicator")
    market_ht_trendmode: float = Field(..., description="Market HT trend mode indicator")
    market_linearreg_slope: float = Field(..., description="Market linear regression slope")
    market_macd_macdext: float = Field(..., description="Market MACDEXT indicator")
    market_macd_macdfix: float = Field(..., description="Market MACDFIX indicator")
    market_macd: float = Field(..., description="Market MACD indicator")
    market_macdsignal_macdext: float = Field(..., description="Market MACDEXT signal line")
    market_macdsignal_macdfix: float = Field(..., description="Market MACDFIX signal line")
    market_macdsignal: float = Field(..., description="Market MACD signal line")
    market_max_drawdown_365d: float = Field(
        ..., description="Market maximum drawdown in last 365 days"
    )
    market_natr: float = Field(..., description="Market NATR indicator")
    market_plus_di: float = Field(..., description="Market +DI indicator")
    market_plus_dm: float = Field(..., description="Market +DM indicator")
    market_ppo: float = Field(..., description="Market PPO indicator")
    market_rocp: float = Field(..., description="Market ROCP indicator")
    market_rocr: float = Field(..., description="Market ROCR indicator")
    unique_borrow_protocol_count: float = Field(
        ..., description="Count of unique borrow protocols used"
    )
    unique_lending_protocol_count: float = Field(
        ..., description="Count of unique lending protocols used"
    )


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
