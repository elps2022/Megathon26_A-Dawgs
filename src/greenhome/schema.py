"""Pydantic data models for Green Home Score and Retrofit Recommendation Engine.

Shared contract defined in docs/ARCHITECTURE.md section 4.
"""

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class Property(BaseModel):
    """Property characteristics and observable state at prediction time."""

    property_id: str
    customer_id: str
    local_authority: str
    region: str
    property_type: str
    age_band: str
    floor_area_m2: float | None = None
    bedrooms: int | None = None
    energy_rating: str | None = None
    sap_score: float | None = None
    heating: str
    wall_type: str  # solid | cavity
    wall_insulation: str | None = None  # none | partial | full
    has_own_roof: bool = True
    loft_insulation_mm: float | None = None
    glazing: str  # single | partial | double_old | double_new | triple
    solar_kwp: float = 0.0
    smart_controls: str = "none"  # none | programmer | smart_thermostat | smart_trvs
    gas_meter: bool = True
    imd_quintile: float | None = None  # 1 = most deprived (energy-model feature, NaN-safe)
    cavity_filled_retrofit: bool | None = None  # EPC "filled cavity": a retrofit fill (energy-model feature)
    current_value_estimate: float | None = None
    annual_elec_kwh: float | None = None
    annual_gas_kwh: float | None = None
    data_flags: list[str] = Field(default_factory=list)

    @field_validator("data_flags", mode="before")
    @classmethod
    def clean_flags(cls, v: Any) -> list[str]:
        if v is None or (isinstance(v, float) and v != v):  # None or NaN
            return []
        if isinstance(v, str):
            return [f.strip() for f in v.split(",") if f.strip() and f != "nan"]
        return list(v)


class Element(BaseModel):
    """Building envelope element thermal loss."""

    name: str  # roof | wall | window | floor
    area_m2: float
    u_value: float
    heat_loss_w_per_k: float
    share: float


class HomeState(BaseModel):
    """Physical state of home. Retrofits mutate this state."""

    elements: list[Element]
    air_changes_per_hr: float
    heating_system: str
    heating_efficiency: float
    smart_controls: str
    solar_kwp: float
    region: str


class EnergyBreakdown(BaseModel):
    """Predicted or calibrated energy metrics."""

    elec_kwh: float
    gas_kwh: float
    heat_demand_kwh: float
    base_load_kwh: float
    hlc_w_per_k: float | None = None
    calibration_k: float | None = None
    calibrated: bool = False
    model_version: str = "greybox"
    elements: list[Element] = Field(default_factory=list)  # per-element heat loss (house diagram)
    # 80% band for this home's actual annual use (conformalised, V1 only); None when not estimated by the model
    elec_p10: float | None = None
    elec_p90: float | None = None
    gas_p10: float | None = None
    gas_p90: float | None = None


class SubScore(BaseModel):
    """Component score out of 100 with weight and contribution."""

    component: str
    weight: float
    score: float
    contribution: float
    inputs_used: dict[str, Any] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)


class ScoreResult(BaseModel):
    """Total Green Home Score and category."""

    total: int
    classification: str  # Highly efficient | Good | Moderate | Poor | Very poor
    components: list[SubScore]


class FuelDelta(BaseModel):
    """Energy changes per fuel. Positive = saving, negative = increase."""

    elec_kwh: float
    gas_kwh: float
    total_kwh: float
    low: float
    high: float


class RetrofitRec(BaseModel):
    """Recommended retrofit with suitability, marginal savings, and cost."""

    retrofit: str
    suitability: Literal["High", "Medium", "Low"]
    reason: str
    marginal_saving: FuelDelta
    carbon_kg: float
    indicative_cost: tuple[float, float]


class Infeasible(BaseModel):
    """Retrofit disqualified by feasibility rules."""

    retrofit: str
    rule: str


class PropertyReport(BaseModel):
    """Single source of truth containing complete audit metrics for a property."""

    property: Property
    score: ScoreResult
    score_after: ScoreResult
    is_poor: bool
    key_issues: list[str] = Field(default_factory=list)
    improvement_potential_pct: float | None = None
    recommendations: list[RetrofitRec] = Field(default_factory=list)
    infeasible: list[Infeasible] = Field(default_factory=list)
    # None = unavailable (no usable bills and too little data to estimate expected use); never a silent 0.
    package_saving: FuelDelta | None = None
    package_carbon_kg: float | None = None
    value: dict[str, Any]  # status, baseline, low, high, pct_low, pct_high, method, caveat (None when unavailable)
    energy_before: EnergyBreakdown | None = None
    energy_after: EnergyBreakdown | None = None
    assumptions: list[str] = Field(default_factory=list)


class PortfolioPropertySummary(BaseModel):
    """Single mortgage record for lender portfolio aggregation."""

    property_id: str
    customer_id: str
    local_authority: str
    score: int
    classification: str
    is_poor: bool
    annual_carbon_kg: float | None = None  # None = energy unavailable
    financed_emissions_kg: float | None = None  # None = valuation or balance unavailable
    improvement_potential_pct: float | None = None
    ltv: float | None = None
    outstanding_balance: float | None = None


class PortfolioSummary(BaseModel):
    """PCAF mortgage book aggregation for lender view."""

    total_properties: int
    poor_count: int
    poor_share: float
    avg_score: float
    total_emissions_tonnes: float
    total_financed_emissions_tonnes: float
    potential_carbon_reduction_tonnes: float
    energy_unavailable_count: int = 0  # homes with no bills and too little data to estimate (excluded from totals)
    high_potential_targets: list[PortfolioPropertySummary] = Field(default_factory=list)
