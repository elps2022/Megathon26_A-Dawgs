"""FastAPI application for Green Home Score and Retrofit Engine.

Exposes REST endpoints for reports, scenarios, lender portfolio, customer memory, and grounding verification.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from greenhome.engine.portfolio import build_portfolio
from greenhome.engine.report import build_report
from greenhome.llm.grounding import verify_grounding
from greenhome.memory.store import CustomerMemoryStore
from greenhome.schema import PortfolioSummary, Property, PropertyReport

# In-memory datasets cache
DATA_CACHE: dict[str, Any] = {}
memory_store = CustomerMemoryStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load dataset on startup if available
    props_path = Path("data/generated/properties.csv")
    loans_path = Path("data/generated/loans.csv")
    if props_path.exists():
        DATA_CACHE["properties"] = pd.read_csv(props_path)
    if loans_path.exists():
        DATA_CACHE["loans"] = pd.read_csv(loans_path)
    yield
    DATA_CACHE.clear()


app = FastAPI(
    title="Green Home Score & Retrofit Recommendation API",
    description="Lloyds Problem Statement 2 (Megathon X)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_property_by_id(property_id: str) -> Property:
    df = DATA_CACHE.get("properties")
    if df is None:
        props_path = Path("data/generated/properties.csv")
        if props_path.exists():
            df = pd.read_csv(props_path)
            DATA_CACHE["properties"] = df
        else:
            raise HTTPException(status_code=500, detail="Data not generated yet. Run make data first.")

    matched = df[df["property_id"] == property_id]
    if matched.empty:
        raise HTTPException(status_code=404, detail=f"Property {property_id} not found.")

    row = matched.iloc[0].to_dict()
    clean_d = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    if clean_d.get("data_flags"):
        clean_d["data_flags"] = [f.strip() for f in str(clean_d["data_flags"]).split(",") if f.strip() and f != "nan"]
    else:
        clean_d["data_flags"] = []

    return Property(**clean_d)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "greenhome-api"}


@app.get("/property/{property_id}", response_model=Property)
def get_property(property_id: str) -> Property:
    """Retrieve raw prediction-time property record."""
    return _get_property_by_id(property_id)


@app.get("/report/{property_id}", response_model=PropertyReport)
def get_report(property_id: str) -> PropertyReport:
    """Generate baseline PropertyReport with Green Home Score, issues, and sequenced recs."""
    p = _get_property_by_id(property_id)
    mem = memory_store.get_memory(p.customer_id)
    p_state = memory_store.apply_memory_to_property(p)
    return build_report(p_state, memory=mem)


class ScenarioRequest(BaseModel):
    retrofits: list[str]


@app.post("/scenario/{property_id}", response_model=PropertyReport)
def run_scenario(property_id: str, req: ScenarioRequest) -> PropertyReport:
    """Rerun engine calculations for proposed what-if retrofit package."""
    p = _get_property_by_id(property_id)
    mem = memory_store.get_memory(p.customer_id)
    p_state = memory_store.apply_memory_to_property(p)
    return build_report(p_state, retrofits=req.retrofits, memory=mem)


@app.get("/portfolio", response_model=PortfolioSummary)
def get_portfolio(sample: int = 200) -> PortfolioSummary:
    """Lender mortgage portfolio view aggregating financed emissions and transition targets."""
    df_props = DATA_CACHE.get("properties")
    df_loans = DATA_CACHE.get("loans")

    if df_props is None:
        props_path = Path("data/generated/properties.csv")
        loans_path = Path("data/generated/loans.csv")
        if props_path.exists():
            df_props = pd.read_csv(props_path)
            DATA_CACHE["properties"] = df_props
        if loans_path.exists():
            df_loans = pd.read_csv(loans_path)
            DATA_CACHE["loans"] = df_loans

    if df_props is None:
        raise HTTPException(status_code=500, detail="Generated properties not available.")

    return build_portfolio(df_props, df_loans, max_sample=sample)


@app.get("/memory/{customer_id}")
def get_customer_memory(customer_id: str) -> dict[str, Any]:
    """Retrieve persistent customer memory."""
    return memory_store.get_memory(customer_id)


class ConstraintRequest(BaseModel):
    constraint: str
    note: str = ""


@app.post("/memory/{customer_id}/constraint")
def add_customer_constraint(customer_id: str, req: ConstraintRequest) -> dict[str, Any]:
    """Add living constraint to customer profile."""
    return memory_store.record_constraint(customer_id, req.constraint, req.note)


class GroundingVerifyRequest(BaseModel):
    llm_text: str
    report: dict[str, Any]


@app.post("/verify_grounding")
def check_grounding(req: GroundingVerifyRequest) -> dict[str, Any]:
    """Audit LLM explanation for hallucinated numbers."""
    result = verify_grounding(req.llm_text, req.report)
    return result.to_dict()
