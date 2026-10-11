"""FastAPI application for Green Home Score and Retrofit Engine.

Exposes REST endpoints for reports, scenarios, lender portfolio, customer memory,
Gemini-powered grounded AI advisor, and grounding verification.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from greenhome.engine.portfolio import build_portfolio
from greenhome.engine.report import build_report
from greenhome.llm.client import LLMAdvisor
from greenhome.llm.grounding import verify_grounding
from greenhome.memory.store import CustomerMemoryStore
from greenhome.schema import PortfolioSummary, Property, PropertyReport

# In-memory datasets cache
DATA_CACHE: dict[str, Any] = {}
memory_store = CustomerMemoryStore()
llm_advisor = LLMAdvisor()

REAL_PROPS_PATH = Path("data/processed/real/properties_share.parquet")
REAL_LOANS_PATH = Path("data/processed/real/loans.parquet")
SYNTH_PROPS_PATH = Path("data/generated/properties.csv")
SYNTH_LOANS_PATH = Path("data/generated/loans.csv")


def _dict_to_property(row: dict[str, Any]) -> Property:
    clean_d = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    if clean_d.get("data_flags"):
        if isinstance(clean_d["data_flags"], list):
            clean_d["data_flags"] = [f for f in clean_d["data_flags"] if f]
        else:
            clean_d["data_flags"] = [
                f.strip() for f in str(clean_d["data_flags"]).split(",") if f.strip() and f != "nan"
            ]
    else:
        clean_d["data_flags"] = []
    return Property(**clean_d)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload synthetic dataset if available
    if SYNTH_PROPS_PATH.exists():
        DATA_CACHE["properties"] = pd.read_csv(SYNTH_PROPS_PATH)
    if SYNTH_LOANS_PATH.exists():
        DATA_CACHE["loans"] = pd.read_csv(SYNTH_LOANS_PATH)
    # Preload real property index / dataset if present
    if REAL_PROPS_PATH.exists():
        DATA_CACHE["real_properties"] = pd.read_parquet(REAL_PROPS_PATH)
    if REAL_LOANS_PATH.exists():
        DATA_CACHE["real_loans"] = pd.read_parquet(REAL_LOANS_PATH)
    yield
    DATA_CACHE.clear()


app = FastAPI(
    title="Green Home Score & Retrofit Recommendation API",
    description="Lloyds Problem Statement 2 (Megathon X) — AI Advisor powered by Google Gemini with Deterministic Grounding Shield",
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
    # 1. Search cached synthetic
    df_synth = DATA_CACHE.get("properties")
    if df_synth is not None:
        matched = df_synth[df_synth["property_id"] == property_id]
        if not matched.empty:
            return _dict_to_property(matched.iloc[0].to_dict())

    # 2. Search cached real 542k properties
    df_real = DATA_CACHE.get("real_properties")
    if df_real is not None:
        matched = df_real[df_real["property_id"] == property_id]
        if not matched.empty:
            return _dict_to_property(matched.iloc[0].to_dict())

    # 3. Try loading real parquet on demand if not yet cached
    if REAL_PROPS_PATH.exists():
        df_real = pd.read_parquet(REAL_PROPS_PATH)
        DATA_CACHE["real_properties"] = df_real
        matched = df_real[df_real["property_id"] == property_id]
        if not matched.empty:
            return _dict_to_property(matched.iloc[0].to_dict())

    # 4. Try loading synthetic on demand if not yet cached
    if SYNTH_PROPS_PATH.exists():
        df_synth = pd.read_csv(SYNTH_PROPS_PATH)
        DATA_CACHE["properties"] = df_synth
        matched = df_synth[df_synth["property_id"] == property_id]
        if not matched.empty:
            return _dict_to_property(matched.iloc[0].to_dict())

    raise HTTPException(status_code=404, detail=f"Property {property_id} not found in synthetic or real datasets.")


@app.get("/")
def root() -> dict[str, Any]:
    """Root metadata and service health status."""
    return {
        "service": "Lloyds Green Home Score & Retrofit Recommendation API",
        "status": "online",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "llm_provider": llm_advisor.provider,
        "llm_model": llm_advisor.model,
        "endpoints": {
            "health": "GET /health",
            "property": "GET /property/{id}",
            "report": "GET /report/{id}",
            "scenario": "POST /scenario/{id}",
            "assess": "POST /assess",
            "chat": "POST /chat",
            "portfolio": "GET /portfolio?sample=200&source=real",
            "verify_grounding": "POST /verify_grounding",
            "memory": "GET /memory/{customer_id}",
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "greenhome-api"}


@app.get("/property/{property_id}", response_model=Property)
def get_property(property_id: str) -> Property:
    """Retrieve raw prediction-time property record (supports both synthetic and real 542k IDs)."""
    return _get_property_by_id(property_id)


@app.get("/report/{property_id}", response_model=PropertyReport)
def get_report(property_id: str) -> PropertyReport:
    """Generate baseline PropertyReport with Green Home Score, issues, and sequenced recs."""
    p = _get_property_by_id(property_id)
    mem = memory_store.get_memory(p.customer_id) if p.customer_id else None
    p_state = memory_store.apply_memory_to_property(p) if mem else p
    return build_report(p_state, memory=mem)


@app.post("/assess", response_model=PropertyReport)
def assess_property(p: Property) -> PropertyReport:
    """Directly calculate Green Home Score, energy breakdown, and retrofits for any custom home payload."""
    mem = memory_store.get_memory(p.customer_id) if p.customer_id else None
    p_state = memory_store.apply_memory_to_property(p) if mem else p
    return build_report(p_state, memory=mem)


class ScenarioRequest(BaseModel):
    retrofits: list[str]


@app.post("/scenario/{property_id}", response_model=PropertyReport)
def run_scenario(property_id: str, req: ScenarioRequest) -> PropertyReport:
    """Rerun engine calculations for proposed what-if retrofit package."""
    p = _get_property_by_id(property_id)
    mem = memory_store.get_memory(p.customer_id) if p.customer_id else None
    p_state = memory_store.apply_memory_to_property(p) if mem else p
    return build_report(p_state, retrofits=req.retrofits, memory=mem)


@app.get("/portfolio", response_model=PortfolioSummary)
def get_portfolio(
    sample: int = Query(default=200, ge=1, le=5000),
    source: str = Query(default="real", pattern="^(real|synthetic)$"),
) -> PortfolioSummary:
    """Lender mortgage portfolio view aggregating financed emissions and transition targets.

    Parameters:
    - sample: number of properties to sample (default 200)
    - source: 'real' (542k real property book + 222k loan book) or 'synthetic' (10k generated)
    """
    if source == "real" and REAL_PROPS_PATH.exists():
        if "real_properties" not in DATA_CACHE:
            DATA_CACHE["real_properties"] = pd.read_parquet(REAL_PROPS_PATH)
        if "real_loans" not in DATA_CACHE and REAL_LOANS_PATH.exists():
            DATA_CACHE["real_loans"] = pd.read_parquet(REAL_LOANS_PATH)
        df_props = DATA_CACHE["real_properties"]
        df_loans = DATA_CACHE.get("real_loans")
    else:
        df_props = DATA_CACHE.get("properties")
        df_loans = DATA_CACHE.get("loans")
        if df_props is None and SYNTH_PROPS_PATH.exists():
            df_props = pd.read_csv(SYNTH_PROPS_PATH)
            DATA_CACHE["properties"] = df_props
        if df_loans is None and SYNTH_LOANS_PATH.exists():
            df_loans = pd.read_csv(SYNTH_LOANS_PATH)
            DATA_CACHE["loans"] = df_loans

    if df_props is None:
        raise HTTPException(status_code=500, detail="Properties dataset not available.")

    return build_portfolio(df_props, df_loans, max_sample=sample)


class ChatRequest(BaseModel):
    query: str
    property_id: str | None = None
    property: Property | None = None
    history: list[dict[str, str]] = []


@app.post("/chat")
def chat_advisor(req: ChatRequest) -> dict[str, Any]:
    """Consult the Lloyds Green Home Energy Advisor powered by Google Gemini with Grounding Shield."""
    if req.property is not None:
        p = req.property
    elif req.property_id:
        p = _get_property_by_id(req.property_id)
    else:
        raise HTTPException(
            status_code=400, detail="Must provide either property_id or property payload for grounding."
        )

    mem = memory_store.get_memory(p.customer_id) if p.customer_id else None
    p_state = memory_store.apply_memory_to_property(p) if mem else p
    report = build_report(p_state, memory=mem)

    resp_text, audit, model_label = llm_advisor.query(
        req.query, report, p_state, history=req.history
    )

    return {
        "response": resp_text,
        "grounded": audit.grounded,
        "grounding_score": audit.grounding_score,
        "audit": audit.to_dict(),
        "model": model_label,
        "provider": llm_advisor.provider,
    }


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
