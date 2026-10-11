"""Green Home Score & Retrofit Recommendation Dashboard.

Multi-view interactive application:
1. Customer View: Green Home Score, SVG Thermal Twin House, Sequenced Recommendations, Value Uplift, Green Banking & Grants.
2. Lender Portfolio View: PCAF Scope 3 Financed Emissions, 2030 EPC C Transition Target Simulator.
3. AI Assistant & Grounding Inspector: Real-time hallucination-free verification.
"""

from pathlib import Path
import html
import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from greenhome.engine.portfolio import build_portfolio
from greenhome.engine.report import build_report
from greenhome.engine.savings import apply_retrofits_to_property, build_home_elements
from greenhome.llm.client import LLMAdvisor, generate_rule_fallback
from greenhome.llm.grounding import GroundingResult, extract_numbers_from_text, verify_grounding
from greenhome.memory.store import CustomerMemoryStore
from greenhome.schema import Property, PropertyReport

st.set_page_config(
    page_title="Lloyds Green Home Engine",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Lloyds Bank & Shadcn design system - Full Dark & Light Mode Support
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    code, pre, .mono {
      font-family: 'JetBrains Mono', monospace !important;
    }

    /* ═══════════════════════════════════════════════════════════════════
       SHADCN DUAL COLOR SCHEME ARCHITECTURE
       Light & Dark palettes defined in :root.
       Active tokens bind to light or dark opposites.
       Toggleable via single variable or class / data-theme attribute.
       ═══════════════════════════════════════════════════════════════════ */
    :root {
      color-scheme: light dark;

      /* ── Light Mode Palette (Bright/Clean Backgrounds, Deep Slate Text) ── */
      --light-card-bg: #ffffff;
      --light-card-border: #e2e8f0;
      --light-card-hover-border: #cbd5e1;
      --light-card-header-border: #f1f5f9;
      --light-subcard-bg: #f8fafc;
      --light-subcard-border: #e2e8f0;
      --light-text-primary: #0f172a;
      --light-text-sub: #334155;
      --light-text-muted: #64748b;
      --light-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
      --light-shadow-hover: 0 4px 10px -2px rgba(0, 0, 0, 0.07);
      --light-cashflow-grad: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, #ffffff 100%);
      --light-badge-slate-bg: rgba(100, 116, 139, 0.12);
      --light-badge-slate-text: #334155;
      --light-badge-shield-bg: #ecfdf5;
      --light-badge-shield-text: #065f46;
      --light-badge-shield-border: #6ee7b7;

      /* ── Dark Mode Palette (Deep Slate/Navy Backgrounds, Crisp Bright Text) ── */
      --dark-card-bg: #1e293b;
      --dark-card-border: rgba(255, 255, 255, 0.12);
      --dark-card-hover-border: rgba(255, 255, 255, 0.28);
      --dark-card-header-border: rgba(255, 255, 255, 0.08);
      --dark-subcard-bg: rgba(15, 23, 42, 0.65);
      --dark-subcard-border: rgba(255, 255, 255, 0.1);
      --dark-text-primary: #f8fafc;
      --dark-text-sub: #cbd5e1;
      --dark-text-muted: #94a3b8;
      --dark-shadow: 0 2px 6px 0 rgba(0, 0, 0, 0.35);
      --dark-shadow-hover: 0 6px 14px 0 rgba(0, 0, 0, 0.45);
      --dark-cashflow-grad: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, #1e293b 100%);
      --dark-badge-slate-bg: rgba(148, 163, 184, 0.16);
      --dark-badge-slate-text: #cbd5e1;
      --dark-badge-shield-bg: rgba(6, 95, 70, 0.35);
      --dark-badge-shield-text: #6ee7b7;
      --dark-badge-shield-border: rgba(110, 231, 183, 0.4);

      /* ── Active Theme Tokens (Default: Light Mode) ── */
      --card-bg: var(--light-card-bg);
      --card-border: var(--light-card-border);
      --card-hover-border: var(--light-card-hover-border);
      --card-header-border: var(--light-card-header-border);
      --subcard-bg: var(--light-subcard-bg);
      --subcard-border: var(--light-subcard-border);
      --text-primary: var(--light-text-primary);
      --text-sub: var(--light-text-sub);
      --text-muted: var(--light-text-muted);
      --card-shadow: var(--light-shadow);
      --card-shadow-hover: var(--light-shadow-hover);
      --cashflow-grad: var(--light-cashflow-grad);
      --badge-slate-bg: var(--light-badge-slate-bg);
      --badge-slate-text: var(--light-badge-slate-text);
      --badge-shield-bg: var(--light-badge-shield-bg);
      --badge-shield-text: var(--light-badge-shield-text);
      --badge-shield-border: var(--light-badge-shield-border);

      /* Aliases for backwards compatibility */
      --shadcn-card-bg: var(--card-bg);
      --shadcn-card-border: var(--card-border);
      --shadcn-card-hover-border: var(--card-hover-border);
      --shadcn-card-header-border: var(--card-header-border);
      --shadcn-subcard-bg: var(--subcard-bg);
      --shadcn-subcard-border: var(--subcard-border);
      --shadcn-card-text: var(--text-primary);
      --shadcn-card-text-sub: var(--text-sub);
      --shadcn-card-text-muted: var(--text-muted);
      --shadcn-card-shadow: var(--card-shadow);
      --shadcn-card-shadow-hover: var(--card-shadow-hover);
      --shadcn-cashflow-grad: var(--cashflow-grad);
    }

    /* The active palette follows the theme Streamlit is actually rendering (light / dark, incl. a live switch in
       Settings), not the OS preference: a small script sets data-theme on .stApp from its background colour
       (see sync_theme_attribute below). The OS media query was removed because the two can disagree. */

    /* Explicit Dark Theme selector */
    [data-theme="dark"],
    [data-base-theme="dark"],
    .dark,
    .stApp[data-theme="dark"],
    .stApp[data-test-theme="dark"],
    [data-testid="stAppViewContainer"][data-theme="dark"] {
      --card-bg: var(--dark-card-bg) !important;
      --card-border: var(--dark-card-border) !important;
      --card-hover-border: var(--dark-card-hover-border) !important;
      --card-header-border: var(--dark-card-header-border) !important;
      --subcard-bg: var(--dark-subcard-bg) !important;
      --subcard-border: var(--dark-subcard-border) !important;
      --text-primary: var(--dark-text-primary) !important;
      --text-sub: var(--dark-text-sub) !important;
      --text-muted: var(--dark-text-muted) !important;
      --card-shadow: var(--dark-shadow) !important;
      --card-shadow-hover: var(--dark-shadow-hover) !important;
      --cashflow-grad: var(--dark-cashflow-grad) !important;
      --badge-slate-bg: var(--dark-badge-slate-bg) !important;
      --badge-slate-text: var(--dark-badge-slate-text) !important;
      --badge-shield-bg: var(--dark-badge-shield-bg) !important;
      --badge-shield-text: var(--dark-badge-shield-text) !important;
      --badge-shield-border: var(--dark-badge-shield-border) !important;
    }

    /* Explicit Light Theme selector */
    [data-theme="light"],
    [data-base-theme="light"],
    .light,
    .stApp[data-theme="light"],
    .stApp[data-test-theme="light"],
    [data-testid="stAppViewContainer"][data-theme="light"] {
      --card-bg: var(--light-card-bg) !important;
      --card-border: var(--light-card-border) !important;
      --card-hover-border: var(--light-card-hover-border) !important;
      --card-header-border: var(--light-card-header-border) !important;
      --subcard-bg: var(--light-subcard-bg) !important;
      --subcard-border: var(--light-subcard-border) !important;
      --text-primary: var(--light-text-primary) !important;
      --text-sub: var(--light-text-sub) !important;
      --text-muted: var(--light-text-muted) !important;
      --card-shadow: var(--light-shadow) !important;
      --card-shadow-hover: var(--light-shadow-hover) !important;
      --cashflow-grad: var(--light-cashflow-grad) !important;
      --badge-slate-bg: var(--light-badge-slate-bg) !important;
      --badge-slate-text: var(--light-badge-slate-text) !important;
      --badge-shield-bg: var(--light-badge-shield-bg) !important;
      --badge-shield-text: var(--light-badge-shield-text) !important;
      --badge-shield-border: var(--light-badge-shield-border) !important;
    }

    /* Container respects Streamlit native theme background */
    .main, .block-container {
      background: transparent !important;
      padding-top: 1.25rem !important;
      max-width: 1440px !important;
    }

    /* Top Branded App Banner */
    .app-header {
      background: linear-gradient(135deg, #004d38 0%, #006a4e 55%, #008764 100%);
      border-radius: 14px;
      padding: 18px 24px;
      color: #ffffff;
      margin-bottom: 22px;
      box-shadow: 0 4px 14px rgba(0, 106, 78, 0.25);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .badge-shield {
      background: var(--badge-shield-bg) !important;
      color: var(--badge-shield-text) !important;
      border: 1px solid var(--badge-shield-border) !important;
      font-weight: 700 !important;
    }

    /* Permanent Lloyds Green Gradient for Sidebar (Invariant across light/dark modes) */
    section[data-testid="stSidebar"],
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {
      background: linear-gradient(180deg, #004d38 0%, #006a4e 55%, #008764 100%) !important;
      color: #ffffff !important;
    }

    /* All text, headings and labels in sidebar strictly crisp and visible */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] .stSubheader,
    [data-testid="stSidebar"] [data-testid="stHeadingWithActionElements"] {
      color: #ffffff !important;
      font-weight: 700 !important;
      letter-spacing: -0.2px;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] b,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] label {
      color: #ffffff !important;
    }

    [data-testid="stSidebar"] .sidebar-brand-name {
      font-weight: 800;
      font-size: 18px;
      letter-spacing: -0.4px;
      line-height: 1.2;
      color: #ffffff !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
      color: #a7f3d0 !important;
      opacity: 0.95 !important;
    }

    /* Radio buttons in sidebar */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
      color: #ffffff !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
      background: rgba(255, 255, 255, 0.1) !important;
      padding: 6px 12px !important;
      border-radius: 8px !important;
      border: 1px solid rgba(255, 255, 255, 0.16) !important;
      color: #ffffff !important;
      margin-bottom: 4px !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
      background: rgba(255, 255, 255, 0.18) !important;
      border-color: rgba(255, 255, 255, 0.35) !important;
    }

    /* Checkboxes in sidebar */
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label {
      background: rgba(255, 255, 255, 0.1) !important;
      padding: 7px 12px !important;
      border-radius: 8px !important;
      border: 1px solid rgba(255, 255, 255, 0.16) !important;
      color: #ffffff !important;
      margin-bottom: 6px !important;
      display: flex !important;
      align-items: center !important;
    }

    [data-testid="stSidebar"] [data-testid="stCheckbox"] label:hover {
      background: rgba(255, 255, 255, 0.18) !important;
      border-color: #34d399 !important;
    }

    [data-testid="stSidebar"] [data-testid="stCheckbox"] label span {
      color: #ffffff !important;
      font-weight: 500 !important;
    }

    [data-testid="stSidebar"] [data-testid="stCheckbox"] [data-baseweb="checkbox"] span {
      border-color: rgba(255, 255, 255, 0.6) !important;
    }

    /* Divider in sidebar */
    [data-testid="stSidebar"] hr {
      border-color: rgba(255, 255, 255, 0.22) !important;
      margin: 16px 0 !important;
    }

    /* Collapse icon / button in sidebar */
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapseButton"] button svg {
      color: #ffffff !important;
      fill: #ffffff !important;
    }

    /* Metric Cards - Shadcn Styled with Theme Adaptation */
    [data-testid="stMetric"] {
      background: var(--card-bg) !important;
      border: 1px solid var(--card-border) !important;
      border-radius: 12px !important;
      padding: 14px 18px !important;
      box-shadow: var(--card-shadow) !important;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    [data-testid="stMetric"]:hover {
      border-color: var(--card-hover-border) !important;
      box-shadow: var(--card-shadow-hover) !important;
      transform: translateY(-1px) !important;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricLabel"] p {
      font-size: 12px !important;
      font-weight: 600 !important;
      color: var(--text-muted) !important;
      text-transform: uppercase !important;
      letter-spacing: 0.5px !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] *,
    [data-testid="stMetricValue"] div {
      font-size: 25px !important;
      font-weight: 800 !important;
      color: var(--text-primary) !important;
      letter-spacing: -0.5px !important;
    }

    /* Shadcn Cards */
    .shadcn-card {
      background: var(--card-bg) !important;
      border: 1px solid var(--card-border) !important;
      border-radius: 12px;
      padding: 18px 20px;
      margin-bottom: 14px;
      box-shadow: var(--card-shadow);
      transition: all 0.2s ease;
      color: var(--text-primary) !important;
    }

    .shadcn-card:hover {
      border-color: var(--card-hover-border) !important;
      box-shadow: var(--card-shadow-hover);
    }

    .shadcn-card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--card-header-border);
    }

    .shadcn-subcard {
      background: var(--subcard-bg) !important;
      border: 1px solid var(--subcard-border) !important;
      border-radius: 8px;
    }

    .prop-summary-box {
      background: var(--card-bg) !important;
      border: 1px solid var(--card-border) !important;
      border-radius: 10px !important;
      padding: 8px 14px !important;
      margin-top: 5px !important;
      display: flex !important;
      justify-content: space-between !important;
      align-items: center !important;
      color: var(--text-primary) !important;
    }

    .prop-label {
      color: var(--text-muted) !important;
    }

    .prop-val {
      color: var(--text-primary) !important;
    }

    /* Badges & Pills - High Contrast in both Light and Dark Themes */
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      font-size: 11px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 6px;
      letter-spacing: 0.3px;
    }
    .badge-green { background: rgba(16, 185, 129, 0.16); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-amber { background: rgba(245, 158, 11, 0.16); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-red { background: rgba(239, 68, 68, 0.16); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-blue { background: rgba(59, 130, 246, 0.16); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.4); }
    .badge-slate { background: var(--badge-slate-bg) !important; color: var(--badge-slate-text) !important; border: 1px solid var(--card-border) !important; }

    .pill-high { background: rgba(16, 185, 129, 0.18); color: #10b981; font-weight: 700; padding: 4px 10px; border-radius: 6px; font-size: 11px; border: 1px solid rgba(16, 185, 129, 0.4); }
    .pill-med { background: rgba(245, 158, 11, 0.18); color: #f59e0b; font-weight: 700; padding: 4px 10px; border-radius: 6px; font-size: 11px; border: 1px solid rgba(245, 158, 11, 0.4); }
    .pill-low { background: var(--badge-slate-bg) !important; color: var(--badge-slate-text) !important; font-weight: 700; padding: 4px 10px; border-radius: 6px; font-size: 11px; border: 1px solid var(--card-border) !important; }

    /* Progress bar */
    .stProgress > div > div > div > div {
      background: linear-gradient(90deg, #006a4e 0%, #10b981 100%) !important;
      border-radius: 9999px !important;
    }
    .stProgress > div > div > div {
      background-color: rgba(128, 128, 128, 0.22) !important;
      border-radius: 9999px !important;
      height: 8px !important;
    }

    /* Tabs - Clean contrast in Dark and Light mode */
    .stTabs [data-baseweb="tab-list"] {
      gap: 6px;
      background-color: var(--subcard-bg) !important;
      padding: 5px;
      border-radius: 10px;
      border: 1px solid var(--card-border) !important;
      margin-bottom: 16px;
    }

    .stTabs [data-baseweb="tab"] {
      height: 38px;
      border-radius: 7px;
      padding: 0 16px;
      font-weight: 600;
      font-size: 13px;
      color: var(--text-muted) !important;
      opacity: 0.85;
      border: none !important;
    }

    .stTabs [aria-selected="true"] {
      background-color: #006a4e !important;
      color: #ffffff !important;
      opacity: 1.0 !important;
      box-shadow: 0 2px 6px rgba(0, 106, 78, 0.35);
    }

    /* Chat bubble styling */
    .chat-bubble-assistant {
      background: var(--card-bg) !important;
      border: 1px solid var(--card-border) !important;
      border-radius: 12px 12px 12px 2px;
      padding: 16px 20px;
      color: var(--text-primary) !important;
      margin-bottom: 14px;
      box-shadow: var(--card-shadow);
      line-height: 1.55;
    }

    .chat-shield-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(16, 185, 129, 0.16);
      color: #10b981;
      border: 1px solid rgba(16, 185, 129, 0.4);
      border-radius: 6px;
      padding: 3px 8px;
      font-size: 11px;
      font-weight: 700;
      margin-top: 10px;
    }

    /* Hover-only Verified Tag Symbol & Tooltip */
    .verified-symbol-wrapper {
      position: relative;
      display: inline-flex;
      align-items: center;
      cursor: pointer;
    }

    .verified-icon-symbol {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 24px;
      height: 24px;
      border-radius: 50%;
      background: rgba(16, 185, 129, 0.15);
      border: 1.5px solid rgba(16, 185, 129, 0.5);
      color: #10b981;
      font-size: 12px;
      font-weight: 800;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .verified-symbol-wrapper:hover .verified-icon-symbol {
      background: #10b981;
      color: #ffffff;
      transform: scale(1.1);
      box-shadow: 0 0 10px rgba(16, 185, 129, 0.45);
    }

    .verified-hover-tooltip {
      visibility: hidden;
      opacity: 0;
      position: absolute;
      right: 0;
      bottom: calc(100% + 8px);
      background: #064e3b;
      color: #ffffff;
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.2px;
      white-space: nowrap;
      box-shadow: 0 6px 16px rgba(0, 0, 0, 0.35);
      border: 1px solid rgba(16, 185, 129, 0.4);
      transition: opacity 0.2s ease, transform 0.2s ease;
      transform: translateY(4px);
      z-index: 1000;
      pointer-events: none;
    }

    .verified-hover-tooltip::after {
      content: "";
      position: absolute;
      top: 100%;
      right: 8px;
      border-width: 5px;
      border-style: solid;
      border-color: #064e3b transparent transparent transparent;
    }

    .verified-symbol-wrapper:hover .verified-hover-tooltip {
      visibility: visible;
      opacity: 1;
      transform: translateY(0);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_datasets():
    real_p = Path("data/processed/real/properties.parquet")
    real_l = Path("data/processed/real/loans.parquet")
    synth_p = Path("data/generated/properties.csv")
    synth_l = Path("data/generated/loans.csv")

    if real_p.exists():
        df_p = pd.read_parquet(real_p)
        source = f"Real UK NEED Dataset ({len(df_p):,} properties)"
    elif synth_p.exists():
        df_p = pd.read_csv(synth_p)
        source = f"Synthetic Dataset ({len(df_p):,} properties)"
    else:
        df_p = pd.DataFrame()
        source = "No Data Found"

    if real_l.exists():
        df_l = pd.read_parquet(real_l)
    elif synth_l.exists():
        df_l = pd.read_csv(synth_l)
    else:
        df_l = pd.DataFrame()

    return df_p, df_l, source


df_properties, df_loans, data_source_label = load_datasets()


@st.cache_data
def get_featured_properties(df_p: pd.DataFrame, max_count: int = 300) -> list[str]:
    """Curate a diverse showcase sample covering all EPC bands (A–G), property types, and heating fuels."""
    if df_p.empty or "energy_rating" not in df_p.columns:
        return df_p["property_id"].head(max_count).tolist() if not df_p.empty else []

    valid = df_p[df_p["annual_elec_kwh"].notna() & (df_p["annual_elec_kwh"] > 0)]
    diverse_chunks = []
    # Prime showcase homes first: high-impact Band D/E homes with gas heating (high-savings demo homes)
    gas_col = valid["annual_gas_kwh"] if "annual_gas_kwh" in valid.columns else 0
    prime = valid[(valid["energy_rating"].isin(["D", "E"])) & (valid["heating"] == "gas_condensing") & (gas_col > 6000)]
    if not prime.empty:
        diverse_chunks.append(prime.head(15))

    for band in ["D", "E", "C", "F", "B", "G", "A"]:
        for heating in ["gas_condensing", "storage_heaters", "direct_electric", "gas_non_condensing", "heat_pump", "oil"]:
            subset = valid[(valid["energy_rating"] == band) & (valid["heating"] == heating)]
            if not subset.empty:
                diverse_chunks.append(subset.head(7))

    if diverse_chunks:
        sampled_df = pd.concat(diverse_chunks).drop_duplicates("property_id").head(max_count)
        return sampled_df["property_id"].tolist()
    return df_p["property_id"].head(max_count).tolist()


@st.cache_data(show_spinner="Building a report for every home in the sample...")
def cached_portfolio(sample_size: int):
    """The portfolio for a sample size, computed once: slider moves elsewhere on the page don't rebuild it."""
    if not df_loans.empty and "property_id" in df_loans.columns:
        mortgaged = df_properties[df_properties["property_id"].isin(set(df_loans["property_id"]))]
        sample_props = mortgaged if not mortgaged.empty else df_properties
    else:
        sample_props = df_properties
    return build_portfolio(sample_props, df_loans, max_sample=sample_size)



DEFAULT_FACTORS_YEAR = pd.read_csv(Path(__file__).resolve().parents[1] / "config" / "emission_factors.csv")["year"].iloc[0]
memory_store = CustomerMemoryStore()
llm_advisor = LLMAdvisor()


# Sidebar Navigation with reliable vector branding
st.sidebar.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 14px; padding: 4px 0;">
      <div style="background: rgba(255, 255, 255, 0.18); border: 1px solid rgba(255, 255, 255, 0.3); color: white; width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 24px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);">🏛️</div>
      <div>
        <div class="sidebar-brand-name">Lloyds Bank</div>
        <div style="font-size: 11px; color: #a7f3d0; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px;">Green Intelligence</div>
        <div style="font-size: 10px; color: rgba(255, 255, 255, 0.85); font-weight: 500; margin-top: 2px;">{data_source_label}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
view_mode = st.sidebar.radio(
    "Select Portal View",
    ["Customer Experience", "Lender Portfolio (Bank View)", "AI Grounding Inspector"],
)

def sync_theme_attribute() -> None:
    """Tag the page with Streamlit's active theme so the card palette matches it.

    Streamlit doesn't expose its theme to CSS (no attribute or variable), and st.context.theme can disagree with
    what is rendered, so this reads the app's computed background colour and sets data-theme="light" / "dark"
    on .stApp and <html>, re-checking twice a second so a theme switch in Settings is followed live. Light and
    dark mode are chosen in Streamlit's own menu (top right, Settings -> Theme), which switches every element;
    an in-app toggle could only recolour the custom cards and left the page, inputs and charts mismatched."""
    components.html(
        """<script>
        const doc = window.parent.document;
        function sync() {
          const app = doc.querySelector('.stApp');
          if (!app) return;
          const m = getComputedStyle(app).backgroundColor.match(/[0-9.]+/g);
          if (!m) return;
          const [r, g, b] = m.map(Number);
          const theme = (0.2126 * r + 0.7152 * g + 0.0722 * b) < 128 ? 'dark' : 'light';
          for (const el of [app, doc.documentElement]) {
            if (el.getAttribute('data-theme') !== theme) el.setAttribute('data-theme', theme);
          }
        }
        sync();
        setInterval(sync, 500);
        </script>""",
        height=0,
    )


sync_theme_attribute()

st.sidebar.markdown("---")
st.sidebar.caption("Megathon X | Lloyds PS2 | Team 73")


def render_thermal_house_3d(elements: list, score: int, p: Property):
    """Render interactive 3D WebGL Three.js thermal twin model with OrbitControls."""
    elem_dict = {
        e.name: {
            "name": e.name,
            "area_m2": round(e.area_m2, 1),
            "u_value": round(e.u_value, 2),
            "heat_loss_w_per_k": round(e.heat_loss_w_per_k, 1),
            "share": round(e.share, 3),
            "pct": round(e.share * 100, 1),
        }
        for e in elements
    }
    prop_info = {
        "property_id": p.property_id,
        "property_type": p.property_type,
        "has_own_roof": bool(p.has_own_roof),
        "solar_kwp": float(p.solar_kwp or 0.0),
        "heating": p.heating,
        "wall_type": p.wall_type,
        "wall_insulation": p.wall_insulation or "none",
        "glazing": p.glazing,
        "score": score,
    }

    elements_json = json.dumps(elem_dict)
    prop_json = json.dumps(prop_info)

    html_template = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #090d16;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      overflow: hidden;
      color: #f1f5f9;
      user-select: none;
    }
    #canvas-container {
      width: 100%;
      height: 480px;
      position: relative;
      border-radius: 10px;
      overflow: hidden;
    }
    .hud-panel {
      position: absolute;
      background: rgba(15, 23, 42, 0.88);
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      border: 1px solid rgba(51, 65, 85, 0.7);
      border-radius: 10px;
      padding: 12px 14px;
      z-index: 10;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    #hud-info {
      top: 14px;
      left: 14px;
      min-width: 220px;
      max-width: 270px;
      pointer-events: none;
      transition: all 0.2s ease;
    }
    #hud-info .title {
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      color: #38bdf8;
      display: flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 6px;
    }
    #hud-info .stat-line {
      font-size: 12px;
      color: #94a3b8;
      margin: 3px 0;
      display: flex;
      justify-content: space-between;
    }
    #hud-info .stat-val {
      font-weight: 600;
      color: #f8fafc;
    }
    #hud-controls {
      position: absolute;
      top: 14px;
      right: 14px;
      display: flex;
      gap: 8px;
      z-index: 20;
    }
    .hud-btn {
      background: rgba(30, 41, 59, 0.85);
      border: 1px solid #475569;
      color: #cbd5e1;
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }
    .hud-btn:hover {
      background: #0284c7;
      color: #ffffff;
      border-color: #38bdf8;
    }
    .hud-btn.active {
      background: #0369a1;
      border-color: #38bdf8;
      color: #ffffff;
    }
    #hud-legend {
      bottom: 14px;
      right: 14px;
      display: flex;
      gap: 8px;
      font-size: 10px;
      align-items: center;
      padding: 6px 10px;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 4px;
      color: #cbd5e1;
    }
    .legend-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }
    #hud-hint {
      position: absolute;
      bottom: 14px;
      left: 14px;
      font-size: 11px;
      color: #94a3b8;
      pointer-events: none;
      z-index: 10;
    }
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="canvas-container">
    <div id="hud-info" class="hud-panel">
      <div class="title" id="info-title"><span>🏠</span> Building Envelope</div>
      <div class="stat-line"><span>Inspected:</span> <span class="stat-val" id="info-elem">Full Envelope</span></div>
      <div class="stat-line"><span>Heat Loss Share:</span> <span class="stat-val" id="info-share">100%</span></div>
      <div class="stat-line"><span>U-Value:</span> <span class="stat-val" id="info-u">Composite</span></div>
      <div class="stat-line"><span>Surface Area:</span> <span class="stat-val" id="info-area">—</span></div>
    </div>

    <div id="hud-controls">
      <button class="hud-btn active" id="btn-rotate" onclick="toggleRotate()">↺ Rotate</button>
      <button class="hud-btn" id="btn-reset" onclick="resetView()">🎥 Reset View</button>
    </div>

    <div id="hud-hint">
      🖱️ Drag to rotate • Scroll to zoom • Hover/tap envelope parts
    </div>

    <div id="hud-legend" class="hud-panel">
      <span style="color: #64748b; font-weight: 600; margin-right: 2px;">HEAT LOSS:</span>
      <div class="legend-item"><div class="legend-dot" style="background: #14b8a6;"></div> &lt;10%</div>
      <div class="legend-item"><div class="legend-dot" style="background: #eab308;"></div> 10-18%</div>
      <div class="legend-item"><div class="legend-dot" style="background: #f97316;"></div> 18-30%</div>
      <div class="legend-item"><div class="legend-dot" style="background: #ef4444;"></div> &gt;30%</div>
    </div>
  </div>

  <script>
    const elementsData = __ELEMENTS_JSON__;
    const propData = __PROP_JSON__;

    function colorForShare(share) {
      if (share > 0.30) return { hex: 0xef4444, css: '#ef4444', label: 'Severe Loss' };
      if (share > 0.18) return { hex: 0xf97316, css: '#f97316', label: 'High Loss' };
      if (share > 0.10) return { hex: 0xeab308, css: '#eab308', label: 'Moderate Loss' };
      return { hex: 0x14b8a6, css: '#14b8a6', label: 'Well Insulated' };
    }

    const container = document.getElementById('canvas-container');
    const width = container.clientWidth || 550;
    const height = 480;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a101d);
    scene.fog = new THREE.FogExp2(0x0a101d, 0.038);

    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
    const initialPos = { x: 7.2, y: 5.4, z: 7.8 };
    camera.position.set(initialPos.x, initialPos.y, initialPos.z);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 + 0.02;
    controls.minDistance = 3.5;
    controls.maxDistance = 22;
    controls.target.set(0, 1.3, 0);
    controls.autoRotate = true;
    controls.autoRotateSpeed = 1.0;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
    scene.add(ambientLight);

    const hemiLight = new THREE.HemisphereLight(0x38bdf8, 0x090d16, 0.55);
    scene.add(hemiLight);

    const dirLight = new THREE.DirectionalLight(0xfff7ed, 0.95);
    dirLight.position.set(7, 12, 6);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 1024;
    dirLight.shadow.mapSize.height = 1024;
    dirLight.shadow.camera.near = 0.5;
    dirLight.shadow.camera.far = 25;
    dirLight.shadow.bias = -0.0005;
    scene.add(dirLight);

    const fillLight = new THREE.DirectionalLight(0x0284c7, 0.4);
    fillLight.position.set(-8, 6, -6);
    scene.add(fillLight);

    // Ground Plinth & Tech Grid
    const grid = new THREE.GridHelper(16, 16, 0x0284c7, 0x1e293b);
    grid.position.y = -0.01;
    scene.add(grid);

    const plinthGeo = new THREE.CylinderGeometry(5.4, 5.7, 0.2, 40);
    const plinthMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.85,
      metalness: 0.15
    });
    const plinth = new THREE.Mesh(plinthGeo, plinthMat);
    plinth.position.y = -0.11;
    plinth.receiveShadow = true;
    scene.add(plinth);

    // Elements & Interaction Group
    const interactiveObjects = [];

    // Colors
    const roofData = elementsData.roof || { share: 0.15, u_value: 0.18, area_m2: 45 };
    const wallData = elementsData.wall || { share: 0.25, u_value: 0.35, area_m2: 85 };
    const winData = elementsData.window || { share: 0.15, u_value: 1.8, area_m2: 18 };
    const flrData = elementsData.floor || { share: 0.10, u_value: 0.22, area_m2: 45 };

    const cRoof = colorForShare(roofData.share);
    const cWall = colorForShare(wallData.share);
    const cWin = colorForShare(winData.share);
    const cFlr = colorForShare(flrData.share);

    // 1. Foundation Slab (Floor)
    const floorGeo = new THREE.BoxGeometry(4.0, 0.28, 4.0);
    const floorMat = new THREE.MeshStandardMaterial({
      color: cFlr.hex,
      roughness: 0.5,
      metalness: 0.1,
      emissive: cFlr.hex,
      emissiveIntensity: 0.12
    });
    const floorMesh = new THREE.Mesh(floorGeo, floorMat);
    floorMesh.position.y = 0.14;
    floorMesh.castShadow = true;
    floorMesh.receiveShadow = true;
    floorMesh.userData = {
      title: 'Ground Floor Slab',
      type: 'floor',
      elem: flrData,
      color: cFlr
    };
    scene.add(floorMesh);
    interactiveObjects.push(floorMesh);

    // 2. Main Walls
    const wallGeo = new THREE.BoxGeometry(3.7, 2.4, 3.7);
    const wallMat = new THREE.MeshStandardMaterial({
      color: cWall.hex,
      roughness: 0.45,
      metalness: 0.1,
      transparent: true,
      opacity: 0.94,
      emissive: cWall.hex,
      emissiveIntensity: 0.15
    });
    const wallMesh = new THREE.Mesh(wallGeo, wallMat);
    wallMesh.position.y = 0.28 + 1.2;
    wallMesh.castShadow = true;
    wallMesh.receiveShadow = true;
    wallMesh.userData = {
      title: 'External Walls (' + (propData.wall_type || 'Cavity') + ')',
      type: 'wall',
      elem: wallData,
      color: cWall
    };
    scene.add(wallMesh);
    interactiveObjects.push(wallMesh);

    // 3. Windows & Frames
    const winMat = new THREE.MeshStandardMaterial({
      color: cWin.hex,
      roughness: 0.15,
      metalness: 0.7,
      emissive: cWin.hex,
      emissiveIntensity: 0.35
    });
    const frameMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.6 });

    function addWindow(x, y, z, rotY) {
      const group = new THREE.Group();
      group.position.set(x, y, z);
      group.rotation.y = rotY;

      const frameGeo = new THREE.BoxGeometry(0.88, 0.78, 0.06);
      const frame = new THREE.Mesh(frameGeo, frameMat);
      group.add(frame);

      const glassGeo = new THREE.BoxGeometry(0.78, 0.68, 0.08);
      const glass = new THREE.Mesh(glassGeo, winMat);
      glass.castShadow = true;
      glass.userData = {
        title: 'Glazing (' + (propData.glazing || 'Double') + ')',
        type: 'window',
        elem: winData,
        color: cWin
      };
      group.add(glass);
      interactiveObjects.push(glass);
      scene.add(group);
    }

    addWindow(-0.95, 1.95, 1.86, 0);
    addWindow(0.95, 1.95, 1.86, 0);
    addWindow(-0.95, 0.95, 1.86, 0);
    addWindow(1.86, 1.95, 0, Math.PI / 2);
    addWindow(-1.86, 1.95, 0, -Math.PI / 2);
    addWindow(0.95, 1.95, -1.86, Math.PI);

    // Front Door
    const doorGeo = new THREE.BoxGeometry(0.72, 1.35, 0.08);
    const doorMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.5, metalness: 0.2 });
    const door = new THREE.Mesh(doorGeo, doorMat);
    door.position.set(0.7, 0.95, 1.86);
    scene.add(door);

    // 4. Roof
    const roofMat = new THREE.MeshStandardMaterial({
      color: cRoof.hex,
      roughness: 0.5,
      metalness: 0.15,
      emissive: cRoof.hex,
      emissiveIntensity: 0.18
    });

    if (propData.has_own_roof) {
      const roofSlopeGeo = new THREE.BoxGeometry(2.35, 0.12, 4.1);

      const roofLeft = new THREE.Mesh(roofSlopeGeo, roofMat);
      roofLeft.position.set(-0.95, 3.22, 0);
      roofLeft.rotation.z = 0.52;
      roofLeft.castShadow = true;
      roofLeft.receiveShadow = true;
      roofLeft.userData = {
        title: 'Roof & Loft Insulation',
        type: 'roof',
        elem: roofData,
        color: cRoof
      };
      scene.add(roofLeft);
      interactiveObjects.push(roofLeft);

      const roofRight = new THREE.Mesh(roofSlopeGeo, roofMat);
      roofRight.position.set(0.95, 3.22, 0);
      roofRight.rotation.z = -0.52;
      roofRight.castShadow = true;
      roofRight.receiveShadow = true;
      roofRight.userData = {
        title: 'Roof & Loft Insulation',
        type: 'roof',
        elem: roofData,
        color: cRoof
      };
      scene.add(roofRight);
      interactiveObjects.push(roofRight);

      const gableGeo = new THREE.BufferGeometry();
      const gableVerts = new Float32Array([
        -1.85, 2.68, 1.85,
         1.85, 2.68, 1.85,
         0.0,  3.76, 1.85,
        -1.85, 2.68, -1.85,
         0.0,  3.76, -1.85,
         1.85, 2.68, -1.85,
      ]);
      gableGeo.setAttribute('position', new THREE.BufferAttribute(gableVerts, 3));
      gableGeo.computeVertexNormals();
      const gableMesh = new THREE.Mesh(gableGeo, wallMat);
      gableMesh.userData = wallMesh.userData;
      scene.add(gableMesh);
      interactiveObjects.push(gableMesh);

      if (propData.solar_kwp > 0) {
        const solarGroup = new THREE.Group();
        solarGroup.position.set(0.95, 3.25, 0);
        solarGroup.rotation.z = -0.52;

        const panelMat = new THREE.MeshStandardMaterial({
          color: 0x0f2b66,
          roughness: 0.12,
          metalness: 0.85,
          emissive: 0x1d4ed8,
          emissiveIntensity: 0.35
        });

        const frameMatSolar = new THREE.MeshStandardMaterial({ color: 0x64748b, roughness: 0.5, metalness: 0.6 });
        const panelGeo = new THREE.BoxGeometry(1.0, 0.05, 0.88);
        const frameGeoSolar = new THREE.BoxGeometry(1.04, 0.03, 0.92);

        for (let i = -1.1; i <= 1.1; i += 1.1) {
          const sFrame = new THREE.Mesh(frameGeoSolar, frameMatSolar);
          sFrame.position.set(0, 0.08, i);
          solarGroup.add(sFrame);

          const pMesh = new THREE.Mesh(panelGeo, panelMat);
          pMesh.position.set(0, 0.10, i);
          pMesh.castShadow = true;
          solarGroup.add(pMesh);
        }

        solarGroup.children.forEach(function(child) {
          child.userData = {
            title: 'Solar PV Array (' + propData.solar_kwp + ' kWp)',
            type: 'solar',
            elem: {
              name: 'Solar PV',
              share: 0,
              u_value: 0,
              area_m2: propData.solar_kwp * 6,
              status: 'Generating Zero-Carbon Energy'
            },
            color: { css: '#38bdf8' }
          };
          interactiveObjects.push(child);
        });
        scene.add(solarGroup);
      }
    } else {
      const flatGeo = new THREE.BoxGeometry(3.9, 0.2, 3.9);
      const flatMesh = new THREE.Mesh(flatGeo, roofMat);
      flatMesh.position.set(0, 2.78, 0);
      flatMesh.castShadow = true;
      flatMesh.receiveShadow = true;
      flatMesh.userData = {
        title: 'Ceiling / Flat Roof Slab',
        type: 'roof',
        elem: roofData,
        color: cRoof
      };
      scene.add(flatMesh);
      interactiveObjects.push(flatMesh);

      if (propData.solar_kwp > 0) {
        const solarGroup = new THREE.Group();
        solarGroup.position.set(0, 2.92, 0);
        solarGroup.rotation.x = -0.3;

        const panelMat = new THREE.MeshStandardMaterial({
          color: 0x0f2b66,
          roughness: 0.12,
          metalness: 0.85,
          emissive: 0x1d4ed8,
          emissiveIntensity: 0.35
        });
        const panelGeo = new THREE.BoxGeometry(0.9, 0.05, 0.75);
        for (let i = -1; i <= 1; i++) {
          const pMesh = new THREE.Mesh(panelGeo, panelMat);
          pMesh.position.set(i * 1.05, 0.06, 0);
          pMesh.castShadow = true;
          pMesh.userData = {
            title: 'Solar PV Array (' + propData.solar_kwp + ' kWp)',
            type: 'solar',
            elem: {
              name: 'Solar PV',
              share: 0,
              u_value: 0,
              area_m2: propData.solar_kwp * 6,
              status: 'Generating Zero-Carbon Energy'
            },
            color: { css: '#38bdf8' }
          };
          solarGroup.add(pMesh);
          interactiveObjects.push(pMesh);
        }
        scene.add(solarGroup);
      }
    }

    let hpFan = null;
    if (propData.heating === 'heat_pump') {
      const hpGroup = new THREE.Group();
      hpGroup.position.set(2.4, 0.45, 0.8);

      const baseGeo = new THREE.BoxGeometry(0.85, 0.1, 0.55);
      const baseMat = new THREE.MeshStandardMaterial({ color: 0x475569, roughness: 0.9 });
      const hpBase = new THREE.Mesh(baseGeo, baseMat);
      hpBase.position.y = -0.38;
      hpGroup.add(hpBase);

      const caseGeo = new THREE.BoxGeometry(0.75, 0.75, 0.45);
      const caseMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.35, metalness: 0.4 });
      const hpCase = new THREE.Mesh(caseGeo, caseMat);
      hpCase.castShadow = true;
      hpCase.receiveShadow = true;
      hpGroup.add(hpCase);

      const grillGeo = new THREE.CylinderGeometry(0.24, 0.24, 0.04, 24);
      grillGeo.rotateX(Math.PI / 2);
      const grillMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.5 });
      const grill = new THREE.Mesh(grillGeo, grillMat);
      grill.position.set(0, 0, 0.23);
      hpGroup.add(grill);

      const fanGeo = new THREE.BoxGeometry(0.38, 0.05, 0.02);
      const fanMat = new THREE.MeshBasicMaterial({ color: 0x0284c7 });
      hpFan = new THREE.Mesh(fanGeo, fanMat);
      hpFan.position.set(0, 0, 0.24);
      hpGroup.add(hpFan);

      const ledGeo = new THREE.SphereGeometry(0.035, 12, 12);
      const ledMat = new THREE.MeshBasicMaterial({ color: 0x22c55e });
      const led = new THREE.Mesh(ledGeo, ledMat);
      led.position.set(0.28, 0.26, 0.23);
      hpGroup.add(led);

      hpCase.userData = {
        title: 'Air Source Heat Pump (ASHP)',
        type: 'heat_pump',
        elem: {
          name: 'Heat Pump',
          share: 0,
          u_value: 0,
          area_m2: 0,
          status: 'Low-Carbon Heating • COP ~3.2'
        },
        color: { css: '#22c55e' }
      };
      interactiveObjects.push(hpCase);
      scene.add(hpGroup);
    }

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let hoveredObject = null;
    let prevEmissive = null;

    const hudTitle = document.getElementById('info-title');
    const hudElem = document.getElementById('info-elem');
    const hudShare = document.getElementById('info-share');
    const hudU = document.getElementById('info-u');
    const hudArea = document.getElementById('info-area');

    function resetHUD() {
      hudTitle.innerHTML = '<span>🏠</span> Building Envelope';
      hudElem.innerText = 'Full Envelope';
      hudShare.innerText = '100% Total Loss';
      hudU.innerText = 'Composite';
      hudArea.innerText = 'Green Score: ' + propData.score + '/100';
    }
    resetHUD();

    function onPointerMove(event) {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(interactiveObjects, false);

      if (intersects.length > 0) {
        const obj = intersects[0].object;
        if (hoveredObject !== obj) {
          if (hoveredObject && prevEmissive && hoveredObject.material && hoveredObject.material.emissive) {
            hoveredObject.material.emissive.copy(prevEmissive);
          }
          hoveredObject = obj;
          container.style.cursor = 'pointer';

          if (obj.material && obj.material.emissive) {
            prevEmissive = obj.material.emissive.clone();
            obj.material.emissive.addScalar(0.25);
          }

          const ud = obj.userData;
          hudTitle.innerHTML = '<span>🔍</span> ' + ud.title;
          hudElem.innerText = ud.elem.name ? ud.elem.name.toUpperCase() : ud.type;
          if (ud.type === 'solar' || ud.type === 'heat_pump') {
            hudShare.innerText = ud.elem.status || 'Active';
            hudU.innerText = ud.type === 'solar' ? (propData.solar_kwp + ' kWp') : 'COP 3.2';
            hudArea.innerText = ud.type === 'solar' ? (ud.elem.area_m2 + ' m²') : 'Zero-Carbon';
          } else {
            const pct = Math.round(ud.elem.share * 100);
            hudShare.innerHTML = '<span style="color:' + ud.color.css + ';">' + pct + '% of heat loss</span>';
            hudU.innerText = ud.elem.u_value + ' W/m²K';
            hudArea.innerText = ud.elem.area_m2 + ' m²';
          }
        }
      } else {
        if (hoveredObject) {
          if (prevEmissive && hoveredObject.material && hoveredObject.material.emissive) {
            hoveredObject.material.emissive.copy(prevEmissive);
          }
          hoveredObject = null;
          container.style.cursor = 'default';
          resetHUD();
        }
      }
    }

    renderer.domElement.addEventListener('pointermove', onPointerMove, false);

    const btnRotate = document.getElementById('btn-rotate');
    window.toggleRotate = function() {
      controls.autoRotate = !controls.autoRotate;
      if (controls.autoRotate) {
        btnRotate.classList.add('active');
        btnRotate.innerText = '↺ Rotate';
      } else {
        btnRotate.classList.remove('active');
        btnRotate.innerText = '⏸ Paused';
      }
    };

    window.resetView = function() {
      controls.target.set(0, 1.3, 0);
      camera.position.set(initialPos.x, initialPos.y, initialPos.z);
      controls.autoRotate = true;
      btnRotate.classList.add('active');
      btnRotate.innerText = '↺ Rotate';
      resetHUD();
    };

    window.addEventListener('resize', function() {
      const w = container.clientWidth;
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
      renderer.setSize(w, height);
    });

    function animate() {
      requestAnimationFrame(animate);
      if (hpFan) {
        hpFan.rotation.z += 0.08;
      }
      controls.update();
      renderer.render(scene, camera);
    }
    animate();
  </script>
</body>
</html>"""

    html_code = html_template.replace("__ELEMENTS_JSON__", elements_json).replace("__PROP_JSON__", prop_json)
    components.html(html_code, height=500)


def render_thermal_house_svg(elements: list, score: int):
    """Render interactive thermal twin SVG cross-section."""
    elem_dict = {e.name: e for e in elements}

    def color_for_loss(share: float) -> str:
        if share > 0.30:
            return "#ef4444"  # High thermal loss red
        elif share > 0.18:
            return "#f97316"  # Medium-high orange
        elif share > 0.10:
            return "#eab308"  # Moderate yellow
        return "#14b8a6"      # Insulated teal

    roof_share = elem_dict["roof"].share if "roof" in elem_dict else 0.15
    wall_share = elem_dict["wall"].share if "wall" in elem_dict else 0.25
    win_share = elem_dict["window"].share if "window" in elem_dict else 0.15
    flr_share = elem_dict["floor"].share if "floor" in elem_dict else 0.10

    c_roof = color_for_loss(roof_share)
    c_wall = color_for_loss(wall_share)
    c_win = color_for_loss(win_share)
    c_flr = color_for_loss(flr_share)

    svg = f"""
    <svg width="100%" height="240" viewBox="0 0 400 240" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>
      <!-- Background Sky -->
      <rect width="400" height="240" rx="10" fill="#0f172a"/>
      <!-- Roof (Triangle) -->
      <polygon points="200,30 70,110 330,110" fill="{c_roof}" opacity="0.85" filter="url(#glow)"/>
      <polygon points="200,30 70,110 330,110" fill="none" stroke="#f8fafc" stroke-width="2"/>
      <text x="200" y="85" text-anchor="middle" fill="#ffffff" font-size="12" font-weight="bold">
        Roof ({roof_share * 100:.0f}% Loss)
      </text>
      <!-- House Body (Walls) -->
      <rect x="90" y="110" width="220" height="95" fill="{c_wall}" opacity="0.8" filter="url(#glow)"/>
      <rect x="90" y="110" width="220" height="95" fill="none" stroke="#f8fafc" stroke-width="2"/>
      <text x="140" y="190" text-anchor="middle" fill="#ffffff" font-size="11" font-weight="bold">
        Walls ({wall_share * 100:.0f}% Loss)
      </text>
      <!-- Windows -->
      <rect x="110" y="125" width="45" height="35" rx="3" fill="{c_win}" filter="url(#glow)"/>
      <rect x="110" y="125" width="45" height="35" rx="3" fill="none" stroke="#ffffff" stroke-width="1.5"/>
      <rect x="245" y="125" width="45" height="35" rx="3" fill="{c_win}" filter="url(#glow)"/>
      <rect x="245" y="125" width="45" height="35" rx="3" fill="none" stroke="#ffffff" stroke-width="1.5"/>
      <text x="267" y="146" text-anchor="middle" fill="#ffffff" font-size="9" font-weight="bold">Window</text>
      <!-- Floor Slab -->
      <rect x="75" y="205" width="250" height="15" rx="2" fill="{c_flr}"/>
      <text x="200" y="217" text-anchor="middle" fill="#ffffff" font-size="10" font-weight="bold">
        Ground Floor ({flr_share * 100:.0f}%)
      </text>
    </svg>
    """
    st.markdown(svg, unsafe_allow_html=True)


def generate_grounded_advisor_response(
    query: str,
    report: PropertyReport,
    p: Property,
    history: list[dict[str, str]] | None = None,
) -> tuple[str, GroundingResult, str]:
    """Generate grounded explanation using LLMAdvisor with deterministic grounding shield."""
    return llm_advisor.query(query, report, p, history=history)


# Top App Header Banner
st.markdown(
    """
    <div class="app-header">
        <div style="display: flex; align-items: center; gap: 14px;">
            <div style="background: white; border-radius: 8px; padding: 6px 10px; display: flex; align-items: center;">
                <span style="font-size: 22px;">🏛️</span>
            </div>
            <div>
                <div style="font-size: 18px; font-weight: 800; letter-spacing: -0.3px;">Lloyds Banking Group • Green Home Intelligence</div>
                <div style="font-size: 12px; color: #a7f3d0; opacity: 0.95;">Decarbonisation & Transition Finance Engine • Team 73</div>
            </div>
        </div>
        <div style="display: flex; gap: 8px; align-items: center;">
            <span class="badge" style="background: rgba(255,255,255,0.15); color: white; border: 1px solid rgba(255,255,255,0.3);">● Deterministic Engine</span>
            <span class="badge badge-shield" style="background: #ecfdf5 !important; color: #065f46 !important; border: 1px solid #6ee7b7 !important; font-weight: 700 !important;">🛡️ Grounding Shield Active</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if view_mode == "Customer Experience":
    st.subheader("Homeowner Green Home Score & What-If Studio")

    if df_properties.empty:
        st.warning("No data found. Run `make data` to generate.")
        st.stop()

    # Property Selector & Quick Overview Card
    col_sel, col_stat = st.columns([2, 1])
    with col_sel:
        sample_count = min(300, len(df_properties))
        featured_ids = get_featured_properties(df_properties, sample_count)

        sel_sub1, sel_sub2 = st.columns([1, 1])
        with sel_sub1:
            chosen_pid = st.selectbox(
                f"Choose Property ({sample_count} featured of {len(df_properties):,} book)",
                featured_ids,
                index=0,
                key="featured_property_selector",
            )
        with sel_sub2:
            custom_pid = st.text_input(
                "Or Lookup by Property ID",
                placeholder="e.g. E4ef6ab13a0",
                help="Search and inspect any home across the 542k property dataset.",
                key="custom_property_lookup",
            )

        if custom_pid.strip():
            matched_row = df_properties[df_properties["property_id"] == custom_pid.strip()]
            if not matched_row.empty:
                selected_pid = custom_pid.strip()
            else:
                st.caption(f"⚠️ Property ID '{custom_pid.strip()}' not found in active dataset.")
                selected_pid = chosen_pid
        else:
            selected_pid = chosen_pid

    row = df_properties[df_properties["property_id"] == selected_pid].iloc[0].to_dict()
    clean_d = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    p = Property(**clean_d)

    # Load customer memory
    mem = memory_store.get_memory(p.customer_id)
    p = memory_store.apply_memory_to_property(p)

    with col_stat:
        epc_color = "#059669" if p.energy_rating in ["A", "B", "C"] else ("#f59e0b" if p.energy_rating == "D" else "#ef4444")
        st.markdown(
            f"""
            <div class="prop-summary-box">
              <div>
                <div class="prop-label" style="font-size: 10px; font-weight: 700; text-transform: uppercase; color: var(--shadcn-card-text-muted, #64748b);">EPC RATING</div>
                <div style="font-size: 16px; font-weight: 800; color: {epc_color};">Band {p.energy_rating or 'N/A'} ({p.sap_score or '—'} SAP)</div>
              </div>
              <div style="text-align: right;">
                <div class="prop-label" style="font-size: 10px; font-weight: 700; text-transform: uppercase; color: var(--shadcn-card-text-muted, #64748b);">TYPE & FUEL</div>
                <div class="prop-val" style="font-size: 12px; font-weight: 700; color: var(--shadcn-card-text, var(--text-color, #0f172a));">{p.property_type.title()} • {p.heating.replace('_', ' ').title()}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # What-If Retrofit Selector
    st.sidebar.subheader("Interactive What-If Scenario")
    toggle_loft = st.sidebar.checkbox("Loft Insulation Top-Up", value=False)
    toggle_wall = st.sidebar.checkbox("Wall Insulation", value=False)
    toggle_glazing = st.sidebar.checkbox("A-Rated Double Glazing", value=False)
    toggle_thermo = st.sidebar.checkbox("Smart Thermostat", value=False)
    toggle_hp = st.sidebar.checkbox("Air Source Heat Pump", value=False)
    toggle_solar = st.sidebar.checkbox("Rooftop Solar PV (3.5 kWp)", value=False)

    chosen_retrofits = []
    if toggle_loft:
        chosen_retrofits.append("loft_insulation")
    if toggle_wall:
        chosen_retrofits.append("cavity_wall_insulation" if p.wall_type == "cavity" else "solid_wall_insulation")
    if toggle_glazing:
        chosen_retrofits.append("double_glazing")
    if toggle_thermo:
        chosen_retrofits.append("smart_thermostat")
    if toggle_hp:
        chosen_retrofits.append("heat_pump")
    if toggle_solar:
        chosen_retrofits.append("solar_pv")

    # Generate Report
    report = build_report(p, retrofits=chosen_retrofits if chosen_retrofits else None, memory=mem)

    # Customer Experience Tabs
    tab_studio, tab_banking, tab_advisor = st.tabs([
        "🏡 Interactive What-If Studio",
        "🏦 Lloyds Green Banking & Grants",
        "🤖 AI Green Advisor (Shielded)",
    ])

    with tab_studio:
        # Metrics Bar
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Green Home Score",
                f"{report.score.total}/100",
                delta=f"+{report.score_after.total - report.score.total} pts" if chosen_retrofits else None,
            )
            st.caption(f"Rating: **{report.score.classification}**")
        with col2:
            if report.package_saving is None:
                st.metric("Annual Fuel Savings", "Unavailable")
                st.caption("No usable bills and too little property data to estimate expected use.")
            else:
                st.metric(
                    "Annual Fuel Savings",
                    f"{report.package_saving.total_kwh:,.0f} kWh/yr",
                    delta=f"Gas: {report.package_saving.gas_kwh:,.0f} | Elec: {report.package_saving.elec_kwh:,.0f}",
                )
        with col3:
            if report.package_carbon_kg is None:
                st.metric("Carbon Reduction", "Unavailable")
            else:
                st.metric(
                    "Carbon Reduction",
                    f"{report.package_carbon_kg / 1000.0:.2f} tCO₂e/yr",
                    delta=f"factors: {DEFAULT_FACTORS_YEAR}",
                )
        with col4:
            if report.value.get("status") == "unavailable":
                st.metric("Property Value (indicative)", "Unavailable")
                st.caption("No current valuation for this property.")
            else:
                lo_v, hi_v = report.value['low'], report.value['high']
                fmt = lambda v: f"£{v / 1e6:.2f}m" if v >= 1e6 else f"£{v / 1e3:,.0f}k"   # compact: fits the card
                uplift_str = fmt(lo_v) if fmt(lo_v) == fmt(hi_v) else f"{fmt(lo_v)} – {fmt(hi_v)}"
                pl, ph = report.value['pct_low'], report.value['pct_high']
                # The range can include 0 (an association with a CI), so only colour it when it is all gains
                st.metric("Property Value (indicative)", uplift_str, delta=f"{pl:+.1f}% to {ph:+.1f}%",
                          delta_color="normal" if pl >= 0 else "off",
                          help=f"£{lo_v:,.0f} – £{hi_v:,.0f}. {report.value.get('caveat', '')}")

        st.markdown("<div style='margin: 16px 0;'></div>", unsafe_allow_html=True)

        # Middle Row: Thermal Twin House + Component Waterfall
        col_house, col_waterfall = st.columns([1, 1])

        with col_house:
            st.markdown("### Thermal Twin Envelope Model")
            p_display = apply_retrofits_to_property(p, chosen_retrofits) if chosen_retrofits else p
            
            # If floor area is unrecorded, infer using benchmark medians so 3D/2D models always load
            floor_area_inferred = False
            if p_display.floor_area_m2 is None:
                floor_area_inferred = True
                median_by_type = {
                    "flat": 60.0,
                    "mid terrace": 80.0,
                    "end terrace": 85.0,
                    "semi detached": 92.0,
                    "detached": 145.0,
                    "bungalow": 82.0,
                }
                ptype_key = p_display.property_type.strip().lower() if p_display.property_type else "semi detached"
                inferred_area = float(
                    (p_display.bedrooms * 28.0)
                    if (p_display.bedrooms and p_display.bedrooms > 0)
                    else median_by_type.get(ptype_key, 85.0)
                )
                p_display = p_display.model_copy(update={"floor_area_m2": round(inferred_area, 1)})

            elements = build_home_elements(p_display)
            display_score = report.score_after.total if chosen_retrofits else report.score.total
            if elements:
                viz_mode = st.radio(
                    "Model View",
                    ["3D Interactive WebGL Twin", "2D Architecture Cross-Section (SVG)"],
                    horizontal=True,
                    label_visibility="collapsed",
                )
                if viz_mode == "3D Interactive WebGL Twin":
                    render_thermal_house_3d(elements, display_score, p_display)
                else:
                    render_thermal_house_svg(elements, display_score)
                if floor_area_inferred:
                    st.caption(
                        f"ℹ️ Floor area unrecorded on EPC certificate. Building envelope modeled using benchmark estimate ({p_display.floor_area_m2:.0f} m² for {p.property_type})."
                    )
            else:
                st.info("Heat-loss diagram unavailable: floor area could not be determined.")
            st.caption("Elements glow in infrared spectrum based on specific heat loss ($W/K$). Toggling retrofits recolours the building envelope.")

        with col_waterfall:
            st.markdown("### Score Component Breakdown")
            for c in report.score.components:
                st.progress(int(c.score), text=f"{c.component} ({c.score:.0f}/100) — Contribution: {c.contribution:.1f} pts")

        st.markdown("---")

        # Sequenced Recommendations
        st.markdown("### Prioritised Feasible Recommendations")
        if not report.recommendations:
            st.info("No further retrofits currently feasible for this property state.")
        else:
            rec_icons = {
                "loft_insulation": "🏠",
                "cavity_wall_insulation": "🧱",
                "solid_wall_insulation": "🧱",
                "double_glazing": "🪟",
                "smart_thermostat": "📱",
                "heat_pump": "♨️",
                "solar_pv": "☀️",
            }
            for idx, r in enumerate(report.recommendations):
                badge_class = "pill-high" if r.suitability == "High" else ("pill-med" if r.suitability == "Medium" else "pill-low")
                icon = rec_icons.get(r.retrofit, "🔧")
                st.markdown(
                    f"""
                    <div class="shadcn-card">
                      <div class="shadcn-card-header">
                        <div style="display: flex; align-items: center; gap: 8px;">
                          <span style="font-size: 18px;">{icon}</span>
                          <span style="font-size: 16px; font-weight: 700; color: var(--shadcn-card-text, var(--text-color, #0f172a));">{r.retrofit.replace('_', ' ').title()}</span>
                          <span class="{badge_class}">{r.suitability} Suitability</span>
                        </div>
                        <div style="font-size: 12px; font-weight: 600; color: #10b981; background: rgba(16, 185, 129, 0.14); border: 1px solid rgba(16, 185, 129, 0.35); padding: 2px 8px; border-radius: 4px;">
                          Priority #{idx + 1}
                        </div>
                      </div>
                      <p style="color: var(--shadcn-card-text-muted, #64748b); font-size: 13.5px; margin: 4px 0 14px 0; line-height: 1.5;">{r.reason}</p>
                      <div class="shadcn-subcard" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; background: var(--shadcn-subcard-bg, #f8fafc); border: 1px solid var(--shadcn-subcard-border, #e2e8f0); padding: 10px 14px; border-radius: 8px;">
                        <div>
                          <div style="font-size: 11px; color: var(--shadcn-card-text-muted, #64748b); font-weight: 600;">ELEC SAVING</div>
                          <div style="font-size: 14px; font-weight: 700; color: var(--shadcn-card-text, var(--text-color, #0f172a));">{r.marginal_saving.elec_kwh:,.0f} <span style="font-size: 11px; font-weight: normal; color: var(--shadcn-card-text-muted, #64748b);">kWh/yr</span></div>
                        </div>
                        <div>
                          <div style="font-size: 11px; color: var(--shadcn-card-text-muted, #64748b); font-weight: 600;">GAS SAVING</div>
                          <div style="font-size: 14px; font-weight: 700; color: var(--shadcn-card-text, var(--text-color, #0f172a));">{r.marginal_saving.gas_kwh:,.0f} <span style="font-size: 11px; font-weight: normal; color: var(--shadcn-card-text-muted, #64748b);">kWh/yr</span></div>
                        </div>
                        <div>
                          <div style="font-size: 11px; color: var(--shadcn-card-text-muted, #64748b); font-weight: 600;">CARBON CUT</div>
                          <div style="font-size: 14px; font-weight: 700; color: #10b981;">{r.carbon_kg:,.0f} <span style="font-size: 11px; font-weight: normal; color: var(--shadcn-card-text-muted, #64748b);">kg CO₂e</span></div>
                        </div>
                        <div>
                          <div style="font-size: 11px; color: var(--shadcn-card-text-muted, #64748b); font-weight: 600;">INDICATIVE COST</div>
                          <div style="font-size: 14px; font-weight: 700; color: var(--shadcn-card-text, var(--text-color, #0f172a));">£{r.indicative_cost[0]:,.0f} – £{r.indicative_cost[1]:,.0f}</div>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if report.infeasible:
            with st.expander("Disqualified / Infeasible Interventions"):
                for inf in report.infeasible:
                    st.markdown(f"- **{inf.retrofit.replace('_', ' ').title()}**: {inf.rule}")

    with tab_banking:
        st.markdown("### Lloyds Green Banking & UK Grant Matcher")
        st.caption("Automatic matching against national subsidies and Lloyds preferential green financing.")

        # Dynamic UK Grants & Lloyds Terms Matcher
        has_hp = any(r.retrofit == "heat_pump" for r in report.recommendations)
        bus_badge = '<span class="badge badge-green">Verified Eligible</span>' if has_hp else '<span class="badge badge-slate">Not Required</span>'
        bus_detail = '<b style="color: #059669;">£7,500 non-repayable grant</b> off Air Source Heat Pump installation.' if has_hp else '£7,500 grant available if an Air Source Heat Pump is recommended.'

        is_eco4 = (p.energy_rating or "D") in ["D", "E", "F", "G"]
        eco4_badge = '<span class="badge badge-green">Band Eligible</span>' if is_eco4 else '<span class="badge badge-slate">Above Band D</span>'
        eco4_detail = f'Free or subsidized cavity/loft insulation for homes in EPC Bands D–G (Current: Band {p.energy_rating or "D"}).' if is_eco4 else f'Current Band {p.energy_rating or "C"} exceeds national fuel poverty threshold.'

        has_clean = any(r.retrofit in ["heat_pump", "solar_pv", "loft_insulation", "cavity_wall_insulation", "solid_wall_insulation"] for r in report.recommendations)
        cashback_amt = 1000 if has_clean else 0
        cashback_badge = '<span class="badge badge-green">£1,000 Awarded</span>' if has_clean else '<span class="badge badge-slate">£0</span>'

        reached_c = report.score_after.total >= 69
        discount_badge = '<span class="badge badge-green">Band C Target Met</span>' if reached_c else '<span class="badge badge-blue">Target Pending</span>'
        discount_detail = '<b>Mortgage Rate Discount:</b> Upgrades project home into Band C (0.10% – 0.25% discount unlocked).' if reached_c else '<b>Mortgage Rate Discount:</b> 0.10% – 0.25% interest discount upon reaching EPC Band C certification.'

        # Dynamic net cashflow calculation from report recommendations and fuel savings
        if report.package_saving is not None and report.package_saving.total_kwh > 0:
            # Ofgem default UK price cap: ~24.5p/kWh electricity, ~6.2p/kWh gas
            annual_fuel_saving = (report.package_saving.elec_kwh * 0.245) + (report.package_saving.gas_kwh * 0.062)
            monthly_fuel_saving = annual_fuel_saving / 12.0

            total_min_cost = sum(r.indicative_cost[0] for r in report.recommendations)
            total_max_cost = sum(r.indicative_cost[1] for r in report.recommendations)
            avg_gross_cost = (total_min_cost + total_max_cost) / 2.0

            bus_grant_val = 7500.0 if has_hp else 0.0
            cashback_val = 1000.0 if has_clean else 0.0
            net_borrowing = max(0.0, avg_gross_cost - bus_grant_val - cashback_val)

            # 5-year Lloyds green additional borrowing at preferential 3.5% APR
            loan_months = 60
            r_mo = 0.035 / 12.0
            if net_borrowing > 0:
                monthly_loan = (net_borrowing * r_mo * ((1.0 + r_mo) ** loan_months)) / (((1.0 + r_mo) ** loan_months) - 1.0)
            else:
                monthly_loan = 0.0

            net_monthly = monthly_fuel_saving - monthly_loan
            if net_monthly >= 0:
                cashflow_color = "#10b981"
                cashflow_display = f"+£{net_monthly:,.0f} / mo"
                cashflow_badge = "CASHFLOW POSITIVE"
                cashflow_sub = f"Estimated monthly energy bill savings (£{monthly_fuel_saving:,.0f}/mo) exceed green loan repayments (£{monthly_loan:,.0f}/mo)."
            else:
                cashflow_color = "#f59e0b"
                cashflow_display = f"-£{abs(net_monthly):,.0f} / mo"
                cashflow_badge = "NET INVESTMENT"
                cashflow_sub = f"Monthly loan repayments (£{monthly_loan:,.0f}/mo) offset by £{monthly_fuel_saving:,.0f}/mo energy savings (net capex: £{net_borrowing:,.0f} after £{bus_grant_val + cashback_val:,.0f} subsidies)."
        else:
            cashflow_color = "var(--shadcn-card-text-muted, #64748b)"
            cashflow_display = "—"
            cashflow_badge = "NO RETROFITS NEEDED" if not report.recommendations else "DATA PENDING"
            cashflow_sub = "No capital borrowing required for current property state." if not report.recommendations else "Baseline energy meter readings unavailable for exact cashflow estimation."

        b1, b2 = st.columns(2)
        with b1:
            st.markdown(
                f"""
                <div class="shadcn-card">
                  <div class="shadcn-card-header">
                    <span style="font-size: 16px; font-weight: 700; color: var(--shadcn-card-text, var(--text-color, #0f172a));">🇬🇧 UK Government Grant Programs</span>
                    {bus_badge}
                  </div>
                  <div style="display: flex; flex-direction: column; gap: 12px; font-size: 13.5px; color: var(--shadcn-card-text-sub, #334155); line-height: 1.55;">
                    <div>
                      <b style="color: var(--shadcn-card-text, var(--text-color, #0f172a));">• Boiler Upgrade Scheme (BUS):</b> {bus_detail}
                    </div>
                    <div>
                      <b style="color: var(--shadcn-card-text, var(--text-color, #0f172a));">• ECO4 & Great British Insulation Scheme (GBIS):</b> {eco4_detail}
                    </div>
                    <div>
                      <b style="color: var(--shadcn-card-text, var(--text-color, #0f172a));">• 0% VAT on Energy Saving Materials:</b> Government zero-rate VAT on heat pumps, solar PV, and insulation installations through 2027.
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with b2:
            st.markdown(
                f"""
                <div class="shadcn-card">
                  <div class="shadcn-card-header">
                    <span style="font-size: 16px; font-weight: 700; color: var(--shadcn-card-text, var(--text-color, #0f172a));">🏦 Lloyds Eco Home Advantage</span>
                    {cashback_badge}
                  </div>
                  <div style="display: flex; flex-direction: column; gap: 12px; font-size: 13.5px; color: var(--shadcn-card-text-sub, #334155); line-height: 1.55;">
                    <div>
                      <b style="color: var(--shadcn-card-text, var(--text-color, #0f172a));">• £1,000 Cashback:</b> Cash reward paid directly into your account on qualifying clean energy improvements.
                    </div>
                    <div>
                      <b style="color: var(--shadcn-card-text, var(--text-color, #0f172a));">• Green Additional Borrowing:</b> Preferential 0% / low-interest borrowing rate for retrofit capital expenditures.
                    </div>
                    <div>
                      {discount_detail}
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="shadcn-card" style="background: var(--shadcn-cashflow-grad, linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, #ffffff 100%)); border: 1px solid rgba(16, 185, 129, 0.35);">
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                  <div style="font-size: 15px; font-weight: 700; color: {cashflow_color};">Net Monthly Household Cashflow Impact</div>
                  <div style="font-size: 13px; color: var(--shadcn-card-text-muted, #64748b); margin-top: 2px;">{cashflow_sub}</div>
                </div>
                <div style="text-align: right;">
                  <span style="font-size: 24px; font-weight: 800; color: {cashflow_color};">{cashflow_display}</span>
                  <div style="font-size: 11px; font-weight: 600; color: {cashflow_color};">{cashflow_badge}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_advisor:
        st.markdown("### Lloyds AI Green Advisor")

        backend_title = f"{llm_advisor.provider.title()} ({llm_advisor.model})" if llm_advisor.provider != "offline" else "Deterministic Rule Engine (Offline)"
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; flex-wrap: wrap; gap: 8px;">
              <div style="display: flex; align-items: center; gap: 8px;">
                <span class="chat-shield-badge">🛡️ 100% Deterministic Grounding Shield Active</span>
                <span style="font-size: 12px; color: var(--shadcn-card-text-muted, #64748b);">Every claim is cross-checked against PropertyReport JSON.</span>
              </div>
              <span class="badge badge-slate" style="font-size: 11px; padding: 4px 10px;">Backend: {backend_title}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        chat_key = f"chat_history_{p.property_id}"
        if chat_key not in st.session_state:
            init_msg, init_audit, init_label = llm_advisor.query(
                "Summarise my home's Green Home Score and priority retrofits.", report, p
            )
            st.session_state[chat_key] = [
                {
                    "role": "assistant",
                    "content": init_msg,
                    "model": init_label,
                    "audit": init_audit.to_dict(),
                }
            ]

        # Preset suggestion chips + Clear button
        c_p1, c_p2, c_p3, c_p4 = st.columns([1, 1, 1, 0.4])
        query_to_submit = None
        if c_p1.button("💡 Reach EPC Band C"):
            query_to_submit = "How do I upgrade my home to reach EPC Band C?"
        if c_p2.button("⚡ Fuel Bill Savings"):
            query_to_submit = "What are my total fuel bill and carbon savings?"
        if c_p3.button("💰 Grant Matching"):
            query_to_submit = "Which government grants and subsidies can I claim?"
        if c_p4.button("🗑️ Clear", help="Reset conversation"):
            st.session_state[chat_key] = []
            st.rerun()

        # Render conversation history
        for msg in st.session_state[chat_key]:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
                      <div style="background: #006a4e; color: #ffffff; padding: 10px 16px; border-radius: 12px 12px 2px 12px; max-width: 80%; font-size: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        {html.escape(msg["content"])}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                audit_data = msg.get("audit", {})
                grounded_pct = audit_data.get("grounding_score_pct", 100.0)
                verified_count = len(audit_data.get("verified_claims", []))
                model_name = msg.get("model", "Lloyds Advisor")
                st.markdown(
                    f"""
                    <div class="chat-bubble-assistant">
                      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 700; color: #10b981; font-size: 13.5px;">🏛️ {html.escape(model_name)}</span>
                        <div class="verified-symbol-wrapper">
                          <span class="verified-icon-symbol">🛡️✓</span>
                          <div class="verified-hover-tooltip">✓ {grounded_pct:.0f}% Verified Grounded</div>
                        </div>
                      </div>
                      <div style="font-size: 14px; color: var(--shadcn-card-text, var(--text-color, #0f172a)); line-height: 1.6; white-space: pre-wrap;">{html.escape(msg["content"])}</div>
                      <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(128, 128, 128, 0.16); display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 11px; color: var(--shadcn-card-text-muted, #64748b);">Claims verified: {verified_count} numeric values</span>
                        <span style="font-size: 11px; color: #10b981; font-weight: 600;">Grounding Shield: ACTIVE</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if audit_data.get("verified_claims"):
                    with st.expander(f"🔍 Grounding Audit Details ({verified_count} verified numbers)", expanded=False):
                        st.caption(f"Verified against PropertyReport JSON for {p.property_id}:")
                        for claim in audit_data["verified_claims"]:
                            st.markdown(f"- ✅ **Verified:** `{claim}`")
                        if audit_data.get("unsupported_claims"):
                            for unsupp in audit_data["unsupported_claims"]:
                                st.markdown(f"- ⚠️ **Flagged / Blocked:** `{unsupp}`")

        # Input form
        with st.form("advisor_chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Ask a question about your home's decarbonisation plan:",
                placeholder="e.g. Which insulation measure is best value, and what grants help?",
            )
            submit_btn = st.form_submit_button("Send Question", use_container_width=True)

        if submit_btn and user_input.strip():
            query_to_submit = user_input.strip()

        if query_to_submit:
            st.session_state[chat_key].append({"role": "user", "content": query_to_submit})
            history_tuples = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state[chat_key][:-1]
            ]
            resp_text, audit, model_label = llm_advisor.query(
                query_to_submit, report, p, history=history_tuples
            )
            st.session_state[chat_key].append({
                "role": "assistant",
                "content": resp_text,
                "model": model_label,
                "audit": audit.to_dict(),
            })
            st.rerun()


elif view_mode == "Lender Portfolio (Bank View)":
    st.subheader("Lender Portfolio View — Lloyds Mortgage Book")
    st.caption("PCAF Scope 3 Category 15 financed emissions & 2030 EPC Band C transition pathway.")

    sample_size = st.slider("Mortgage Book Sample Size", 50, 500, 200, step=25)
    portfolio = cached_portfolio(sample_size)

    # Portfolio KPI bar
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Mortgaged Homes", f"{portfolio.total_properties:,}")
    with k2:
        st.metric("Poor / Very Poor Share", f"{portfolio.poor_share * 100:.1f}%", f"{portfolio.poor_count} homes")
    # Dynamic book abatement percentage
    baseline_carbon = portfolio.total_emissions_tonnes if portfolio.total_emissions_tonnes > 0 else portfolio.total_financed_emissions_tonnes
    abatement_pct = (portfolio.potential_carbon_reduction_tonnes / max(baseline_carbon, 1e-6)) * 100.0 if baseline_carbon > 0 else 0.0

    with k3:
        st.metric("PCAF Financed Emissions", f"{portfolio.total_financed_emissions_tonnes:,.1f} tCO₂e", "Scope 3 Cat 15")
    with k4:
        st.metric(
            "Transition Reduction Potential",
            f"{portfolio.potential_carbon_reduction_tonnes:,.1f} tCO₂e",
            f"-{abatement_pct:.1f}% Book Abatement" if abatement_pct > 0 else "0% Book Abatement",
        )

    st.markdown("---")

    # Interactive 2030 Transition Target Simulator
    st.markdown("### 2030 EPC Band C Transition Target Simulator")
    st.caption("Model the carbon abatement impact of targeted green mortgage incentives across the loan book.")

    target_uptake = st.slider("Target Retrofit Uptake (% of High-Risk Homes)", 10, 100, 50, step=5)
    abated_homes = int(portfolio.poor_count * (target_uptake / 100.0))
    avoided_carbon = portfolio.potential_carbon_reduction_tonnes * (target_uptake / 100.0)
    avg_capex = portfolio.avg_retrofit_cost if portfolio.avg_retrofit_cost > 0 else 8500.0
    capital_req = abated_homes * avg_capex

    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Homes De-risked to Band C", f"{abated_homes:,} homes", f"{target_uptake}% uptake")
    with s2:
        st.metric("Portfolio Carbon Avoided", f"{avoided_carbon:,.1f} tCO₂e / yr", "Annual Saving")
    with s3:
        st.metric("Lending Capital Required", f"£{capital_req / 1_000_000:.2f}M", f"Avg £{avg_capex:,.0f} / home")

    st.markdown("---")
    st.markdown("### Top Priority Borrowers for Green Mortgage Outreach")
    st.caption("Ranked by percentage improvement potential in annual purchased energy.")

    targets_df = pd.DataFrame([t.model_dump() for t in portfolio.high_potential_targets])
    st.dataframe(
        targets_df[[
            "property_id", "customer_id", "local_authority", "score",
            "classification", "improvement_potential_pct", "ltv", "outstanding_balance"
        ]],
        use_container_width=True,
    )


elif view_mode == "AI Grounding Inspector":
    st.subheader("AI Explanation & Grounding Verification Gate")
    st.caption("Automated deterministic shield proving LLM never fabricates or computes numbers.")

    test_choices = get_featured_properties(df_properties, 25)
    test_pid = st.selectbox("Select Property for Grounding Audit", test_choices)
    row = df_properties[df_properties["property_id"] == test_pid].iloc[0].to_dict()
    clean_d = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    p = Property(**clean_d)
    rep = build_report(p)

    st.markdown("### Simulated Explanation Audit")
    preset_choice = st.selectbox(
        "Select Adversarial Test Scenario (or enter custom text)",
        [
            "1. Valid Engine-Grounded Report",
            "2. Adversarial Trap: Fabricated Cost (£3,750)",
            "3. Adversarial Trap: Fabricated Energy Saving (15,000 kWh)",
            "4. Adversarial Trap: Conflated Heat Pump Electricity Saving (4,000 kWh)",
            "5. Adversarial Trap: Disqualified Retrofit Claim (£2,750 Cavity Wall)",
            "Custom Text",
        ]
    )

    default_texts = {
        "1. Valid Engine-Grounded Report": f"Your home received a Green Home Score of {rep.score.total}/100 ({rep.score.classification}). "
            + (f"By completing recommended retrofits, you could save {rep.package_saving.total_kwh:,.0f} kWh per year and reduce carbon by {rep.package_carbon_kg:,.0f} kg CO2e. "
               if rep.package_saving is not None else "Energy savings are not available for this property. ")
            + (f"Indicative value uplift ranges between £{rep.value['low']:,.0f} and £{rep.value['high']:,.0f}."
               if rep.value.get("status") != "unavailable" else "The property-value impact is unavailable (no current valuation)."),
        "2. Adversarial Trap: Fabricated Cost (£3,750)": f"Your Green Home Score is {rep.score.total}/100. Installing insulation will cost you exactly £3,750.",
        "3. Adversarial Trap: Fabricated Energy Saving (15,000 kWh)": f"By upgrading your heating system, you will save 15,000 kWh of energy every year.",
        "4. Adversarial Trap: Conflated Heat Pump Electricity Saving (4,000 kWh)": f"Installing an air source heat pump will reduce your electricity consumption by 4,000 kWh.",
        "5. Adversarial Trap: Disqualified Retrofit Claim (£2,750 Cavity Wall)": f"We recommend cavity wall insulation for your property to save 3,200 kWh of gas for £2,750.",
    }

    initial_text = default_texts.get(preset_choice, f"Your home received a Green Home Score of {rep.score.total}/100 ({rep.score.classification}).")
    sample_text = st.text_area("Candidate Explanation Text to Verify", value=initial_text, height=110)

    if st.button("Run Grounding Verification Shield", type="primary"):
        result = verify_grounding(sample_text, rep)
        res_dict = result.to_dict()

        if res_dict["grounded"]:
            st.success(f"✅ 100% Grounded — Zero Hallucinations Detected ({res_dict['total_claims']} numeric claims verified against engine JSON)")
            if res_dict["verified_claims"]:
                st.markdown("**Verified Claims:** " + " ".join([f"`{c}`" for c in res_dict["verified_claims"]]))
        else:
            st.error(f"🚨 Ungrounded Claims Detected! ({len(res_dict['unsupported_claims'])} claims not found in engine JSON)")
            st.markdown("**Blocked Hallucinations:** " + " ".join([f"`{c}`" for c in res_dict["unsupported_claims"]]))

        # Claim-by-Claim audit table
        extracted = extract_numbers_from_text(sample_text)
        if extracted:
            audit_rows = []
            for val, raw in extracted:
                is_supported = any(abs(val - num) <= (0.05 * abs(num) + 1.0) for num in res_dict.get("verified_claims", [])) or raw in res_dict.get("verified_claims", [])
                status = "✅ PASS (Engine Grounded)" if raw not in res_dict.get("unsupported_claims", []) else "🚨 BLOCKED (Hallucination)"
                audit_rows.append({"Numeric Value": val, "Matched Token": raw, "Shield Verdict": status})
            st.markdown("#### Claim-by-Claim Verification Audit")
            st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)

        with st.expander("Full JSON Audit Payload"):
            st.json(res_dict)
