"""Live LLM Advisor Client with Deterministic Grounding Shield.

Supports pluggable providers (Anthropic Claude, Google Gemini, OpenAI) with:
1. Zero-arithmetic system prompt framing.
2. Compact PropertyReport JSON grounding payload.
3. Deterministic Grounding Shield verification gate.
4. 1-shot corrective retry if unverified numbers are produced.
5. Safe deterministic rule fallback if offline, ungrounded, or without API keys.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
import httpx

from greenhome.llm.grounding import GroundingResult, verify_grounding
from greenhome.llm.guardrails import evaluate_guardrails
from greenhome.schema import Property, PropertyReport


ADVISOR_SYSTEM_PROMPT = """You are the Lloyds Bank Green Home Energy Advisor. Your role is to explain property energy reports, recommendations, and green mortgage incentives clearly, objectively, and responsibly.

MANDATORY RULES:
1. NUMBERS COME FROM PROPERTY REPORT ONLY: Never compute, guess, or extrapolate numbers. Every score, saving (kWh), carbon reduction (kg/tonnes), cost (£), or value uplift (£/%) MUST be an exact number present in the provided PropertyReport JSON.
2. HONEST UNCERTAINTY:
   - Always state that energy savings are estimates, not guarantees.
   - Always state that property value impact is an indicative market estimate, not a guaranteed causal gain.
   - If a field is missing or unavailable, say "unavailable" and advise consulting a qualified surveyor. NEVER invent 0.
3. PER-FUEL ACCOUNTING:
   - Always distinguish electricity from gas savings.
   - Note that heat pumps eliminate gas heating but increase electricity consumption.
   - Solar PV offsets imported electricity; it does not reduce space heating load.
4. INFEASIBILITY CITATIONS:
   - When asked about retrofits marked infeasible, cite the exact physical barrier from the report.
5. TONE & REGULATION:
   - Professional, supportive, compliant with FCA Consumer Duty. Highlight grant schemes (BUS, ECO4) and Lloyds preferential green financing qualitatively unless specific amounts are in the report.

FEW-SHOT DEMONSTRATIONS (EXEMPLARS OF CORRECT GROUNDED BEHAVIOUR):

[Example 1: Reaching EPC Band C & Sequencing]
User: "How do I upgrade my home to reach EPC Band C?"
Assistant: "To upgrade towards EPC Band C standards, the report prioritises envelope insulation before or alongside heating system changes. For your home, the top recommendations are:
• Cavity Wall Insulation: indicative cost £1,000 – £2,500, targeting 3,200 kWh gas saving.
• Smart Thermostat: indicative cost £150 – £400, preventing room overheating through weather compensation.
Implementing the recommended package yields estimated annual savings of 4,100 kWh across fuels. Please note these are modelled estimates; an on-site retrofit coordinator assessment is advised before commencing works."

[Example 2: Heat Pump Trade-off & Fuel Segregation]
User: "Will installing a heat pump cut my electricity bills?"
Assistant: "A heat pump replaces your gas boiler, which eliminates gas heating use. However, because heat pumps run on electricity, your home's electricity consumption will INCREASE. Overall operational carbon drops substantially (cutting an estimated 1,815 kg CO2e annually in your report), but because electricity unit rates are typically ~3.5x higher than gas, electricity bills rise while gas bills disappear. Government Boiler Upgrade Scheme (BUS) provides a £7,500 grant to help offset installation capital costs."

[Example 3: Disqualified / Infeasible Interventions]
User: "Can I install cavity wall insulation on my home?"
Assistant: "Cavity wall insulation is assessed as infeasible for your property. According to the report records, your property has solid brick walls without an unfilled cavity. Blowing insulation into a solid wall is physically impossible and would cause severe moisture issues. Instead, solid wall insulation (internal or external) would be the suitable fabric alternative."

[Example 4: Protecting Score / What NOT to do]
User: "What should I avoid doing so my Green Home Score doesn't drop?"
Assistant: "To protect your current Green Home Score and avoid performance deterioration:
• Do not replace your heating system with direct electric resistance heaters (such as plug-in panels), which severely penalise efficiency scores.
• Avoid compromising your thermal envelope: ensure loft insulation is not compressed or removed during storage use, and maintain double-glazing window seals.
• Retain programmable heating controls and TRVs rather than reverting to manual continuous heating.
• Ensure regular annual heating servicing to preserve seasonal COP and boiler efficiency."

[Example 5: Grants and Lloyds Green Banking]
User: "What financial support or grants can I access for these upgrades?"
Assistant: "Depending on eligibility, you may qualify for key UK support schemes:
• Boiler Upgrade Scheme (BUS): Up to £7,500 non-repayable grant for air-source heat pumps.
• Great British Insulation Scheme (GBIS) & ECO4: Free or subsidised cavity and loft insulation for qualifying homes in lower EPC bands.
• 0% VAT: UK government zero-rate VAT on energy-saving materials installations through 2027.
• Lloyds Bank Green Incentives: Lloyds offers preferential green additional borrowing and Eco Home Reward cashback for customers installing eligible energy efficiency measures."
"""


def _load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (no extra dependency). Real env vars take precedence."""
    env_path = path or Path(__file__).resolve().parents[3] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def format_report_for_prompt(report: PropertyReport, p: Property | None = None) -> str:
    """Extract a compact, token-efficient JSON string for LLM prompting."""
    rep_dict = report.model_dump()
    summary = {
        "property_id": rep_dict.get("property_id"),
        "green_home_score": rep_dict.get("score", {}).get("total"),
        "classification": rep_dict.get("score", {}).get("classification"),
        "components": rep_dict.get("score", {}).get("components", {}),
        "package_saving_kwh": rep_dict.get("package_saving", {}).get("total_kwh") if rep_dict.get("package_saving") else 0.0,
        "package_gas_kwh": rep_dict.get("package_saving", {}).get("gas_kwh") if rep_dict.get("package_saving") else 0.0,
        "package_elec_kwh": rep_dict.get("package_saving", {}).get("elec_kwh") if rep_dict.get("package_saving") else 0.0,
        "package_carbon_cut_kg": rep_dict.get("package_carbon_kg", 0.0),
        "property_value_impact": rep_dict.get("value", {}),
        "recommendations": [
            {
                "retrofit": r.get("retrofit"),
                "suitability": r.get("suitability"),
                "reason": r.get("reason"),
                "elec_saving_kwh": r.get("marginal_saving", {}).get("elec_kwh", 0.0),
                "gas_saving_kwh": r.get("marginal_saving", {}).get("gas_kwh", 0.0),
                "carbon_kg": r.get("carbon_kg", 0.0),
                "indicative_cost": r.get("indicative_cost", [0.0, 0.0]),
            }
            for r in rep_dict.get("recommendations", [])
        ],
        "infeasible_interventions": [
            {"retrofit": inf.get("retrofit"), "rule": inf.get("rule")}
            for inf in rep_dict.get("infeasible", [])
        ],
    }
    return json.dumps(summary, indent=2)


ADVISOR_TOOLS = [
    {
        "name": "get_property_report",
        "description": "Retrieve current verified Green Home Score and recommendations for a property.",
        "parameters": {
            "type": "object",
            "properties": {
                "property_id": {"type": "string", "description": "Unique property identifier"}
            },
            "required": ["property_id"]
        }
    },
    {
        "name": "run_scenario",
        "description": "Rerun calculation engine for a proposed what-if package of retrofits, recalculating Green Home Score, per-fuel savings, carbon cut, and value uplift.",
        "parameters": {
            "type": "object",
            "properties": {
                "property_id": {"type": "string", "description": "Unique property identifier"},
                "retrofits": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of retrofit measures to simulate (e.g. ['solar_pv', 'heat_pump'])"
                }
            },
            "required": ["retrofits"]
        }
    },
    {
        "name": "get_customer_memory",
        "description": "Retrieve customer memory profile, constraints, completed retrofits, and interaction history.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "Customer identifier"}
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "update_customer_memory",
        "description": "Record stated customer constraints or completed/declined retrofit decisions in persistent memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "Customer identifier"},
                "action": {"type": "string", "enum": ["record_constraint", "record_declined", "record_completed"]},
                "detail": {"type": "string", "description": "Constraint or retrofit detail"}
            },
            "required": ["customer_id", "action", "detail"]
        }
    }
]

RETROFIT_KEYWORDS = {
    "solar_pv": ["solar", "pv", "solar panel", "solar panels"],
    "heat_pump": ["heat pump", "ashp", "gshp", "heatpump"],
    "loft_insulation": ["loft", "roof insulation", "attic insulation", "loft insulation"],
    "cavity_wall_insulation": ["cavity", "cavity wall", "cavity insulation"],
    "solid_wall_insulation": ["solid wall", "external wall insulation", "internal wall insulation"],
    "double_glazing": ["double glazing", "glazing", "windows", "new windows"],
    "smart_thermostat": ["smart thermostat", "thermostat", "heating controls", "smart controls"],
}


def detect_what_if_retrofits(query: str) -> list[str] | None:
    """Detect if query is proposing a what-if retrofit scenario (Architecture Spec 3.10)."""
    q = query.lower()
    is_scenario = any(phrase in q for phrase in [
        "what if", "what happens if", "what would happen if", "if i add", "if i install",
        "if i put", "if i do", "if we add", "if we install", "if we do", "simulate",
        "scenario", "suppose i", "what if we", "how about adding", "how about installing"
    ])
    if not is_scenario:
        return None
    found = []
    for ret_name, keywords in RETROFIT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            found.append(ret_name)
    return found if found else None


def detect_memory_action(query: str) -> dict[str, str] | None:
    """Detect natural language expressions of memory updates (completed works, constraints, preferences)."""
    q = query.lower()

    # 1. Completed work mentions:
    if any(phrase in q for phrase in [
        "already installed", "already done", "already completed", "have installed",
        "got installed", "i fitted", "we fitted", "completed installation", "finished installing"
    ]):
        for ret, kws in RETROFIT_KEYWORDS.items():
            if any(kw in q for kw in kws):
                return {
                    "action": "record_completed",
                    "detail": ret,
                    "label": f"Recorded completed {ret.replace('_', ' ').title()}",
                }

    # 2. Decline / rejection mentions:
    if any(phrase in q for phrase in [
        "don't want", "dont want", "do not want", "decline", "not interested in", "skip", "reject"
    ]):
        for ret, kws in RETROFIT_KEYWORDS.items():
            if any(kw in q for kw in kws):
                return {
                    "action": "record_declined",
                    "detail": ret,
                    "label": f"Recorded preference: declined {ret.replace('_', ' ').title()}",
                }

    # 3. Tenure constraints:
    if any(phrase in q for phrase in ["i'm renting", "i am renting", "i rent", "i'm a tenant", "i am a tenant", "tenancy agreement"]):
        return {
            "action": "record_constraint",
            "detail": "tenant",
            "label": "Recorded housing tenure: Tenant (structural works restricted)",
        }

    # 4. Budget constraints:
    m = re.search(r"(?:budget|spend|afford|max(?:imum)?)\s*(?:is|of|capped at)?\s*[£$]?\s*(\d{1,2}(?:,\d{3})+|\d{3,6})", q)
    if m:
        val_str = m.group(1).replace(",", "")
        return {
            "action": "record_constraint",
            "detail": f"budget_{val_str}",
            "label": f"Recorded budget limit: £{float(val_str):,.0f}",
        }

    return None


def execute_advisor_tool(
    tool_name: str,
    args: dict[str, Any],
    report: PropertyReport,
    p: Property | None = None,
) -> dict[str, Any]:
    """Execute tool calls deterministically via engine and memory layers."""
    prop = p or getattr(report, "property", None)
    if tool_name == "get_property_report":
        return json.loads(format_report_for_prompt(report, prop))
    elif tool_name == "run_scenario":
        from greenhome.engine.report import build_report
        retrofits = args.get("retrofits", [])
        if prop is None:
            return {"error": "Property record required for simulation"}
        scen_report = build_report(prop, retrofits=retrofits)
        return json.loads(format_report_for_prompt(scen_report, prop))
    elif tool_name == "get_customer_memory":
        from greenhome.memory.store import CustomerMemoryStore
        cust_id = args.get("customer_id") or (prop.customer_id if prop else "DEFAULT")
        return CustomerMemoryStore().get_memory(cust_id)
    elif tool_name == "update_customer_memory":
        from greenhome.memory.store import CustomerMemoryStore
        cust_id = args.get("customer_id") or (prop.customer_id if prop else "DEFAULT")
        store = CustomerMemoryStore()
        action = args.get("action")
        detail = args.get("detail", "")
        if action == "record_constraint":
            return store.record_constraint(cust_id, detail)
        elif action == "record_declined":
            return store.record_declined(cust_id, detail)
        elif action == "record_completed":
            return store.record_completed(cust_id, detail)
        return {"status": "unknown_action"}
    return {"error": f"Unknown tool: {tool_name}"}


def generate_rule_fallback(
    query: str,
    report: PropertyReport,
    p: Property | None = None,
    what_if_retrofits: list[str] | None = None,
    baseline_report: PropertyReport | None = None,
) -> str:
    """Deterministic, guaranteed-grounded fallback response covering all key user query intents."""
    # 0. Safety and Domain Guardrail Gate
    g_res = evaluate_guardrails(query)
    if not g_res.passed:
        return g_res.message or "I can only assist with home energy and decarbonisation queries."

    q = query.lower().strip()
    score = report.score.total
    cls = report.score.classification

    # 0b. What-if Scenario Engine Response (Architecture Spec 3.10)
    if what_if_retrofits and baseline_report is not None:
        ret_names = ", ".join(r.replace("_", " ").title() for r in what_if_retrofits)
        base_score = baseline_report.score.total
        new_score = report.score.total
        diff = new_score - base_score

        score_line = (
            f"Simulating {ret_names} updates your Green Home Score from {base_score}/100 to {new_score}/100 ({cls})."
        )
        save_line = ""
        if report.package_saving:
            save_line = (
                f"\n• Electricity impact: {report.package_saving.elec_kwh:,.0f} kWh/yr"
                f"\n• Gas saving: {report.package_saving.gas_kwh:,.0f} kWh/yr"
                f"\n• Net carbon reduction: {report.package_carbon_kg:,.0f} kg CO2e"
            )

        cost_parts = []
        for r in report.recommendations:
            if r.retrofit in what_if_retrofits:
                cost_parts.append(f"£{r.indicative_cost[0]:,.0f} – £{r.indicative_cost[1]:,.0f}")
        cost_line = f"\n• Indicative cost: {', '.join(cost_parts)}" if cost_parts else ""

        val_line = ""
        if report.value.get("status") != "unavailable" and "low" in report.value and "high" in report.value:
            val_line = f"\n• Property value impact: £{report.value['low']:,.0f} – £{report.value['high']:,.0f}."

        infeas = [inf for inf in report.infeasible if inf.retrofit in what_if_retrofits]
        infeas_line = ""
        if infeas:
            infeas_line = "\n" + "\n".join(f"⚠️ {inf.retrofit.replace('_', ' ').title()} is infeasible: {inf.rule}" for inf in infeas)

        return (
            f"{score_line}{save_line}{cost_line}{val_line}{infeas_line}\n\n"
            f"Note: Modelled scenario estimates; on-site surveyor assessment advised."
        )

    # 1. Greetings & introductory queries
    if q in {"hello", "hi", "hey", "good morning", "good afternoon", "good evening"} or "who are you" in q or "what can you do" in q:
        top_recs = [r.retrofit.replace("_", " ").title() for r in report.recommendations[:2]]
        recs_str = " and ".join(top_recs) if top_recs else "efficiency retrofits"
        return (
            f"Hello! I am your Lloyds Bank Green Home Energy Advisor. "
            f"Your property has an overall Green Home Score of {score}/100 ({cls}). "
            f"Key high-impact upgrades identified for your home are {recs_str}. "
            f"You can ask me how to improve your score, how to avoid score deterioration, what grants apply, or about specific heating and insulation measures."
        )

    # 2. What NOT to do / Avoid score deterioration / Protection
    elif any(term in q for term in ["not to do", "dont do", "don't do", "deteriorat", "worse", "decrease", "drop", "protect", "harm", "penal"]):
        lines = [
            f"To protect your current Green Home Score of {score}/100 ({cls}) and avoid performance deterioration, keep in mind:",
            "• Do not replace efficient condensing heating with direct electric resistance heaters or uncertified systems.",
            "• Avoid degrading thermal fabric: do not compress or remove loft insulation during renovations, and maintain window seals.",
            "• Do not remove heating controls: operating without a programmable thermostat or TRVs significantly penalises consumption intensity.",
            "• Ensure heating equipment is serviced annually to maintain rated seasonal efficiency."
        ]
        return "\n".join(lines)

    # 3. How to improve score / Recommendations / Action plan
    elif any(term in q for term in ["improve", "better", "increase", "boost", "what can i do", "what should i do", "action", "recommend", "how to"]):
        if report.recommendations:
            rec_lines = []
            for r in report.recommendations[:3]:
                name = r.retrofit.replace("_", " ").title()
                cost_str = f"£{r.indicative_cost[0]:,.0f} – £{r.indicative_cost[1]:,.0f}"
                rec_lines.append(f"• {name}: indicative cost {cost_str} (reason: {r.reason}).")
            recs_text = "\n".join(rec_lines)
            save_text = ""
            if report.package_saving:
                save_text = f"\nImplementing the full package targets estimated savings of {report.package_saving.total_kwh:,.0f} kWh/yr with an annual carbon cut of {report.package_carbon_kg:,.0f} kg CO2e."
            return (
                f"To improve your Green Home Score from {score}/100 ({cls}), the engine recommends these sequenced measures:\n"
                f"{recs_text}{save_text}"
            )
        else:
            return f"Your home already holds an optimal Green Home Score of {score}/100 ({cls}). No major fabric or heating interventions are currently pending."

    # 4. Grants, subsidies, and Lloyds financing
    elif any(term in q for term in ["grant", "bus", "subsidy", "eco4", "gbis", "cashback", "finance", "borrow", "lloyds", "loan"]):
        has_hp = any(r.retrofit == "heat_pump" for r in report.recommendations)
        hp_note = " For heat pumps, the Boiler Upgrade Scheme (BUS) offers grant funding toward capital installation costs." if has_hp else ""
        return (
            f"Based on your score of {score}/100 ({cls}), qualifying homes may access government schemes including GBIS (insulation) or ECO4.{hp_note} "
            f"Lloyds Bank also offers preferential green additional borrowing and Eco Home Reward cashback for eligible energy upgrades."
        )

    # 5. Heating systems & heat pump specific questions
    elif any(term in q for term in ["heat pump", "boiler", "gas heating", "heating", "radiator", "fuel switch"]):
        hp_rec = next((r for r in report.recommendations if r.retrofit == "heat_pump"), None)
        if hp_rec:
            return (
                f"Heat pump installation is recommended for your home ({hp_rec.reason}). "
                f"Indicative cost is £{hp_rec.indicative_cost[0]:,.0f} – £{hp_rec.indicative_cost[1]:,.0f}. "
                f"Note: switching from gas heating to a heat pump eliminates space gas use, but electricity consumption will increase accordingly."
            )
        else:
            inf = next((inf for inf in report.infeasible if inf.retrofit == "heat_pump"), None)
            if inf:
                return f"A heat pump is currently assessed as infeasible for this property because: {inf.rule}."
            return f"Your home currently has a heating score evaluated within the overall {score}/100 score."

    # 6. Insulation & fabric (loft, walls, glazing)
    elif any(term in q for term in ["insulat", "loft", "cavity", "wall", "glaz", "window", "roof", "draught"]):
        fabric_recs = [r for r in report.recommendations if any(w in r.retrofit for w in ["loft", "wall", "glaz"])]
        if fabric_recs:
            lines = [f"Recommended fabric upgrades for your {score}/100 home:"]
            for r in fabric_recs:
                lines.append(f"• {r.retrofit.replace('_', ' ').title()}: £{r.indicative_cost[0]:,.0f} – £{r.indicative_cost[1]:,.0f} ({r.reason})")
            return "\n".join(lines)
        return f"Thermal fabric assessment indicates existing insulation standards are consistent with your current score of {score}/100 ({cls})."

    # 7. Solar PV questions
    elif any(term in q for term in ["solar", "pv", "panel"]):
        solar_rec = next((r for r in report.recommendations if r.retrofit == "solar_pv"), None)
        if solar_rec:
            return (
                f"Solar PV is recommended ({solar_rec.reason}) with an indicative cost of "
                f"£{solar_rec.indicative_cost[0]:,.0f} – £{solar_rec.indicative_cost[1]:,.0f}. "
                f"Solar offsets grid electricity imports; note that it does not directly reduce space heating demand."
            )
        inf = next((inf for inf in report.infeasible if inf.retrofit == "solar_pv"), None)
        if inf:
            return f"Solar PV is marked infeasible for this property because: {inf.rule}."
        return f"Solar PV feasibility has been evaluated for your {score}/100 home."

    # 8. Bills, savings, and value impact
    elif any(term in q for term in ["saving", "bill", "cost", "money", "payback", "value"]):
        if report.package_saving:
            val_text = ""
            if report.value.get("status") != "unavailable" and "low" in report.value and "high" in report.value:
                val_text = f" Indicative property value impact is £{report.value['low']:,.0f} – £{report.value['high']:,.0f}."
            return (
                f"Implementing the recommended retrofit package could yield estimated fuel savings of "
                f"{report.package_saving.total_kwh:,.0f} kWh per year ({report.package_saving.gas_kwh:,.0f} kWh gas).{val_text}"
            )
        return f"Your current Green Home Score is {score}/100 ({cls}). Detailed energy savings require recorded billing meters."

    # 9. Carbon & environmental impact
    elif any(term in q for term in ["carbon", "emission", "co2", "climate"]):
        if report.package_carbon_kg:
            return (
                f"Your recommended retrofit package can eliminate approximately {report.package_carbon_kg:,.0f} kg CO2e per year. "
                f"This helps reduce your home's operational carbon footprint significantly."
            )
        return f"Carbon calculations for score {score}/100 ({cls}) require regional emission factors."

    # 10. Default contextual overview
    top = [r.retrofit.replace("_", " ").title() for r in report.recommendations[:3]]
    top_str = ", ".join(top) if top else "efficiency measures"
    return (
        f"Your home holds a Green Home Score of {score}/100 ({cls}). "
        f"Key evaluated retrofits include: {top_str}. "
        f"You can ask about fuel bill savings, grants, heat pumps, insulation, or protecting your current score."
    )


class LLMAdvisor:
    """Manages LLM API calls with deterministic grounding shield and 1-shot retry."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
    ):
        _load_dotenv()
        self.provider = provider or self._detect_provider()
        self.api_key = api_key or self._detect_api_key(self.provider)
        self.model = model or self._default_model(self.provider)
        self.timeout = timeout_seconds

    def _detect_provider(self) -> str:
        # Explicit override, e.g. LLM_PROVIDER=offline for demos without network.
        forced = os.environ.get("LLM_PROVIDER", "").strip().lower()
        if forced in {"gemini", "anthropic", "openai", "offline"}:
            return forced
        # Gemini free-tier key is the team default.
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            return "gemini"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        # Always prefer gemini over paid anthropic
        return "gemini"

    def _detect_api_key(self, provider: str) -> str | None:
        if provider == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY")
        elif provider == "gemini":
            return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        elif provider == "openai":
            return os.environ.get("OPENAI_API_KEY")
        return None

    def _default_model(self, provider: str) -> str:
        if provider == "anthropic":
            return os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
        elif provider == "gemini":
            return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        elif provider == "openai":
            return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return "deterministic-rule-engine"

    def _call_anthropic(self, messages: list[dict[str, str]], system_prompt: str) -> str:
        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 800,
            "temperature": 0.2,
            "system": system_prompt,
            "messages": messages,
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"].strip()

    def _call_gemini(self, messages: list[dict[str, str]], system_prompt: str) -> str:
        # Key goes in a header, not the query string, so it never appears in error messages/logs.
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key or "", "content-type": "application/json"}
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            # Generous cap: on thinking models, reasoning tokens count toward the output budget.
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            parts = data["candidates"][0]["content"].get("parts", [])
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            if not text.strip():
                raise RuntimeError("Gemini returned no text")
            return text.strip()

    def _call_openai(self, messages: list[dict[str, str]], system_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 800,
            "messages": full_messages,
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    def _call_llm(self, messages: list[dict[str, str]], system_prompt: str) -> str:
        if self.provider == "anthropic":
            return self._call_anthropic(messages, system_prompt)
        elif self.provider == "gemini":
            return self._call_gemini(messages, system_prompt)
        elif self.provider == "openai":
            return self._call_openai(messages, system_prompt)
        raise RuntimeError(f"Unknown or offline provider: {self.provider}")

    def query(
        self,
        query: str,
        report: PropertyReport,
        p: Property | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> tuple[str, GroundingResult, str]:
        """Generate response with grounding shield and 1-shot corrective retry.

        Returns:
            (response_text, GroundingResult, model_label)
        """
        # 1. Enforce Safety & Domain Guardrails prior to any LLM execution
        has_hist = bool(history and len(history) > 0)
        g_res = evaluate_guardrails(query, has_history=has_hist)
        if not g_res.passed:
            msg = g_res.message or "I can only assist with property decarbonisation queries."
            audit = verify_grounding(msg, report)
            if g_res.action == "BLOCK_SAFETY":
                label = "Safety Guardrail (Blocked: Unacceptable Content)"
            elif g_res.action == "REFOCUS_GIBBERISH":
                label = "Input Guardrail (Gibberish Refocused)"
            else:
                label = "Domain Guardrail (Off-Topic Refocused)"
            return msg, audit, label

        # 2. Check for natural language memory updates (Architecture Spec 3.11 & Section 10)
        target_property = p or getattr(report, "property", None)
        active_report = report
        scenario_label_suffix = ""
        memory_badge = ""

        mem_action = detect_memory_action(query)
        if mem_action and target_property is not None and target_property.customer_id:
            execute_advisor_tool("update_customer_memory", {"action": mem_action["action"], "detail": mem_action["detail"]}, report, target_property)
            from greenhome.memory.store import CustomerMemoryStore
            from greenhome.engine.report import build_report
            updated_mem = CustomerMemoryStore().get_memory(target_property.customer_id)
            updated_p = CustomerMemoryStore().apply_memory_to_property(target_property)
            active_report = build_report(updated_p, memory=updated_mem)
            target_property = updated_p
            memory_badge = f"[🧠 Memory Updated: {mem_action['label']}]\n\n"
            scenario_label_suffix = " (Memory Updated)"

        # 3. Check for What-If scenario (Architecture Spec 3.10 & CLAUDE.md invariant)
        what_if_retrofits = detect_what_if_retrofits(query)
        if what_if_retrofits and target_property is not None:
            from greenhome.engine.report import build_report
            active_report = build_report(target_property, retrofits=what_if_retrofits)
            scenario_label_suffix = f" (Scenario Engine: {', '.join(what_if_retrofits)})"

        if self.provider == "offline" or not self.api_key:
            fallback = generate_rule_fallback(
                query, active_report, target_property,
                what_if_retrofits=what_if_retrofits,
                baseline_report=report if what_if_retrofits else None,
            )
            full_fallback = f"{memory_badge}{fallback}" if memory_badge else fallback
            audit = verify_grounding(full_fallback, active_report)
            return full_fallback, audit, f"Deterministic Rule Engine{scenario_label_suffix or ' (Offline)'}"

        report_json_str = format_report_for_prompt(active_report, target_property)
        scenario_note = (
            f"\nNOTE: User proposed what-if scenario [{', '.join(what_if_retrofits)}]. "
            f"The calculation engine reran build_report() and produced the updated PropertyReport JSON below. "
            f"Ground your answer strictly in these updated figures."
            if what_if_retrofits else ""
        )
        system_with_context = (
            f"{ADVISOR_SYSTEM_PROMPT}\n{scenario_note}\n"
            f"=== VERIFIED PROPERTY REPORT JSON (SINGLE SOURCE OF TRUTH) ===\n"
            f"{report_json_str}\n"
            f"===============================================================\n"
        )

        # Build message history. Anthropic/Gemini require the first message to be from the
        # user and roles to alternate, so drop leading assistant turns and merge repeats.
        messages: list[dict[str, str]] = []
        for item in (history or [])[-6:]:
            role = "assistant" if item["role"] == "assistant" else "user"
            if not messages and role == "assistant":
                continue
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += "\n\n" + item["content"]
            else:
                messages.append({"role": role, "content": item["content"]})
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += "\n\n" + query
        else:
            messages.append({"role": "user", "content": query})

        # Attempt 1: Initial Generation
        try:
            raw_response = self._call_llm(messages, system_with_context)
            audit = verify_grounding(raw_response, active_report)

            if audit.grounded:
                model_label = f"{self.provider.title()} ({self.model}){scenario_label_suffix}"
                return raw_response, audit, model_label

            # Attempt 2: 1-shot corrective retry
            unsupported_str = ", ".join(audit.unsupported_claims)
            corrective_user_message = (
                f"GROUNDING SHIELD CORRECTION:\n"
                f"Your previous response cited numbers [{unsupported_str}] that do NOT exist in the verified PropertyReport JSON.\n"
                f"Please re-write your explanation. Use ONLY exact numbers from the PropertyReport JSON or speak qualitatively."
            )
            retry_messages = messages + [
                {"role": "assistant", "content": raw_response},
                {"role": "user", "content": corrective_user_message},
            ]

            retry_response = self._call_llm(retry_messages, system_with_context)
            retry_audit = verify_grounding(retry_response, active_report)

            if retry_audit.grounded:
                model_label = f"{self.provider.title()} ({self.model} - Shield Corrected){scenario_label_suffix}"
                return retry_response, retry_audit, model_label

            # Still ungrounded after retry: Fall back safely to deterministic template
            fallback = generate_rule_fallback(
                query, active_report, target_property,
                what_if_retrofits=what_if_retrofits,
                baseline_report=report if what_if_retrofits else None,
            )
            fb_audit = verify_grounding(fallback, active_report)
            shield_notice = "[Shield Notice: Unverified figures were blocked by Grounding Shield]\n\n"
            return (
                f"{shield_notice}{fallback}",
                fb_audit,
                f"Deterministic Rule Engine (Shield Intervened){scenario_label_suffix}",
            )

        except Exception as exc:
            # On any network/API failure, cleanly fall back to deterministic response
            reason = type(exc).__name__
            if isinstance(exc, httpx.HTTPStatusError):
                code = exc.response.status_code
                reason = "rate limited (429)" if code == 429 else f"HTTP {code}"
            fallback = generate_rule_fallback(
                query, active_report, target_property,
                what_if_retrofits=what_if_retrofits,
                baseline_report=report if what_if_retrofits else None,
            )
            fb_audit = verify_grounding(fallback, active_report)
            return fallback, fb_audit, f"Deterministic Rule Engine (Fallback: {reason}){scenario_label_suffix}"
