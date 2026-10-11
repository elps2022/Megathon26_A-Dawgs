"""Generate Gemini API supervised fine-tuning dataset (JSONL) on verified property queries.

Formats data strictly according to Google Gemini API fine-tuning specifications:
Each line: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "model", ...}]}

Covers:
- Reaching EPC Band C
- Multi-fuel heat pump trade-offs
- Fabric sequencing & insulation
- Infeasible / disqualified measures
- Protecting score / avoiding deterioration
- UK grants (BUS, ECO4, GBIS) & Lloyds green incentives
- Carbon abatement & PCAF Scope 3 reporting
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from greenhome.engine.report import build_report
from greenhome.llm.client import ADVISOR_SYSTEM_PROMPT, format_report_for_prompt
from greenhome.schema import Property

ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = ROOT / "eval" / "gemini_tuning_dataset.jsonl"


def create_training_examples(p: Property, rep) -> list[dict]:
    examples = []
    sys_prompt = (
        f"{ADVISOR_SYSTEM_PROMPT}\n\n"
        f"=== VERIFIED PROPERTY REPORT JSON (SINGLE SOURCE OF TRUTH) ===\n"
        f"{format_report_for_prompt(rep, p)}\n"
        f"==============================================================="
    )

    score = rep.score.total
    cls = rep.score.classification

    # Q1: Score & Overview
    q1 = "What is my current Green Home Score, and how is it calculated?"
    a1 = (
        f"Your property holds an overall Green Home Score of {score}/100 ({cls}). "
        f"The score is calculated deterministically across 7 building physics and operational pillars: "
        f"thermal fabric insulation (walls, roof, glazing), heating efficiency, heating controls, "
        f"consumption intensity against similar homes, and solar generation. "
        f"Energy savings quoted for your home are model estimates, not guarantees."
    )
    examples.append({
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q1},
            {"role": "model", "content": a1},
        ]
    })

    # Q2: Reaching Band C / Top upgrades
    if rep.recommendations:
        top_rec = rep.recommendations[0]
        cost_str = f"£{top_rec.indicative_cost[0]:,.0f} – £{top_rec.indicative_cost[1]:,.0f}"
        q2 = "What is the single best retrofit measure I should prioritise?"
        a2 = (
            f"Based on your report, the highest-priority measure is {top_rec.retrofit.replace('_', ' ').title()}. "
            f"Indicative capital cost is {cost_str} ({top_rec.reason}). "
            f"Fabric insulation is prioritised before mechanical systems to minimise peak heat loss."
        )
        examples.append({
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q2},
                {"role": "model", "content": a2},
            ]
        })

    # Q3: Heat pump fuel segregation
    hp_rec = next((r for r in rep.recommendations if r.retrofit == "heat_pump"), None)
    if hp_rec:
        q3 = "If I install a heat pump, will my electricity bills decrease?"
        a3 = (
            f"No, your electricity bills will not decrease. Installing a heat pump replaces your current heating, "
            f"eliminating gas usage. However, because the heat pump runs on electricity, your electricity consumption "
            f"will increase. Overall carbon footprint decreases significantly (by an estimated {hp_rec.carbon_kg:,.0f} kg CO2e/yr), "
            f"but electricity unit rates are higher than gas. Indicative installation cost is £{hp_rec.indicative_cost[0]:,.0f} – £{hp_rec.indicative_cost[1]:,.0f}."
        )
        examples.append({
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q3},
                {"role": "model", "content": a3},
            ]
        })

    # Q4: Infeasible retrofits
    if rep.infeasible:
        inf = rep.infeasible[0]
        name = inf.retrofit.replace("_", " ").title()
        q4 = f"Can I install {name} on my home?"
        a4 = (
            f"{name} is assessed as infeasible for your property. "
            f"According to your property characteristics, this measure is physically disqualified because: {inf.rule}. "
            f"The engine does not recommend measures that compromise structural integrity or ventilation."
        )
        examples.append({
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q4},
                {"role": "model", "content": a4},
            ]
        })

    # Q5: What NOT to do / score protection
    q5 = "What should I avoid doing so that my Green Home Score doesn't deteriorate?"
    a5 = (
        f"To protect your Green Home Score of {score}/100 and prevent performance deterioration:\n"
        f"• Avoid replacing efficient condensing heating with direct electric resistance heaters or uncertified systems.\n"
        f"• Do not compress or remove loft insulation during renovations, and keep window seals maintained.\n"
        f"• Maintain smart thermostat schedules and TRVs rather than running continuous unzoned heating.\n"
        f"• Ensure annual boiler or heat pump servicing to maintain rated seasonal efficiency."
    )
    examples.append({
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q5},
            {"role": "model", "content": a5},
        ]
    })

    # Q6: Grants & Lloyds Finance
    q6 = "What grants or financing options are available to help me fund these upgrades?"
    a6 = (
        f"For qualifying homes with a score of {score}/100 ({cls}), several funding streams apply:\n"
        f"• Boiler Upgrade Scheme (BUS): Up to £7,500 non-repayable government grant for air-source heat pumps.\n"
        f"• Great British Insulation Scheme (GBIS) & ECO4: Subsidised or free fabric insulation for eligible homes.\n"
        f"• 0% VAT on Energy Saving Materials: Government zero-rate VAT through 2027.\n"
        f"• Lloyds Bank Green Incentives: Preferential green additional borrowing and Eco Home Reward cashback for eligible retrofits."
    )
    examples.append({
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q6},
            {"role": "model", "content": a6},
        ]
    })

    return examples


def main() -> None:
    # Use real properties if available, fallback to synthetic
    real_path = ROOT / "data" / "processed" / "real" / "properties_share.parquet"
    synth_path = ROOT / "data" / "generated" / "properties.csv"

    if real_path.exists():
        df = pd.read_parquet(real_path).head(15)
    elif synth_path.exists():
        df = pd.read_csv(synth_path).head(15)
    else:
        raise RuntimeError("No properties dataset found. Run make data first.")

    all_examples = []
    for _, row in df.iterrows():
        d = row.to_dict()
        clean_d = {k: (None if pd.isna(v) else v) for k, v in d.items()}
        flags_val = clean_d.get("data_flags")
        clean_d["data_flags"] = [f.strip() for f in str(flags_val).split(",") if f.strip() and f != "nan"] if flags_val else []
        p = Property(**clean_d)
        rep = build_report(p)
        examples = create_training_examples(p, rep)
        all_examples.extend(examples)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Generated {len(all_examples)} Gemini fine-tuning examples -> {OUT_FILE}")


if __name__ == "__main__":
    main()
