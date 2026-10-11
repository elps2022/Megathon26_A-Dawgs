"""Generate deliverables/M2.pdf (Megathon X, PS2: mid-evaluation 2, max 2 pages).

Sections follow the deliverables spec: current approach and changes since M1, experiments and results, testing
progress, and the plan for the remaining time. Double-blind: team number only. Reuses the M1 layout helpers.
Every number comes from eval/results/*.json and docs/EVALUATION.md (validation split; test split untouched).

    uv run --no-project --with reportlab python scripts/generate_m2_pdf.py
"""
import argparse

from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer

import generate_m1_pdf as m1
from generate_m1_pdf import A4, ACCENT, GREEN_BG, MARGIN, PAGE_HEIGHT, PAGE_WIDTH, PRIMARY, TEXT_MUTED, BORDER

STAGE = "M2: MID-EVALUATION"


class M2Canvas(m1.NumberedCanvas):
    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(PRIMARY)
        self.drawString(MARGIN, PAGE_HEIGHT - 20, f"MEGATHON X : SINGULARITY  |  PROBLEM STATEMENT 2  |  {STAGE}")
        self.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 20, self.team_header_str)
        self.setStrokeColor(BORDER)
        self.setLineWidth(0.75)
        self.line(MARGIN, PAGE_HEIGHT - 24, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 24)
        self.line(MARGIN, 22, PAGE_WIDTH - MARGIN, 22)
        self.setFont("Helvetica", 7.5)
        self.setFillColor(TEXT_MUTED)
        self.drawString(MARGIN, 13, "Green Home Score & Retrofit Recommendation Engine")
        self.drawRightString(PAGE_WIDTH - MARGIN, 13, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


def build_pdf(output_path: str, team_number: str = "73"):
    M2Canvas.team_header_str = f"TEAM NUMBER: {team_number}"
    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=32, bottomMargin=30)
    st = m1.styles()
    b = lambda text: Paragraph(text, st["bullet"], bulletText="•")
    w = m1.CONTENT_W
    story = [
        Paragraph("Green Home Score & Retrofit Recommendation Engine", st["title"]),
        Paragraph("Mid-evaluation 2: real-data pipeline built end to end, energy models trained and honestly scored "
                  "on held-out metered consumption", st["subtitle"]),
    ]

    # 1. Current approach and changes since M1
    story += [
        Paragraph("1. Current Approach and Changes Since M1", st["h1"]),
        Paragraph("The architecture from M1 stands: a deterministic engine produces one PropertyReport per home, and "
                  "the LLM only explains it. What is now built on <b>real public data</b>:", st["body"]),
        b("<b>Property book:</b> 542,711 homes from energy performance certificates in three contrasting local "
          "authorities, free-text fields mapped to model inputs (99.3-100% coverage), deprivation index joined by postcode."),
        b("<b>Sales and values:</b> 175,457 house sales matched to homes (92.2% match rate; property type agrees for 93%), "
          "using only the certificate lodged <i>before</i> each sale. Current value for 153,511 homes = last sale x a "
          "price index for that property type and area."),
        b("<b>Observed retrofit labels:</b> 4,791 real loft, cavity-wall and solar installs with metered use 2 years "
          "before vs after, kept in the fixed train/val/test split. Synthetic data only where none exists publicly "
          "(monthly bills, customer IDs, smart-thermostat level, loan-to-value), never used to report accuracy."),
        Paragraph("Changes since M1, and why", st["h2"]),
        b("<b>Physics added back as V1 (grey box).</b> Heat loss per element (roof, walls, windows, floor, ventilation) "
          "x regional degree days, times a learned residual. V2 is the tuned gradient-boosted tree. Both are scored on the "
          "same folds; V1 ships because it models every retrofit physically and drives the heat-loss house diagram."),
        b("<b>Two prediction modes.</b> <i>Similar homes</i> (description only) for homes without bills and for the "
          "'worse than similar homes' detector; <i>this home</i> adds the home's own last-year bill, for existing customers."),
        b("<b>Fairer consumption sub-score.</b> Raw kWh/m<super>2</super> penalised small homes: on real metered use the "
          "smallest homes scored 23.5 points below the largest. It is now actual / expected for similar homes: gap 6.6."),
        b("<b>Feature parity and weather.</b> One feature definition for training and deployment (tested to match). "
          "Each home's gas bills are weather-normalised by a degree-day regression, as the national metered data is."),
        b("<b>Unbiased totals.</b> Log-scale models predict the typical home, which undercounted portfolio totals by "
          "12-17%; a smearing correction fixes it. Engine fixes: heat-pump savings in electrically heated homes, savings "
          "taken off the fuel the heating uses, and what-if scenarios now respect feasibility rules."),
    ]

    # 2. Experiments and results
    rows = [
        ["Model (validation split, real metered use)", "Gas MAE (kWh)", "Gas R<super>2</super> (log)",
         "Gas within 20%", "Elec MAE (kWh)", "Elec R<super>2</super> (log)", "Elec within 20%"],
        ["Global median (no model)", "4,930", "-0.01", "27.6%", "1,714", "0.00", "25.0%"],
        ["Median of similar homes (no model)", "4,131", "0.26", "33.4%", "1,614", "0.11", "27.1%"],
        ["Log-linear", "4,029", "0.30", "33.7%", "1,618", "0.11", "27.0%"],
        ["Physics x one scale factor", "4,439", "0.16", "31.8%", "1,653", "0.03", "25.7%"],
        ["V2: LightGBM, Optuna-tuned", "3,972", "0.30", "35.1%", "1,580", "0.16", "27.6%"],
        ["<b>V1: physics x tuned residual</b>", "<b>3,967</b>", "<b>0.30</b>", "<b>34.9%</b>", "<b>1,578</b>", "<b>0.16</b>", "<b>27.6%</b>"],
        ["This home: last year's bill (no model)", "1,974", "0.68", "70.7%", "969", "0.53", "58.5%"],
        ["<b>This home: change model (GBDT, L1)</b>", "<b>1,923</b>", "<b>0.71</b>", "<b>71.2%</b>", "<b>942</b>", "<b>0.57</b>", "<b>59.0%</b>"],
    ]
    story += [
        Paragraph("2. Experiments and Results", st["h1"]),
        Paragraph("Training data: 50,000-home national sample with metered 2023 gas and electricity; fixed 70/15/15 "
                  "split; 5-fold CV on train for every choice; test split not yet touched.", st["body"]),
        KeepTogether([m1.grid_table(rows, [0.31 * w, 0.115 * w, 0.115 * w, 0.115 * w, 0.115 * w, 0.115 * w, 0.115 * w], st)]),
        Spacer(1, 3),
        b("<b>The inputs cap description-only accuracy.</b> Homes with identical type, age, size, rating and region still "
          "vary so much that these bands explain at most R<super>2</super> 0.25 (gas) / 0.12 (electricity); V1 reaches "
          "0.30 / 0.16. Optuna (50 trials per fuel) gained 1.1% on gas and nothing on electricity. Occupants, not "
          "buildings, drive the rest, so we report the ceiling rather than overclaim."),
        b("<b>The bill is the strongest predictor.</b> 'Last year's bill' alone gets 70.7% of gas homes within 20%; "
          "plain models with the bill as a feature lost to it, so we predict the year-on-year change instead, which beats "
          "it by 2.6% (CV and validation)."),
        b("<b>Physics finding (prebound effect):</b> steady-state physics overpredicts metered gas about 2x (fitted scale "
          "0.51), as documented for real homes heated less than assumed. Retrofit check on 2,968 observed installs: loft "
          "top-up observed -3.1%, V1 predicts -3.3%, V2 cannot (no loft input); cavity wall observed -10.9%, V1 -4.7%, "
          "physics alone -28%."),
        b("<b>Portfolio level:</b> single-home errors cancel: total gas for 100 homes within 3.1%, for 1,000 homes within "
          "0.9%. Picking the top 25% of users: precision 0.55, recall 0.54 for gas (0.47 / 0.46 electricity) vs 0.25 random."),
    ]

    # 3. Testing progress
    story += [
        Paragraph("3. Testing Progress", st["h1"]),
        b("<b>128 automated tests pass:</b> score weights and monotone mappings, size-neutral consumption score, "
          "feasibility rules, carbon, annualisation, a hand-checked heat-loss calculation, physics retrofit directions, "
          "training/deployment feature parity, degree-day fit recovery, schema and API."),
        b("<b>Data verification:</b> 29 checks recomputed independently from the raw files (match rate, no sale uses a "
          "later certificate, 2,000 valuations recomputed exactly, loan balances, label splits); a clean rebuild "
          "reproduces every committed summary. 3,000 real homes run through the engine with 0 errors; per-retrofit savings "
          "sum to the package total for every home."),
        b("<b>Review passes found and fixed real bugs:</b> future months parsed as fake degree days, 282 gas-heated homes "
          "with no gas meter, heat pumps raising electricity in electrically heated homes, an evaluation script that "
          "silently skipped electricity. Known open items are tracked, e.g. the grounding checker's sign handling."),
    ]

    # 4. Plan for the remaining time
    story += [
        Paragraph("4. Plan for the Remaining Time", st["h1"]),
        b("<b>Energy model:</b> conformal p10/p90 bands (target ~80% coverage); one model interface wired into the "
          "savings simulator; retrofit check on held-out installs; then a <b>single</b> test-set scoring and the held-out "
          "predictions file, with errors by region, property type, age and rating for the fairness audit."),
        b("<b>Value model:</b> quantile gradient-boosted price level plus a hedonic rating-band uplift, trained on the "
          "168,351 sales with a pre-sale certificate; interval coverage vs a regional-median baseline."),
        b("<b>Detector and engine:</b> 'worse than similar homes' from actual / expected with a band-based threshold; "
          "remove placeholder figures from the dashboard; fix grounding sign handling and per-home memory."),
        b("<b>LLM and delivery:</b> tool-calling agent over the report JSON with the grounding gate, the ~50-scenario LLM "
          "evaluation, fairness write-up, README, final report and demo."),
        m1.boxed(Paragraph(
            "<b>Reproducibility.</b> <i>make build</i> rebuilds the whole real data layer (~2.5 min), <i>make train</i> "
            "trains and scores every model, <i>make test</i> runs the suite. Splits are fixed and versioned, and every "
            "constant in the configuration carries a source.", st["callout"]), bg=GREEN_BG, border=ACCENT),
    ]
    doc.build(story, canvasmaker=M2Canvas)
    print(f"Successfully generated {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="deliverables/M2.pdf")
    parser.add_argument("--team-number", default="73")
    parser.add_argument("--text-scale", type=float, default=1.35)
    args = parser.parse_args()
    m1.TEXT_SCALE = args.text_scale
    build_pdf(args.output, args.team_number)
