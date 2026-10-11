"""Generate deliverables/final_report.pdf (Megathon X, PS2: final report, max 3 pages).

Sections follow the deliverables spec: solution, approach, features and results, with the emphasis on what is new.
Double-blind: team number only. Reuses the M1 layout helpers. Every number comes from eval/results/*.json and
docs/EVALUATION.md (headline energy and value figures are the single held-out test scoring; anything chosen after
it is labelled post-test).

    uv run --no-project --with reportlab python scripts/generate_final_pdf.py
"""
import argparse

from pathlib import Path

from reportlab.graphics.shapes import Drawing, PolyLine, String
from reportlab.lib import colors
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import generate_m1_pdf as m1
from generate_m1_pdf import A4, ACCENT, BORDER, GREEN_BG, MARGIN, PAGE_HEIGHT, PAGE_WIDTH, PRIMARY, TEXT_MUTED

STAGE = "FINAL REPORT"
FIGURES = Path(__file__).resolve().parents[1] / "eval" / "figures"


def pipeline_diagram() -> Drawing:
    """The M1 pipeline (snake layout, what-if loop), with the boxes updated to what was built."""
    from generate_m1_pdf import CONTENT_W, SECONDARY, arrow, box
    w_total, gap, bw, bh = CONTENT_W, 12, (CONTENT_W - 4 * 12) / 5, 52
    d = Drawing(w_total, 2 * bh + 52)
    y1, y2 = bh + 40, 26
    xs = [i * (bw + gap) for i in range(5)]
    blue, green, amber = colors.HexColor("#EBF4FF"), GREEN_BG, colors.HexColor("#FFF7E6")
    row1 = [
        ("1. Data layer", ["Certificates, metered use,", "sales, price index,", "weather, real installs"], blue),
        ("2. Features", ["Seasonal annualisation,", "weather-corrected bills,", "heating share, parity"], blue),
        ("3. Score + energy model", ["7 sub-scores, size-fair;", "V1 grey box + bill model,", "80% conformal range"], green),
        ("4. Detector", ["Above p90 of identical", "homes: 10% false-alarm", "rate, fair by group"], green),
        ("5. Retrofit engine", ["Feasibility rules, memory;", "rank by payback + carbon,", "install order fabric first"], green),
    ]
    row2 = [  # drawn right -> left
        ("6. Savings simulator", ["Sequential per-fuel", "re-prediction; marginal", "savings sum to package"], green),
        ("7. Carbon + value", ["Carbon = sum dE x EF;", "value range, same home", "and date, before vs after"], green),
        ("8. PropertyReport JSON", ["Single source of truth", "for every number shown", "(score, kWh, CO2e, value)"], amber),
        ("9. LLM explainer", ["Tool calls only; grounding", "gate: every number", "must be in tool JSON"], amber),
        ("10. Dashboards + API", ["Customer: score, house,", "recs, chat. Lender:", "book + emissions"], blue),
    ]
    for i, (t, ls, fill) in enumerate(row1):
        box(d, xs[i], y1, bw, bh, t, ls, fill, SECONDARY)
        if i < 4:
            arrow(d, xs[i] + bw, y1 + bh / 2, xs[i + 1], y1 + bh / 2)
    for i, (t, ls, fill) in enumerate(row2):
        x = xs[4 - i]
        box(d, x, y2, bw, bh, t, ls, fill, SECONDARY)
        if i < 4:
            arrow(d, x, y2 + bh / 2, xs[3 - i] + bw, y2 + bh / 2)
    arrow(d, xs[4] + bw / 2, y1, xs[4] + bw / 2, y2 + bh)
    lx, sx, yb = xs[1] + bw / 2, xs[4] + bw / 2 - 18, y2 - 12
    d.add(PolyLine([lx, y2, lx, yb, sx, yb], strokeColor=colors.HexColor("#C05621"), strokeWidth=1.1,
                   strokeDashArray=[3, 2]))
    arrow(d, sx, yb, sx, y2, color=colors.HexColor("#C05621"), dashed=True)
    d.add(String((lx + sx) / 2, yb - 9, "what-if: run_scenario(retrofits) reruns the engine before the LLM answers",
                 fontName="Helvetica-Oblique", fontSize=6.8, fillColor=colors.HexColor("#C05621"), textAnchor="middle"))
    return d


def figure_row(files: list[str], width: float, captions: list[str], st) -> Table:
    cells = []
    for f, cap in zip(files, captions):
        img = Image(str(FIGURES / f))
        scale = (width / len(files) - 6) / img.imageWidth
        img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
        cells.append([img, Paragraph(cap, st["td"])])
    t = Table([[c[0] for c in cells], [c[1] for c in cells]], colWidths=[width / len(files)] * len(files))
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                           ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    return t


class FinalCanvas(m1.NumberedCanvas):
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
    FinalCanvas.team_header_str = f"TEAM NUMBER: {team_number}"
    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=32, bottomMargin=30)
    st = m1.styles()
    b = lambda text: Paragraph(text, st["bullet"], bulletText="•")
    w = m1.CONTENT_W
    R2 = "R<super>2</super>"
    model_rows = [
        ["Model", "Predicts", "Method", "Training data and split"],
        ["<b>Energy V1 (shipped)</b>", "Annual gas and electricity from the home's description",
         "Heat-loss physics x LightGBM residual (Huber loss, Optuna-tuned); physics elasticity fitted on real installs",
         "50,000 metered homes (2023); fixed 70/15/15 split stratified by region x fuel; 5-fold CV on train"],
        ["Energy V2", "Same", "LightGBM direct, same tuning, monotone constraints", "Same split and folds"],
        ["Bill model", "Next year's use for a customer with bills",
         "LightGBM on log(this year / last year), L1 loss; one or two years of history", "Same homes, 2021-2023"],
        ["80% ranges", "p10 / p90 for every prediction", "Quantile LightGBM + split-conformal correction (CQR)",
         "Fitted on train, calibrated on validation"],
        ["<b>Value</b>", "Price range today, before vs after",
         "LightGBM on log price indexed to the valuation date, monotone in rating; CQR range",
         "151,587 real resales 2018-2025; 106k / 23k / 22k by customer; CV grouped by home"],
        ["Rating premium", "% value per rating band", "Hedonic log-linear, 259 postcode-sector fixed effects, clustered errors",
         "Train-split sales"],
        ["Rating after retrofit", "SAP score after a measure", "Linear regression on fabric fields",
         "376,769 real certificates (train customers)"],
    ]
    story = [
        Paragraph("Green Home Score & Retrofit Recommendation Engine", st["title"]),
        Paragraph("A deterministic engine that scores a home, recommends feasible retrofits in the right order and "
                  "estimates savings, carbon and value, validated on real metered homes and real installs, with an "
                  "assistant that can only explain the engine's numbers", st["subtitle"]),
    ]

    # 1. Solution
    story += [
        Paragraph("1. Solution", st["h1"]),
        Paragraph("For each home the engine builds one <b>PropertyReport</b>, the single source of truth that the "
                  "customer dashboard, the lender portfolio view, the API and the assistant all read and never recompute: "
                  "a 0-100 Green Home Score with its seven weighted parts, expected electricity and gas use with an 80% "
                  "range, a 'worse than similar homes' check, feasible retrofits in priority order (fabric, controls, "
                  "heating, generation) with marginal per-fuel savings, bill saving and payback, carbon from official 2026 "
                  "factors, and an indicative property value range before vs after. It runs on <b>542,711 real homes</b> "
                  "from energy performance certificates in three contrasting local authorities (urban terraces, rural "
                  "off-gas cottages, inner-city flats), joined to 175,457 real sales (92.2% match, only certificates "
                  "lodged before each sale). Models learn from a 50,000-home national sample of <b>metered</b> annual "
                  "gas and electricity, and are checked against <b>4,791 real loft, cavity-wall and solar installs</b> "
                  "with metered use two years before and after. Synthetic data fills only what no public source has "
                  "(monthly bills, customer IDs, smart controls, loan-to-value) and is never used to report accuracy.",
                  st["body"]),
        Spacer(1, 2),
        pipeline_diagram(),
    ]

    # 2. Approach: what is new
    story += [
        Paragraph("2. Approach", st["h1"]),
        Paragraph("Data, models and training", st["h2"]),
        KeepTogether([m1.grid_table(model_rows, [0.17 * w, 0.20 * w, 0.33 * w, 0.30 * w], st)]),
        Spacer(1, 3),
        b("<b>Inputs and leakage rules.</b> One feature function serves training and deployment, using only what a "
          "certificate also gives: property type, age band, floor area, rating band, region, heating fuel, deprivation "
          "quintile and retrofit flags (tested to match). Targets are log annual kWh per fuel. Installs count only "
          "before the target year, sales use only the certificate lodged before the sale, and splits are by home or "
          "customer, fixed and versioned. Every choice (objective, Optuna search, monotone constraints, features, "
          "LightGBM vs CatBoost) was made on 5-fold CV of the train split; validation calibrates the ranges; the test "
          "split was scored once."),
        b("<b>Deterministic engine (non-ML).</b> Monthly bills are annualised with seasonal weights and flagged when "
          "coverage is low; a per-home degree-day regression (PRISM) weather-corrects them and splits heating from "
          "base load. Heat loss uses RdSAP-style U-values and geometry with SAP hot-water and appliance formulas. "
          "The score combines seven monotone sub-scores with the fixed weights; feasibility rules come from a sourced "
          "retrofit catalogue; the savings simulator applies measures in order (fabric, controls, heating, solar), "
          "each re-predicting from what the last left, per fuel; carbon uses official 2026 factors and bills the "
          "current price cap. Every constant lives in config with its source. The assistant is Gemini 2.5 Flash with "
          "tool calling behind a grounding gate; FastAPI and Streamlit serve the API and dashboards."),
        Paragraph("What is new", st["h2"]),
        b("<b>A grey-box energy model calibrated on real retrofits, not on its own equations.</b> V1 computes heat loss "
          "per element (roof, walls, windows, floor, air) times regional degree days, then log(use) = &beta; &middot; "
          "log(physics) + GBDT(home description). Steady-state physics over-predicts metered gas about 2&times; (the "
          "prebound effect: real homes are heated less than assumed), and over-predicts retrofit savings 3&times;. "
          "We fit the elasticity &beta; = 0.38 on the observed installs in the train split only, so a retrofit's effect "
          "is (physics after / before)<super>&beta;</super>: smooth, never the wrong sign, with diminishing returns "
          "and interactions from the physics, and a per-element heat-loss share for the house diagram."),
        b("<b>Non-circular evaluation.</b> Headline accuracy comes only from real metered homes and real installs. "
          "Synthetic bills carry hidden occupancy and behaviour the engine never sees, and are used only to test the "
          "pipeline and the parts real data can't (controls, heat pumps, glazing), never to report accuracy."),
        b("<b>Two prediction modes, and a model that learns from the customer's own bills.</b> <i>Similar homes</i> "
          "(description only) for prospects and for the detector, which must not see the home's own use; <i>this home</i> "
          "for existing customers, a GBDT on the year-on-year change log(this year / last year). Since the bill alone "
          "carries most of the signal, it is the reference to beat. Each year of history is weather-corrected to a "
          "typical year with the home's own degree-day fit, as the national metered data is; the same fit splits each "
          "home's heating from hot water and cooking for the savings simulator."),
        b("<b>Honest uncertainty with stated error rates.</b> Conformalised quantile models (CQR) give every prediction "
          "an 80% range. The 'worse than similar homes' flag fires above the 90th percentile for identical homes, so "
          "its false-alarm rate is 10% by construction, and it is checked in every region, type, age, rating and "
          "deprivation group."),
        b("<b>A size-fair score.</b> Raw kWh/m<super>2</super> scored small homes 23.5 points below large ones on real "
          "metered use, penalising flats and smaller (often lower-income) homes for their size. The consumption part "
          "now scores actual / expected for similar homes: the gap falls to 3.6 points. Every mapping is monotone and "
          "in config with a source."),
        b("<b>A value uplift that compares homes on the same streets.</b> Without location controls the rating premium "
          "comes out <b>backwards</b> (badly rated homes are often expensive central period houses). A hedonic model "
          "with 259 postcode-sector fixed effects and clustered errors gives the expected order (vs D: A/B +1.0%, C "
          "+0.7%, E -2.7%, F/G -5.1%), confirmed by a refit on test sales. The report shows the same home on the same "
          "date before vs after, as a range that can include 0, labelled an association, never causal."),
        b("<b>Recommendations ranked by value and checked against hidden truth.</b> Suitability comes from payback at "
          "current capped prices, with a carbon floor and an 'insulate first' rule for heat pumps. The ranking is "
          "scored against the synthetic world's hidden true savings, not the engine's own numbers."),
        b("<b>An assistant that cannot invent numbers.</b> The LLM only reads tool output (report, scenario rerun, "
          "customer memory). A what-if question reruns the engine before the model answers, and a grounding gate "
          "checks every number in the answer against the tool JSON, retries once, then falls back to a deterministic "
          "template."),
    ]

    story.insert(len(story), b("<b>Customer memory that changes the engine's inputs, not just the chat.</b> Each "
          "customer has a persistent profile the assistant reads and writes through tools. Completed retrofits update "
          "the home before the engine runs, so they are never recommended again and savings start from the new state; "
          "stated constraints become feasibility rules (a tenant gets no structural works, measures above a stated "
          "budget drop to Low); a declined measure drops to Low with the reason shown ('You declined this on 12 Oct') "
          "and every rank is recomputed. Numbers still come only from the engine."))

    # 3. Features
    story += [
        Paragraph("3. Features", st["h1"]),
        b("<b>Customer view:</b> score with points per part; heat-loss house (3D and SVG) coloured by each element's "
          "share of heat loss, click a part to see its fix; what-if toggles that rerun the engine and move the score, "
          "savings and house together; ranked recommendation cards (bill saving, payback, carbon, cost, why); grants "
          "and green-finance matching with monthly cash flow; the grounded assistant with an audit of every number; "
          "'Mark completed' / 'Not interested' buttons and a three-session memory timeline."),
        b("<b>Lender view:</b> a mortgage book of real homes: score and rating mix, share Poor / Very poor, financed "
          "emissions (PCAF, loan-to-value attribution), reduction potential, a 2030 EPC C target simulator and a "
          "ranked list of who to contact about green finance."),
        b("<b>Electricity and gas are tracked separately end to end;</b> a heat pump's extra electricity is a negative "
          "saving, and solar cuts what is bought from the grid, not demand. Missing data is 'unavailable' and flagged, "
          "never 0. FastAPI exposes reports, scenarios, portfolio, chat and memory."),
    ]

    # 4. Results
    energy_rows = [
        ["Real metered homes, test split (scored once)", "Gas MAE (kWh)", "Gas median error", f"Gas {R2} (log)",
         "Elec MAE (kWh)", "Elec median error"],
        ["Median of similar homes (no-model baseline)", "4,177", "30.7%", "0.25", "1,569", "37.9%"],
        ["<b>V1 grey box, description only</b>", "<b>4,018</b>", "<b>29.6%</b>", "<b>0.27</b>", "<b>1,535</b>", "<b>37.2%</b>"],
        ["Last year's bill alone (no-model baseline)", "2,005", "11.4%", "0.67", "917", "16.0%"],
        ["<b>V1 with the customer's own bills</b>", "<b>1,953</b>", "<b>10.9%</b>", "<b>0.69</b>", "<b>886</b>", "<b>15.6%</b>"],
    ]
    retro_rows = [
        ["Real installs, test split", "Observed", "Physics alone", "<b>V1</b>", "V2 (GBDT)"],
        ["Cavity wall insulation (225 homes)", "-11.4%", "-31.5%", "<b>-13.4%</b>", "-3.8%"],
        ["Loft insulation (319 homes)", "-1.3%", "-5.3%", "<b>-2.1%</b>", "no loft input"],
    ]
    value_rows = [
        ["Price model, real resales, test split", "Median error", "Within 20%", f"{R2} (log)", "MAE (GBP)"],
        ["Area median x fixed rating premium (brief's baseline)", "32.6%", "32.0%", "0.17", "150,200"],
        ["<b>Value model (LightGBM, conformal range)</b>", "<b>14.9%</b>", "<b>62.8%</b>", "<b>0.76</b>", "<b>73,900</b>"],
        ["<b>Homes sold before: last sale indexed + model</b>", "<b>8.2%</b>", "<b>82.9%</b>", "<b>0.87</b>", "<b>44,100</b>"],
    ]
    rank_rows = [
        ["Ranking of feasible retrofits (1,691 homes)", "NDCG@3, bill saving per GBP", "Top pick right",
         "NDCG@3, carbon per GBP", "Top pick right (carbon)"],
        ["Fixed order, all feasible (brief's baseline)", "0.813", "37.6%", "0.846", "45.5%"],
        ["<b>Engine (payback suitability)</b>", "<b>0.978</b>", "<b>91.1%</b>", "<b>0.968</b>", "<b>89.2%</b>"],
    ]
    story += [
        KeepTogether([Paragraph("4. Results", st["h1"]), Paragraph("Energy model", st["h2"]),
                      m1.grid_table(energy_rows, [0.34 * w, 0.13 * w, 0.13 * w, 0.13 * w, 0.13 * w, 0.14 * w], st)]),
        Spacer(1, 3),
        b("<b>Accurate where it matters for a lender:</b> predicted totals for 1,000 homes are within <b>1.7%</b> of "
          f"metered use, gas {R2} is <b>0.79</b> across groups of similar homes, and with the customer's own bills the "
          "median error is <b>10.9%</b>, better than the bill alone on the same held-out homes."),
        b("<b>Calibrated 80% ranges:</b> test coverage 79.2% gas and 80.7% electricity, and 76-84% in every region, "
          "type, age, rating and deprivation group; the detector flags 10.2% / 9.7% of homes against 10% by design."),
        b("<b>Bill history:</b> a second year of bills cuts error a further 1.5% (gas) / 2.7% (electricity) on "
          "validation, and weather correction removes a 7% under-forecast after a mild year."),

        Paragraph("Retrofit savings model", st["h2"]),
        KeepTogether([m1.grid_table(retro_rows, [0.36 * w, 0.16 * w, 0.16 * w, 0.16 * w, 0.16 * w], st)]),
        Spacer(1, 3),
        b("<b>Savings match real installs:</b> one elasticity fitted on train installs brings V1 within about 2 points "
          "of the observed change on held-out installs, where physics alone overstates 3-4&times;. Solar, recalibrated "
          "on real installs, predicts -488 vs -483 kWh/yr observed."),
        b("<b>All seven measures:</b> on 1,175 synthetic homes that retrofit mid-series, seeing only pre-install bills, "
          "the predicted total saving is within <b>3%</b> of the hidden truth and the sign is right for 96% of homes "
          "and every heat pump and solar install."),

        Paragraph("Property value model", st["h2"]),
        KeepTogether([m1.grid_table(value_rows, [0.40 * w, 0.15 * w, 0.15 * w, 0.15 * w, 0.15 * w], st)]),
        Spacer(1, 3),
        b("<b>Price level:</b> LightGBM trained on 106k real resales matched to the certificate lodged before each sale, "
          "monotone in rating, with conformal 80% ranges (test coverage <b>79.3%</b>, 78-81% in every area, type and "
          "rating band)."),
        b("<b>Rating uplift:</b> comparing homes on the same streets (259 postcode sectors) gives the premium vs band D: "
          "A/B +1.0%, C +0.7%, E -2.7%, F/G -5.1%, confirmed by a refit on test sales. A retrofit's new rating comes from "
          f"a SAP-score model on 376,769 real certificates ({R2} 0.70)."),

        Paragraph("Recommendations and assistant", st["h2"]),
        KeepTogether([m1.grid_table(rank_rows, [0.34 * w, 0.165 * w, 0.165 * w, 0.165 * w, 0.165 * w], st)]),
        Spacer(1, 3),
        b("<b>Ranking</b> is scored against hidden true saving per pound, not the engine's own numbers: the top pick is "
          "right 91% of the time vs 38% for the brief's fixed-order list."),
        b("<b>Assistant:</b> on 50 held-out adversarial scenarios, groundedness <b>100%</b> and unsupported claims "
          "<b>0%</b> (all 42 traps blocked); what-if answers on 20 real homes: 60 of 60 grounded after the engine rerun."),
        m1.boxed(Paragraph(
            "<b>Engineering.</b> 186 automated tests (monotone score mappings, feasibility, per-fuel carbon, physics "
            "directions, feature parity between training and deployment, grounding). <i>make build</i> rebuilds the "
            "real data layer, <i>make train</i> and <i>make eval-test</i> train and score every model, and trained models "
            "and processed data install from hosted links in one command each (verified from a clean clone). Every "
            "constant in the configuration names its source.", st["callout"]), bg=GREEN_BG, border=ACCENT),
    ]
    doc.build(story, canvasmaker=FinalCanvas)
    print(f"Successfully generated {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="deliverables/final_report.pdf")
    parser.add_argument("--team-number", default="73")
    parser.add_argument("--text-scale", type=float, default=1.22)
    args = parser.parse_args()
    m1.TEXT_SCALE = args.text_scale
    build_pdf(args.output, args.team_number)
