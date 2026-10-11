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
        Paragraph("2. Approach: What Is New", st["h1"]),
        b("<b>A grey-box energy model calibrated on real retrofits, not on its own equations.</b> V1 computes heat loss "
          "per element (roof, walls, windows, floor, air) times regional degree days, then log(use) = &beta; &middot; "
          "log(physics) + GBDT(home description). Steady-state physics over-predicts metered gas about 2&times; (the "
          "prebound effect: real homes are heated less than assumed), and over-predicts retrofit savings 3&times;. "
          "We fit the elasticity &beta; = 0.38 on the observed installs in the train split only, so a retrofit's effect "
          "is (physics after / before)<super>&beta;</super>: smooth, never the wrong sign, with diminishing returns "
          "and interactions from the physics, and a per-element heat-loss share for the house diagram. The residual "
          "never sees the physics value: when it did, 18% of gas homes were told insulation raises use."),
        b("<b>Non-circular evaluation.</b> Headline accuracy comes only from real metered homes the model never saw, "
          "on a fixed split with the <b>test set scored once</b>; every choice was made on 5-fold CV. Changes made after "
          "that are labelled post-test and judged on validation. Synthetic bills carry hidden occupancy and behaviour "
          "the engine never sees, and are used only to test the pipeline and the parts real data can't (controls, "
          "heat pumps, glazing)."),
        b("<b>Two prediction modes, and a model that learns from the customer's own bills.</b> <i>Similar homes</i> "
          "(description only) for prospects and for the detector, which must not see the home's own use; <i>this home</i> "
          "for existing customers, a GBDT on the year-on-year change log(this year / last year). Since the bill alone "
          "carries most of the signal, it is the reference to beat. Each year of history is weather-corrected to a "
          "typical year with the home's own degree-day fit, as the national metered data is; the same fit splits each "
          "home's heating from hot water and cooking for the savings simulator."),
        b("<b>Honest uncertainty with stated error rates.</b> Conformalised quantile models (CQR) give every prediction "
          "an 80% range. The 'worse than similar homes' flag fires above the 90th percentile for identical homes, so "
          "its false-alarm rate is 10% by construction, and it is checked in every region, type, age, rating and "
          "deprivation group. A big house is compared with similar big houses."),
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
          "template. Customer memory turns stated constraints (tenant, declined measures) into feasibility rules."),
    ]

    # 3. Features
    story += [
        Paragraph("3. Features", st["h1"]),
        b("<b>Customer view:</b> score with points per part; heat-loss house (3D and SVG) coloured by each element's "
          "share of heat loss, click a part to see its fix; what-if toggles that rerun the engine and move the score, "
          "savings and house together; ranked recommendation cards (bill saving, payback, carbon, cost, why); grants "
          "and green-finance matching with monthly cash flow; the grounded assistant with an audit of every number."),
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
        ["Median of similar homes (no model)", "4,177", "30.7%", "0.25", "1,569", "37.9%"],
        ["Log-linear model", "4,061", "30.3%", "0.28", "1,573", "38.6%"],
        ["Physics x one scale factor", "4,546", "33.5%", "0.14", "-", "-"],
        ["V2: LightGBM, Optuna-tuned", "4,008", "29.4%", "0.28", "1,533", "37.2%"],
        ["<b>V1 grey box (shipped), description only</b>", "<b>4,018</b>", "<b>29.6%</b>", "<b>0.27</b>", "<b>1,535</b>", "<b>37.2%</b>"],
        ["This home: last year's bill (no model)", "2,005", "11.4%", "0.67", "917", "16.0%"],
        ["<b>This home: change model with the bill</b>", "<b>1,953</b>", "<b>10.9%</b>", "<b>0.69</b>", "<b>886</b>", "<b>15.6%</b>"],
    ]
    retro_rows = [
        ["Real installs, test split", "Observed", "Physics alone", "<b>V1</b>", "V2 (GBDT)"],
        ["Cavity wall insulation (225 homes)", "-11.4%", "-31.5%", "<b>-13.4%</b>", "-3.8%"],
        ["Loft insulation (319 homes)", "-1.3%", "-5.3%", "<b>-2.1%</b>", "no loft input"],
    ]
    story += [
        Paragraph("4. Results", st["h1"]),
        Paragraph("Energy model", st["h2"]),
        KeepTogether([m1.grid_table(energy_rows, [0.34 * w, 0.13 * w, 0.13 * w, 0.13 * w, 0.13 * w, 0.14 * w], st)]),
        Spacer(1, 3),
        b("<b>Single homes are noisy, books are not.</b> Identical homes vary 0.5-1.7&times; in gas use, which caps "
          f"description-only accuracy (Optuna tuning gained 1.1% on gas; CatBoost and interaction features moved MAE &lt; 0.3%, inside fold noise). "
          f"Yet predicted totals for 1,000 homes are within <b>1.7%</b> of metered, gas {R2} is <b>0.79</b> across "
          "groups of similar homes, and the top 25% of users are found with precision 0.55 (0.25 random). That is what "
          "a lender's portfolio view needs."),
        b("<b>80% ranges hold on new homes:</b> test coverage 79.2% gas, 80.7% electricity, and 76-84% in every "
          "region, type, age, rating and deprivation group. The detector flags 10.2% (gas) / 9.7% (electricity) of test "
          "homes against 10% expected, and 6-12% in every group."),
        KeepTogether([m1.grid_table(retro_rows, [0.36 * w, 0.16 * w, 0.16 * w, 0.16 * w, 0.16 * w], st)]),
        Spacer(1, 3),
        b("<b>Savings match real installs.</b> One fitted &beta; brings V1 within about 2 points of the observed change "
          "for both measures on held-out installs, where physics alone overstates 3-4&times;. The tuned GBDT sees only "
          "a third of the cavity saving: a model trained on consumption levels learns a weak between-home difference, "
          "not a before-after change. This, not accuracy (the two tie), is why V1 ships. The solar rule doubled real "
          "savings (1,260 vs 487 kWh/yr observed); recalibrated on train installs it predicts -488 vs -483 on validation."),
        figure_row(["eval_retrofit_check.png", "eval_test_calibration.png"], w,
                   ["Retrofit savings vs real installs, held-out splits: V1 tracks the observed change, physics alone "
                    "overstates it 3-4x.",
                    "Test calibration by decile of V1's prediction: mean metered use follows the diagonal; shading is "
                    "the mean 80% range."], st),
        Spacer(1, 2),
        b("<b>Post-test, chosen on validation:</b> a second year of the customer's bills cuts error a further 1.5% (gas) "
          "and 2.7% (electricity); weather-correcting the history removes a 7% under-forecast after a mild year "
          "(synthetic check: error 798 &rarr; 308 kWh). On 1,175 synthetic mid-series retrofits covering all seven "
          "measures, the predicted total is within 3% of the hidden truth and a heat pump's electricity rise has the "
          "right sign every time."),
        Paragraph("Value, recommendations, assistant", st["h2"]),
        b("<b>Value (test):</b> median error <b>14.9%</b> vs 24.6% for the area x type median and 32.6% for the "
          f"baseline in the brief; {R2} 0.76. For homes sold before, 85% indexed last sale + 15% model gives 8.2%. "
          "80% ranges cover 79.3%, and 78-81% in every area, type and rating band."),
        b("<b>Recommendations:</b> against hidden true value per pound on 1,691 synthetic homes, NDCG@3 is <b>0.978</b> "
          "vs 0.813 for the brief's fixed-order baseline, and the top pick is right 91% vs 38%. A strong 'cheapest first' "
          "reference nearly ties on bill value (0.968), but the engine clearly wins on carbon per pound (0.968 vs 0.904)."),
        b("<b>Assistant:</b> on 50 held-out adversarial scenarios, all 42 traps (invented figures, wrong signs, "
          "infeasible asks) were blocked and all 8 legitimate answers passed: groundedness 100%, unsupported claims 0%. "
          "What-if answers on 20 real homes: 60 of 60 grounded after the engine rerun."),
        Paragraph("Honest limits", st["h2"]),
        b("V1 gets the <i>average</i> saving right, but real installs save about the same whatever physics predicts "
          "(cavity slope 0.01 on train vs 0.38 assumed), so per-home fabric savings for the worst homes are the "
          "optimistic end. Electricity is driven by occupants ({R2} 0.16 from the description). Monthly bills for real "
          "homes don't exist publicly, so bill-based gains are shown on validation and synthetic data. Property value "
          "lacks a street-level input. Each is quantified in the evaluation docs.".replace("{R2}", R2)),
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
    parser.add_argument("--text-scale", type=float, default=1.25)
    args = parser.parse_args()
    m1.TEXT_SCALE = args.text_scale
    build_pdf(args.output, args.team_number)
