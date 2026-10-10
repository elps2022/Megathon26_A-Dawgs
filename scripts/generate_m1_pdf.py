"""Generate deliverables/M1.pdf (Megathon X, PS2: approach document, max 2 pages).

Sections follow the deliverables spec: problem statement and our understanding, idea / abstract,
pipeline (with diagram) and planned testing methodology. Double-blind: team number only.

    uv run --no-project --with reportlab python scripts/generate_m1_pdf.py
"""
import argparse

from reportlab.graphics.shapes import Drawing, Line, Polygon, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PAGE_WIDTH, PAGE_HEIGHT = A4  # 595.27 x 841.89 pt
MARGIN = 30
CONTENT_W = PAGE_WIDTH - 2 * MARGIN
# Scales all text sizes and spacing; tuned so the content fills exactly 2 pages.
TEXT_SCALE = 1.25

PRIMARY = colors.HexColor("#1A365D")
SECONDARY = colors.HexColor("#2B6CB0")
ACCENT = colors.HexColor("#22543D")
TEXT_DARK = colors.HexColor("#1A202C")
TEXT_MUTED = colors.HexColor("#4A5568")
BG_BOX = colors.HexColor("#EDF2F7")
BG_CALLOUT = colors.HexColor("#F0F4F8")
BORDER = colors.HexColor("#CBD5E0")
GREEN_BG = colors.HexColor("#E6F4EA")


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas: running header (team number only) and 'Page x of y' footer."""

    team_header_str = "TEAM NUMBER: 73"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(n)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(PRIMARY)
        self.drawString(MARGIN, PAGE_HEIGHT - 20, "MEGATHON X : SINGULARITY  |  PROBLEM STATEMENT 2  |  M1: APPROACH")
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


def styles():
    s = TEXT_SCALE
    return {
        "title": ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=14 * s, leading=17 * s, textColor=PRIMARY),
        "subtitle": ParagraphStyle("st", fontName="Helvetica-Oblique", fontSize=8.6 * s, leading=11 * s,
                                   textColor=SECONDARY, spaceAfter=4 * s),
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=10.5 * s, leading=13 * s, textColor=PRIMARY,
                             spaceBefore=5 * s, spaceAfter=2.5 * s, keepWithNext=True),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=8.6 * s, leading=10.5 * s, textColor=ACCENT,
                             spaceBefore=3 * s, spaceAfter=1.5 * s, keepWithNext=True),
        "body": ParagraphStyle("b", fontName="Helvetica", fontSize=8 * s, leading=10.2 * s, textColor=TEXT_DARK,
                               spaceAfter=2.5 * s, alignment=4),
        "bullet": ParagraphStyle("bl", fontName="Helvetica", fontSize=8 * s, leading=10.2 * s, textColor=TEXT_DARK,
                                 leftIndent=10, bulletIndent=2, spaceAfter=1.2 * s, alignment=4),
        "callout": ParagraphStyle("c", fontName="Helvetica", fontSize=8 * s, leading=10.3 * s, textColor=TEXT_DARK,
                                  alignment=4),
        "th": ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=7.6 * s, leading=9.2 * s, textColor=colors.white),
        "td": ParagraphStyle("td", fontName="Helvetica", fontSize=7.4 * s, leading=9 * s, textColor=TEXT_DARK),
        "tdb": ParagraphStyle("tdb", fontName="Helvetica-Bold", fontSize=7.4 * s, leading=9 * s, textColor=PRIMARY),
    }


def boxed(paragraph, bg=BG_CALLOUT, border=SECONDARY):
    t = Table([[paragraph]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg), ("BOX", (0, 0), (-1, -1), 0.75, border),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def grid_table(rows, col_widths, st):
    data = [[Paragraph(c, st["th"]) for c in rows[0]]]
    for r in rows[1:]:
        data.append([Paragraph(r[0], st["tdb"])] + [Paragraph(c, st["td"]) for c in r[1:]])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_BOX]),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    return t


def arrow(d, x1, y1, x2, y2, color=TEXT_MUTED, dashed=False):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.1, strokeDashArray=[3, 2] if dashed else None))
    # Arrowhead pointing from (x1,y1) towards (x2,y2); only axis-aligned arrows are used.
    h = 4.5
    if x2 > x1:
        pts = [x2, y2, x2 - h, y2 + h / 1.6, x2 - h, y2 - h / 1.6]
    elif x2 < x1:
        pts = [x2, y2, x2 + h, y2 + h / 1.6, x2 + h, y2 - h / 1.6]
    elif y2 < y1:
        pts = [x2, y2, x2 - h / 1.6, y2 + h, x2 + h / 1.6, y2 + h]
    else:
        pts = [x2, y2, x2 - h / 1.6, y2 - h, x2 + h / 1.6, y2 - h]
    d.add(Polygon(pts, fillColor=color, strokeColor=color, strokeWidth=0.5))


def box(d, x, y, w, h, title, lines, fill, border):
    d.add(Rect(x, y, w, h, rx=3, ry=3, fillColor=fill, strokeColor=border, strokeWidth=0.9))
    cx = x + w / 2
    d.add(String(cx, y + h - 11, title, fontName="Helvetica-Bold", fontSize=7.6, fillColor=PRIMARY, textAnchor="middle"))
    for i, line in enumerate(lines):
        d.add(String(cx, y + h - 21 - i * 8.6, line, fontName="Helvetica", fontSize=6.6, fillColor=TEXT_DARK,
                     textAnchor="middle"))


def pipeline_diagram() -> Drawing:
    """Snake layout: row 1 left->right, then down, row 2 right->left; dashed what-if loop under row 2."""
    w_total, gap, bw, bh = CONTENT_W, 12, (CONTENT_W - 4 * 12) / 5, 52
    d = Drawing(w_total, 2 * bh + 52)
    y1, y2 = bh + 40, 26
    xs = [i * (bw + gap) for i in range(5)]
    blue, green, amber = colors.HexColor("#EBF4FF"), GREEN_BG, colors.HexColor("#FFF7E6")
    row1 = [
        ("1. Data layer", ["Real public data:", "certificates, metered use,", "sales, price index, weather"], blue),
        ("2. Features", ["Seasonal annualisation,", "kWh/m2, data-quality", "flags, band mapping"], blue),
        ("3. Score + energy model", ["7 sub-scores, fixed w;", "tuned GBDT per fuel", "(expected use)"], green),
        ("4. Efficiency detector", ["Poor / Very poor flag,", "key issues, intensity vs", "similar homes"], green),
        ("5. Retrofit engine", ["Feasibility rules,", "priority order: fabric >", "controls > heat > solar"], green),
    ]
    row2 = [  # drawn right -> left
        ("6. Savings simulator", ["Sequential per-fuel", "re-prediction; marginal", "savings sum to package"], green),
        ("7. Carbon + value", ["Carbon = sum dE x EF;", "value p10-p90 range,", "before vs after"], green),
        ("8. PropertyReport JSON", ["Single source of truth", "for every number shown", "(score, kWh, CO2e, value)"], amber),
        ("9. LLM explainer", ["Tool calls only; grounding", "checker: every number", "must be in tool JSON"], amber),
        ("10. Dashboards", ["Customer: score, house,", "recs, chat. Lender:", "portfolio + emissions"], blue),
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
    arrow(d, xs[4] + bw / 2, y1, xs[4] + bw / 2, y2 + bh)  # retrofit engine -> savings
    # What-if loop: LLM (col 1) -> reruns retrofit scenario at savings simulator (col 4).
    lx, sx, yb = xs[1] + bw / 2, xs[4] + bw / 2 - 18, y2 - 12
    d.add(PolyLine([lx, y2, lx, yb, sx, yb], strokeColor=colors.HexColor("#C05621"), strokeWidth=1.1,
                   strokeDashArray=[3, 2]))
    arrow(d, sx, yb, sx, y2, color=colors.HexColor("#C05621"), dashed=True)
    d.add(String((lx + sx) / 2, yb - 9, "what-if: run_scenario(retrofits) reruns steps 6-8 before the LLM answers",
                 fontName="Helvetica-Oblique", fontSize=6.8, fillColor=colors.HexColor("#C05621"), textAnchor="middle"))
    return d


def build_pdf(output_path: str, team_number: str = "73"):
    NumberedCanvas.team_header_str = f"TEAM NUMBER: {team_number}"
    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=32, bottomMargin=30)
    st = styles()
    b = lambda text: Paragraph(text, st["bullet"], bulletText="•")
    story = [
        Paragraph("Green Home Score & Retrofit Recommendation Engine", st["title"]),
        Paragraph("Explainable scoring, accuracy-first energy models and a grounded LLM explainer, "
                  "for homeowners and for a lender's mortgage book", st["subtitle"]),
    ]

    # 1. Problem statement and our understanding
    story += [
        Paragraph("1. Problem Statement and Our Understanding", st["h1"]),
        Paragraph("Many customers want an energy-efficient home but don't know where to start. The task is an AI system "
                  "that analyses a property's characteristics, energy consumption and existing features to (1) calculate a "
                  "0-100 <b>Green Home Score</b>, (2) identify poorly performing homes, (3) recommend <b>feasible</b> "
                  "retrofits (insulation, heat pumps, solar, glazing, smart controls), (4) estimate energy savings and "
                  "carbon reduction, and (5) estimate the potential property-value impact. It must answer: <i>how green "
                  "is my home, what should I improve, how much could I save, and could it raise my home's value?</i>",
                  st["body"]),
        Paragraph("What we take from it: the requirements that make this hard", st["h2"]),
        b("<b>Numbers come from code, never the LLM.</b> Score, savings, carbon and value are computed by documented "
          "models; the LLM only explains them. A what-if must rerun the models first, and then the LLM explains the "
          "new outputs."),
        b("<b>Per-fuel accounting.</b> Electricity and gas are tracked separately end to end. Purchased energy = grid "
          "import + gas; solar cuts imports, not demand; a heat pump lowers gas but raises electricity, which is reported "
          "as a <i>negative</i> saving. Carbon = sum over fuels of dE<sub>f</sub> x EF<sub>f</sub>."),
        b("<b>Interactions, not sums.</b> Retrofits are ordered (insulation before a heat pump), and each acts on the "
          "demand left by the previous one, so package savings are not the sum of individual estimates."),
        b("<b>Fair efficiency judgement.</b> A large home must not be flagged just for using more energy: we judge "
          "intensity (kWh/m<super>2</super>) against what <i>similar homes</i> are expected to use."),
        b("<b>Honest uncertainty.</b> Savings are estimates; value impact is an indicative <i>range</i> for the same home "
          "on the same date, never a causal claim; missing data is reported as unavailable, never silently 0."),
        b("<b>Explainable, fixed-weight score.</b> Seven sub-scores with fixed weights (25/20/15/15/10/10/5%), each "
          "mapping monotonic (a better state never scores lower) and documented, with each component's contribution shown."),
    ]

    # 2. Idea and abstract
    story += [
        Paragraph("2. Idea and Abstract", st["h1"]),
        boxed(Paragraph(
            "We build a <b>deterministic engine with a thin, grounded LLM on top</b>. Every property flows through one "
            "pipeline that produces a single <b>PropertyReport</b> JSON: score with per-component contributions, "
            "Poor/Very-poor flag and key issues, a prioritised list of feasible retrofits, per-fuel marginal and package "
            "savings, carbon, and an indicative value range. The core models are trained on <b>real public data</b>: a "
            "national anonymised sample of 50,000 homes with <b>metered</b> annual gas and electricity (2005-2023, "
            "including dated loft, cavity-wall and solar installs), energy performance certificates for three contrasting "
            "local authorities (a northern city, a rural off-gas-grid county and a dense inner-city borough), house-sale "
            "transactions with a regional price index, and regional monthly temperatures for heating degree days. "
            "Synthetic data is used only where none exists publicly (monthly bills per certificate home, customer IDs, "
            "smart-control level), and is <b>never used to report accuracy</b>. Because model accuracy matters most, the "
            "energy model is a tuned gradient-boosted tree per fuel with monotone constraints and calibrated p10/p90 "
            "bands. Retrofits the data cannot teach (heat pump, glazing, smart controls) use a small, sourced and clearly "
            "labelled rule layer. The LLM can only call tools (<i>get_property_report</i>, <i>run_scenario</i>, memory), "
            "and a grounding checker rejects any answer containing a number that is not in the tool output. The same "
            "reports feed a <b>customer dashboard</b> (score breakdown, heat-loss house, savings, chat) and a <b>lender "
            "portfolio view</b> (score mix, financed emissions, customers ranked by improvement potential).",
            st["callout"])),
    ]

    # 3. Pipeline
    story += [
        Paragraph("3. Pipeline and Architecture", st["h1"]),
        KeepTogether([pipeline_diagram()]),
        Spacer(1, 3),
        Paragraph("How each stage works", st["h2"]),
        b("<b>Data and features (1-2).</b> Monthly bills are annualised with separate seasonal weights for gas and "
          "electricity; low coverage is flagged, never filled with 0. Certificate fields are mapped to the same bands "
          "the energy model was trained on."),
        b("<b>Green Home Score (3).</b> Seven monotonic sub-score mappings live in a versioned config, each with a source. "
          "Energy efficiency uses the recorded rating; consumption intensity uses annualised metered kWh/m<super>2</super>, "
          "to avoid double-counting."),
        b("<b>Energy model (3).</b> One model per fuel, trained on real metered consumption: a LightGBM vs CatBoost "
          "bake-off tuned with Optuna on identical cross-validation folds, monotone constraints (worse rating or less "
          "insulation never predicts lower use), and conformalised p10/p90 bands. Its residual (actual vs expected for "
          "similar homes) drives the efficiency detector (4)."),
        b("<b>Retrofit engine and savings (5-6).</b> Hard feasibility rules (e.g. no cavity fill on solid walls, no "
          "solar without roof access) give a prioritised package with High/Medium/Low suitability. The simulator applies "
          "retrofits in priority order and re-predicts each fuel after each step, so per-retrofit savings are marginal "
          "and add up to the package total."),
        b("<b>Carbon and value (7).</b> Carbon uses official greenhouse-gas conversion factors per fuel. Value: a quantile "
          "model gives the price level, and a hedonic model's rating-band coefficients give a smooth uplift <i>range</i> "
          "for the same home and date, labelled indicative."),
        b("<b>LLM and dashboards (8-10).</b> The LLM explains only report JSON, says 'not available' for nulls, and "
          "cites the rule that blocked an infeasible retrofit. A what-if reruns steps 6-8 first."),
    ]

    # 4. Planned testing methodology
    rows = [
        ["Layer", "What we test", "How", "Target / metric"],
        ["Unit and invariant tests", "Weights sum to 1; every sub-score mapping monotonic; sample poor home "
         "(rating E, gas boiler, single glazing) scores ~38 / Poor; missing data gives 'unavailable', not 0",
         "pytest on score, carbon, feasibility, annualisation and schema; property-based checks over mapping grids",
         "100% pass, run on every commit"],
        ["Energy model accuracy", "Annual electricity and gas prediction on homes the model never saw",
         "Fixed 70/15/15 split of the metered sample, test set scored once; 5-fold CV for all tuning",
         "MAE, MAPE, R<super>2</super> per fuel; must beat median-of-similar-homes and log-linear baselines"],
        ["Savings validity", "Predicted effect of loft, cavity-wall and solar installs",
         "Compare with observed before/after changes in the metered sample (difference-in-differences)",
         "Within the observed effect range; catalogue ranges as sanity bounds"],
        ["Interactions and per-fuel logic", "Ordered package; heat pump raises electricity; solar cuts imports only",
         "Scenario tests on fixed homes through the full engine", "Marginal savings sum to package; signs correct"],
        ["Value model", "Price level and rating uplift", "Held-out sales; interval coverage; baseline: regional "
         "median x rating premium", "p10-p90 coverage ~80%; beats baseline"],
        ["Leakage audit", "Every feature available at prediction time",
         "Installs after the target year excluded; splits by customer; no post-retrofit or post-sale fields",
         "Automated tests on the feature pipeline"],
        ["LLM grounding", "~50 held-out scenarios: missing or conflicting data, infeasible asks, heat-pump "
         "electricity rise", "Grounding checker extracts every number and matches it to tool JSON",
         "Groundedness >= 90%, accuracy >= 90%, unsupported claims < 5%"],
        ["Fairness audit", "Score, flag rate and model error by region, property type, age band and value band",
         "Test-split breakdowns; affordability flag puts low-cost options first for high energy-burden homes",
         "No group systematically worse; limitations documented"],
    ]
    w = CONTENT_W
    story += [
        Paragraph("4. Planned Testing Methodology", st["h1"]),
        Paragraph("Testing is layered from deterministic code to models to the LLM. Model results are only reported on "
                  "held-out real data, and the test split is touched once.", st["body"]),
        grid_table(rows, [0.15 * w, 0.32 * w, 0.31 * w, 0.22 * w], st),
        Spacer(1, 5),
        boxed(Paragraph(
            "<b>Reproducibility.</b> Every result is regenerated from the repository with fixed seeds: <i>make data</i> "
            "(synthetic bills, seed 42), <i>make train</i> (energy model with baselines; cross-validation and validation "
            "only), <i>make test</i> (unit and invariant suite). The train/validation/test split is saved once and "
            "versioned, so the test set cannot drift between runs, and every constant in the configuration (score "
            "mappings, retrofit catalogue, emission factors, efficiencies) carries a <i>source</i> field.",
            st["callout"]), bg=GREEN_BG, border=ACCENT),
    ]
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="deliverables/M1.pdf")
    parser.add_argument("--team-number", default="73")
    parser.add_argument("--text-scale", type=float, default=None)
    args = parser.parse_args()
    if args.text_scale:
        TEXT_SCALE = args.text_scale
    build_pdf(args.output, args.team_number)
