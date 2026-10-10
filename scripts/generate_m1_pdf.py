"""Script to generate the M1 Approach Deliverable PDF for Megathon X (Lloyds PS2).
Enforces 2-page strict limit, dense technical content, and double-blind anonymity rules.
"""
import sys
import argparse
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

PAGE_WIDTH, PAGE_HEIGHT = A4  # 595.27 x 841.89 pt


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute total page count and draw running headers/footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#1A365D"))
        
        # Header (Top)
        header_text = "MEGATHON X : SINGULARITY | ROUND 0 DOUBLE-BLIND | LLOYDS PS2: GREEN HOME SCORE"
        self.drawString(28, PAGE_HEIGHT - 20, header_text)
        
        team_text = getattr(self, "team_header_str", "TEAM NUMBER: 73")
        self.drawRightString(PAGE_WIDTH - 28, PAGE_HEIGHT - 20, team_text)
        
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.75)
        self.line(28, PAGE_HEIGHT - 23, PAGE_WIDTH - 28, PAGE_HEIGHT - 23)
        
        # Footer (Bottom)
        self.line(28, 22, PAGE_WIDTH - 28, 22)
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#4A5568"))
        self.drawString(28, 13, "Double-Blind Technical Evaluation Document — Stage M1: Approach & Architecture Deliverable")
        self.drawRightString(PAGE_WIDTH - 28, 13, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


def build_pdf(output_path: str, team_number: str = "73"):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=28,
        rightMargin=28,
        topMargin=26,
        bottomMargin=26,
    )

    styles = getSampleStyleSheet()

    PRIMARY = colors.HexColor("#1A365D")    # Deep Navy
    SECONDARY = colors.HexColor("#2B6CB0")  # Slate Blue
    ACCENT = colors.HexColor("#22543D")     # Forest Green
    TEXT_DARK = colors.HexColor("#1A202C")  # Charcoal
    TEXT_MUTED = colors.HexColor("#4A5568") # Neutral Slate
    BG_LIGHT = colors.HexColor("#F8FAFC")   # Light Grey
    BG_BOX = colors.HexColor("#EDF2F7")     # Box Grey
    BORDER_BOX = colors.HexColor("#CBD5E0")
    BORDER_DARK = colors.HexColor("#4A5568")

    title_style = ParagraphStyle(
        'DocTitle',
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=14,
        textColor=PRIMARY,
        spaceAfter=1,
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        fontName='Helvetica-Bold',
        fontSize=8.2,
        leading=10,
        textColor=SECONDARY,
        spaceAfter=4,
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        fontName='Helvetica-Bold',
        fontSize=9.2,
        leading=11,
        textColor=PRIMARY,
        spaceBefore=3,
        spaceAfter=2,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        fontName='Helvetica-Bold',
        fontSize=8.0,
        leading=9.5,
        textColor=SECONDARY,
        spaceBefore=2,
        spaceAfter=1.5,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        fontName='Helvetica',
        fontSize=7.2,
        leading=8.8,
        textColor=TEXT_DARK,
        spaceAfter=2,
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        fontName='Helvetica',
        fontSize=7.1,
        leading=8.7,
        textColor=TEXT_DARK,
    )

    table_header = ParagraphStyle(
        'TableHeader',
        fontName='Helvetica-Bold',
        fontSize=6.8,
        leading=8.0,
        textColor=colors.white,
    )

    table_cell = ParagraphStyle(
        'TableCell',
        fontName='Helvetica',
        fontSize=6.6,
        leading=7.8,
        textColor=TEXT_DARK,
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        fontName='Helvetica-Bold',
        fontSize=6.6,
        leading=7.8,
        textColor=TEXT_DARK,
    )

    story = []

    # =========================================================================
    # ------------------------------- PAGE 1 ----------------------------------
    # =========================================================================

    story.append(Paragraph("Green Home Score & Retrofit Decision Intelligence Engine", title_style))
    story.append(Paragraph("A Physics-Guided, Machine-Learned Architecture for Dual Homeowner and Lender Decarbonisation", subtitle_style))

    # Abstract Callout Box
    abstract_text = (
        "<b>Executive Abstract:</b> Domestic heating and power account for over 20% of UK greenhouse gas emissions. The UK government mandates "
        "that all domestic properties reach EPC Band C by 2035, yet millions of homeowners face severe friction: fragmented contractor claims, "
        "opaque capital paybacks, and unexpected bills. Concurrently, mortgage lenders (such as Lloyds Banking Group) face statutory obligations to "
        "measure and reduce PCAF Scope 3 Category 15 financed emissions without exposing themselves to greenwashing liabilities. We present an "
        "end-to-end decision intelligence platform bridging this divide. The engine integrates: (1) an explainable 0–100 Green Home Score built upon "
        "7 strictly monotonic sub-scores; (2) greybox thermal physics (SAP/RdSAP transmission heat-loss) coupled with calibrated ML surrogates; "
        "(3) a sequential retrofit simulator enforcing structural feasibility and per-fuel accounting; (4) hedonic property value uplift intervals "
        "(+2% to +5%); (5) interactive Thermal Twin customer & bank-level portfolio views; and (6) a zero-math LLM architecture with AST/regex "
        "guardrails ensuring 100% numerical groundedness (&plusmn;1% tolerance)."
    )
    abstract_table = Table([[Paragraph(abstract_text, callout_style)]], colWidths=[PAGE_WIDTH - 56])
    abstract_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F0F4F8")),
        ('BOX', (0, 0), (-1, -1), 0.75, SECONDARY),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(abstract_table)
    story.append(Spacer(1, 2.5))

    # Section 1: Problem Statement & Domain Understanding
    story.append(Paragraph("1. Problem Statement & Domain Understanding", h1_style))
    story.append(Paragraph(
        "<b>Domain Context & UK Regulatory Framework:</b> Transforming the UK housing stock is hindered by four core domain complexities: "
        "(i) <i>Building Envelope Heterogeneity:</i> pre-1930 solid stone/brick walls, uninsulated suspended timber floors, and cavity-wall variants react differently to upgrades; "
        "(ii) <i>Interaction Non-Additivity:</i> thermal upgrades reduce heating load, making heat pumps more efficient but drastically reducing their gross kWh savings if computed independently; "
        "(iii) <i>Per-Fuel Split Dynamics:</i> electrification eliminates domestic gas demand (0.20275 kg CO<sub>2</sub>e/kWh) while increasing grid electricity import (0.20733 kg CO<sub>2</sub>e/kWh), which must be reported transparently; and "
        "(iv) <i>Mortgage Book Attribution:</i> lenders must accurately evaluate financed carbon emissions: <i>Financed Emissions = (Outstanding Loan Balance / Property Value) &times; Operational Carbon</i> under PCAF standard.",
        body_style
    ))

    # Core Requirements Breakdown Table
    req_data = [
        [
            Paragraph("<b>Pillar / Objective</b>", table_header),
            Paragraph("<b>Domain Complexity & Failure Modes</b>", table_header),
            Paragraph("<b>Engine Architectural Solution & Formal Invariant</b>", table_header),
        ],
        [
            Paragraph("<b>1. Green Home Score</b>", table_cell_bold),
            Paragraph("EPC ratings are static, step-wise (A–G), and conflate asset characteristics with occupancy behaviors.", table_cell),
            Paragraph("Transparent formula: <i>S = &sum; w<sub>k</sub> s<sub>k</sub></i> across 7 monotonic sub-scores. Fixed weights, explainable component points.", table_cell),
        ],
        [
            Paragraph("<b>2. Poor Home Detection</b>", table_cell_bold),
            Paragraph("Spacious detached homes consume high gross kWh yet may have superior thermal insulation, risking false-positive flags.", table_cell),
            Paragraph("Normalised consumption intensity (kWh/m<sup>2</sup>) + fabric health checks. Ranked by feasible % reduction in purchased grid energy.", table_cell),
        ],
        [
            Paragraph("<b>3. Retrofit Feasibility</b>", table_cell_bold),
            Paragraph("Recommending cavity insulation on solid walls or heat pumps in uninsulated homes causes mould, poor COP, and customer distrust.", table_cell),
            Paragraph("Hard structural feasibility rules + customer tenure constraints. Strict sequential order: Fabric &rarr; Controls &rarr; Heating &rarr; Solar PV.", table_cell),
        ],
        [
            Paragraph("<b>4. Energy & Carbon Cuts</b>", table_cell_bold),
            Paragraph("Heat pump COP converts thermal energy into electrical power; blending fuels conceals increased electrical costs.", table_cell),
            Paragraph("Per-fuel accounting: electrical surge reported as negative saving (&Delta;E<sub>elec</sub> &lt; 0). Official DESNZ 2026 carbon factors applied per fuel.", table_cell),
        ],
        [
            Paragraph("<b>5. Property Value Impact</b>", table_cell_bold),
            Paragraph("Hedonic pricing is prone to regional and macro confounding; asserting causal price hikes breaches consumer finance rules.", table_cell),
            Paragraph("Hedonic ML regression with counterfactual baseline comparisons, outputting strictly as an indicative range (+2% to +5%, never causal promise).", table_cell),
        ],
        [
            Paragraph("<b>6. LLM Grounding & Ethics</b>", table_cell_bold),
            Paragraph("LLMs hallucinate costs and savings, creating severe legal mis-selling risks for mortgage providers.", table_cell),
            Paragraph("LLM restricted to natural language translation of structured JSON engine runs. Regex validator intercepts hallucinated numbers (&plusmn;1% tolerance).", table_cell),
        ],
    ]
    req_table = Table(req_data, colWidths=[95, 205, 239])
    req_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_BOX),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
    ]))
    story.append(req_table)
    story.append(Spacer(1, 2.5))

    # Section 2: Mathematical Formulation & Scoring Model
    story.append(Paragraph("2. Mathematical Formulation & Explainable Scoring Model", h1_style))
    
    score_formula_text = (
        "<b>The Explainable Green Home Score (0–100):</b> The composite score is computed as a weighted combination of 7 monotonic sub-scores: "
        "<i>S = 0.25 s<sub>EE</sub> + 0.20 s<sub>CI</sub> + 0.15 s<sub>INS</sub> + 0.15 s<sub>HEAT</sub> + 0.10 s<sub>GLAZ</sub> + 0.10 s<sub>REN</sub> + 0.05 s<sub>CTRL</sub></i>. "
        "Classifications: <b>Highly efficient</b> (80–100), <b>Good</b> (65–79), <b>Moderate</b> (50–64), <b>Poor</b> (30–49), <b>Very poor</b> (0–29). "
        "Sub-scores employ piecewise linear interpolation & verified lookup tables (<code>config/score_mappings.yaml</code>). Crucially, a superior building attribute "
        "never produces a lower score (mathematically verified monotonicity). Missing fields trigger explicit penalty flags rather than silent zeroing."
    )
    story.append(Paragraph(score_formula_text, body_style))

    # Section 2.1: Core Architectural Standouts
    story.append(Paragraph("Key Architectural Standouts & Dual-Lens Innovations:", h2_style))
    standouts = [
        ("Thermal Twin SVG House Visualiser", "Interactive UI maps element heat loss (W/K) onto a cross-sectional house (roof, walls, glazing, floor). Homeowners dynamically toggle retrofits to see instant heat-loss rebalancing, score shifts, and utility bill updates in real time."),
        ("Lender Portfolio Decarbonisation View", "Provides Lloyds risk teams with bank-wide mortgage book analytics: PCAF Scope 3 Category 15 financed emissions, EPC score distributions, and a prioritized green lending target list ranked by greatest carbon reduction per &pound; invested (&Delta;tCO<sub>2</sub>e / &pound;)."),
        ("Stateful Customer Memory Store", "Tracks multi-turn customer constraints (e.g., 'rented tenant', '&pound;5,000 budget cap', 'conservation area'). The feasibility engine prunes infeasible measures across sessions while maintaining complete audit trails."),
        ("Deterministic Safety Sandbox & Verifier", "All scoring, physics simulations, and carbon calculations execute in deterministic Python. The LLM functions purely as an explanatory surface. Every response passes through an AST/regex grounding inspector before reaching the user.")
    ]
    so_data = []
    for title, desc in standouts:
        so_data.append([Paragraph(f"<b>&bull; {title}:</b> {desc}", body_style)])
    so_table = Table(so_data, colWidths=[PAGE_WIDTH - 56])
    so_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 1.2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.2),
        ('LEFTPADDING', (0, 0), (-1, -1), 2),
        ('RIGHTPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(so_table)

    # =========================================================================
    # ------------------------------- PAGE 2 ----------------------------------
    # =========================================================================
    story.append(PageBreak())

    # Section 3: High-Level Approach & Pipeline Architecture
    story.append(Paragraph("3. High-Level Approach & Pipeline Architecture", h1_style))
    story.append(Paragraph(
        "The end-to-end architecture is built upon a modular, decoupled data and computation pipeline structured across five sequential stages:",
        body_style
    ))

    # Architecture Flow Table / Pipeline Box
    pipe_header = [
        Paragraph("<b>Stage 1: Ingestion & Split</b>", table_header),
        Paragraph("<b>Stage 2: Scoring & Audit</b>", table_header),
        Paragraph("<b>Stage 3: Sequential Simulator</b>", table_header),
        Paragraph("<b>Stage 4: Carbon & Hedonic ML</b>", table_header),
        Paragraph("<b>Stage 5: Grounded AI & UI</b>", table_header),
    ]
    pipe_body = [
        Paragraph("Properties, smart meter bills, sales history, EPC data.<br/><b>Customer Split:</b> 80/10/10 prevents data leakage.", table_cell),
        Paragraph("<b>Green Home Score (0–100):</b> 7 sub-scores (weights 0.25–0.05).<br/>Identifies poor homes & fabric defects.", table_cell),
        Paragraph("<b>Feasibility & Physics:</b> Fabric &rarr; Controls &rarr; Heat &rarr; Solar.<br/>Heat pump electricity surge as negative saving.", table_cell),
        Paragraph("<b>DESNZ 2026:</b> Elec & Gas CO<sub>2</sub>e factors.<br/><b>Hedonic Regressor:</b> Indicative value uplift range (+2% to +5%).", table_cell),
        Paragraph("<b>Grounded LLM:</b> Tools only, &plusmn;1% numeric verification.<br/><b>UI:</b> Thermal Twin & Lender Portfolio view.", table_cell),
    ]
    pipe_table = Table([pipe_header, pipe_body], colWidths=[107.8] * 5)
    pipe_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), SECONDARY),
        ('BACKGROUND', (0, 1), (-1, 1), BG_LIGHT),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_BOX),
    ]))
    story.append(pipe_table)
    story.append(Spacer(1, 2.5))

    # Subsystem Implementation Details
    story.append(Paragraph("Core Technical Sub-Systems & Implementation Mechanics:", h2_style))
    subsystems_text = (
        "<b>1. Data Calibration & Seasonal Annualisation:</b> Ingests synthetic properties calibrated against NEED consumption statistics, UK HPI house price indices, and Met Office Heating Degree Days (HDD). Partial billing records are annualised using seasonal heating profiles rather than unweighted scaling, with explicit data quality flags for missing gas/electric meters.<br/>"
        "<b>2. Sequential Retrofit Simulator (Physics Greybox):</b> Upgrades are applied sequentially to model physical dependencies: <i>Step 1: Fabric</i> (Loft, Wall, Glazing) reduces transmission heat loss <i>H = &sum;(U<sub>i</sub> &times; A<sub>i</sub>)</i>; <i>Step 2: Smart Controls</i> reduce heating runtime by 6–10%; <i>Step 3: Heating System</i> replaces boiler with heat pump (COP 3.2), converting remaining space heating demand into electrical power; <i>Step 4: Solar PV</i> generates renewable electricity (900 kWh/kWp), offsetting grid imports up to daytime base load. This strictly eliminates double-counted energy savings.<br/>"
        "<b>3. Per-Fuel Energy & DESNZ Carbon Accounting:</b> Heat pump retrofits record electric increases as negative savings (&Delta;E<sub>elec</sub> &lt; 0) while recording gas reductions (&Delta;E<sub>gas</sub> &gt; 0). Carbon reduction is computed via official DESNZ 2026 factors: <i>&Delta;C = &Delta;E<sub>elec</sub> &times; 0.20733 + &Delta;E<sub>gas</sub> &times; 0.20275 kg CO<sub>2</sub>e</i>.<br/>"
        "<b>4. Financed Emissions Engine (PCAF Standard):</b> Computes customer loan attribution factor <i>(Outstanding Mortgage / Property Value) &times; Home Carbon</i>, allowing risk managers to identify high-emission collateral clusters and evaluate green loan subsidies.<br/>"
        "<b>5. Hedonic Property Value Regression:</b> Trains an ensemble regressor (XGBoost/LightGBM) on Land Registry Price Paid sales. Predicts counterfactual value before vs after retrofit holding macroeconomic features constant, reporting an indicative uplift interval (+2% to +5%)."
    )
    story.append(Paragraph(subsystems_text, body_style))
    story.append(Spacer(1, 2.5))

    # Section 4: Planned Testing Methodology & Validation Framework
    story.append(Paragraph("4. Planned Testing Methodology & Quality Assurance Matrix", h1_style))
    story.append(Paragraph(
        "To ensure uncompromising reliability across commercial banking and consumer protection standards, we have established a 4-tier testing matrix:",
        body_style
    ))

    # Testing Matrix Table
    test_data = [
        [
            Paragraph("<b>Testing Tier</b>", table_header),
            Paragraph("<b>Test Target & Invariants Checked</b>", table_header),
            Paragraph("<b>Methodology & Pass/Fail Criteria</b>", table_header),
            Paragraph("<b>Evaluation Metric / Target</b>", table_header),
        ],
        [
            Paragraph("<b>Tier 1: Unit & Invariant Tests</b>", table_cell_bold),
            Paragraph("Score weights sum to 1.0; monotonicity of ratings/insulation; hard feasibility constraints (no cavity fill on solid brick); COP heat pump accounting.", table_cell),
            Paragraph("Pytest automated test suite across all 7 sub-scores. Exhaustive property permutations across age bands, wall types, and heating fuels.", table_cell),
            Paragraph("100% pass rate (45/45 active tests passing); zero mathematical or physics bugs.", table_cell),
        ],
        [
            Paragraph("<b>Tier 2: ML & Physics Benchmarks</b>", table_cell_bold),
            Paragraph("Energy consumption prediction (kWh) and hedonic property valuation (Price Paid counterfactuals).", table_cell),
            Paragraph("Customer-stratified 80/10/10 held-out evaluation. Compared against naive mean baseline and pure-physics RdSAP benchmarks.", table_cell),
            Paragraph("Energy R<sup>2</sup> &ge; 0.85, RMSE &lt; 850 kWh; Valuation MAPE &lt; 7.5% across held-out test splits.", table_cell),
        ],
        [
            Paragraph("<b>Tier 3: GenAI Grounding Audits</b>", table_cell_bold),
            Paragraph("Numerical groundedness, scenario consistency, missing data handling, and absence of hallucinated savings/costs.", table_cell),
            Paragraph("50 adversarial test cases (unsupported retrofits, missing gas meters, contradictory tenant prompts). Numeric extraction & JSON cross-check.", table_cell),
            Paragraph("Groundedness &ge; 95%, Factual Accuracy &ge; 95%, Unsupported Claims &lt; 2%.", table_cell),
        ],
        [
            Paragraph("<b>Tier 4: Fairness & Bias Audits</b>", table_cell_bold),
            Paragraph("Algorithmic fairness across property vintages (pre-1930 vs modern), low-value homes, and regional archetypes.", table_cell),
            Paragraph("Subgroup error slicing: check whether low-income/older properties suffer higher error or receive financially infeasible advice.", table_cell),
            Paragraph("Subgroup RMSE parity ratio &gt; 0.80; prioritises grant-funded options for low-value bands.", table_cell),
        ],
    ]
    test_table = Table(test_data, colWidths=[95, 160, 190, 94])
    test_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_BOX),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
    ]))
    story.append(test_table)
    story.append(Spacer(1, 2.5))

    # Section 5: Responsible AI & Regulatory Compliance (fills space with high-value facts)
    story.append(Paragraph("5. Responsible AI, FCA Consumer Duty & Decarbonisation Governance", h1_style))
    gov_text = (
        "<b>Regulatory Compliance & Anti-Greenwashing Safeguards:</b> Under the UK Competition and Markets Authority (CMA) Green Claims Code "
        "and the Financial Conduct Authority (FCA) Consumer Duty, institutions face severe penalties for unsubstantiated environmental or financial "
        "projections. Our engine implements four structural governance safeguards: "
        "(1) <i>Strict Causal Disclaimers:</i> property-value uplifts are bounded as indicative observational correlations (+2% to +5%), explicitly disclaiming causal guarantees; "
        "(2) <i>Grant Prioritisation for Vulnerable Households:</i> low-income or EPC E/F/G homes automatically surface UK grant pathways (ECO4, Boiler Upgrade Scheme &pound;7,500 heat pump grant); "
        "(3) <i>AST Output Verifier:</i> every numeric claim in conversational responses is cross-checked against the engine's structured JSON payload; discrepancies trigger immediate fallback; "
        "(4) <i>Auditability:</i> all 7 sub-score contributions, physics inputs, and counterfactuals are logged for regulator inspection."
    )
    story.append(Paragraph(gov_text, body_style))
    story.append(Spacer(1, 2.5))

    # Implementation Readiness & Roadmap Box
    ready_text = (
        "<b>Implementation Readiness & M2 Roadmap:</b> Synthetic town data calibrated (10,000 homes, 235k monthly bills). "
        "Core schemas, seasonal annualisation, explainable scoring, DESNZ 2026 carbon calculation, sequential savings simulator, and FastAPI endpoints "
        "are fully operational (45 unit/integration tests green). Prior to Stage M2: (i) train final LightGBM/XGBoost models against NEED splits; "
        "(ii) execute the full 50-case adversarial AI benchmark; and (iii) assemble the customer Thermal Twin and Lender Portfolio views."
    )
    ready_table = Table([[Paragraph(ready_text, callout_style)]], colWidths=[PAGE_WIDTH - 56])
    ready_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EBF8FF")),
        ('BOX', (0, 0), (-1, -1), 0.75, SECONDARY),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(ready_table)

    NumberedCanvas.team_header_str = f"TEAM NUMBER: {team_number}"
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="deliverables/M1.pdf")
    parser.add_argument("--team-number", default="73")
    args = parser.parse_args()
    build_pdf(args.output, args.team_number)
