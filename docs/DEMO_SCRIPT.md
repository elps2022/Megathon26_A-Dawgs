# Demo video script (4 minutes)

Team 73. No names, college or team name on screen or in the voice-over.

## Before recording (5 minutes)

1. `git pull`, install both zips (`make models SRC=...`, `make real-data SRC=...`), then `make app`.
2. Optional: `export GEMINI_API_KEY=...` before `make app` so the chat runs on Gemini. Without it the chat uses the offline rule engine. The numbers are the same either way, because they come from the engine.
3. Open `http://localhost:8501/?property_id=Efd940d9b1d`. This is the demo home: a semi-detached house built 1930–1972, rated F, with an uninsulated loft and unfilled cavity walls.
4. Open **Lender Portfolio (Bank View)** once and wait until it loads (about 10 s), then go back to **Customer Experience**. This keeps the load out of the video.
5. Untick every sidebar what-if box. If the floating memory panel (bottom right) is not empty, go to **Recommended Retrofits & Priority → Not Recommended & Excluded → Reset All Memory**.
6. Browser zoom to 80–90% so the four top cards and the house fit on screen. Record at 1080p.
7. Don't click the **Simulate …** button under the element inspector; use the sidebar checkboxes instead.
8. After recording, click **Reset All Memory** again so the demo home is clean for judges.

## Script

| Time | On screen | Voice-over |
|---|---|---|
| **0:00–0:15** | App open on the demo home, header visible | "Every home in a mortgage book gets a Green Home Score, a retrofit plan, and savings, carbon and value, all computed by one engine and checked against real metered homes. Here's one real home." |
| **0:15–0:40** | Point at the rating box, then each of the four top cards | "A mid-century semi, rated F, with an old gas boiler. It scores 20 out of 100: Very poor. The full retrofit package saves 7,246 kilowatt-hours a year. Gas falls by 8,085, but electricity shows minus 839, because the heat pump adds electricity: we track each fuel separately. That's 1.36 tonnes of CO2 a year at the official 2026 factors, and an indicative value of £228k to £236k, plus 4 to 8 percent, because the rating moves from F to C." |
| **0:40–1:05** | 3D thermal twin: drag to rotate, hover the walls, then the roof; point at the legend. Under **Inspect Envelope Element & Fix**, pick **External Walls** | "This 3D twin is built from the home's own description. Each part glows by its share of heat loss from our physics model. The walls lose 55 percent of the heat, with a U-value of 1.6; the uninsulated roof loses 25. The inspector names the fix: cavity wall insulation saves about 1,500 kilowatt-hours and 273 kilos of CO2 a year." |
| **1:05–1:35** | Sidebar: tick **Loft Insulation Top-Up** and **Wall Insulation**; watch the score card and the roof. Then tick **Air Source Heat Pump**. Then untick all three | "Now a what-if. Add loft and wall insulation and the engine reruns: plus 13 points, 2,227 kilowatt-hours of gas saved, and the roof turns teal, from 25 percent of the heat loss to 5. Add a heat pump: plus 28 points. Gas use drops 8,085, while electricity rises by 1,367, shown as a negative saving. Measures run in order, fabric first, so the heat pump is sized for the smaller demand." |
| **1:35–2:10** | Tab **Recommended Retrofits & Priority**. Scroll the ranked cards. Click **Mark Completed** on Loft Insulation, then **Not Interested** on Solar PV. Open the **Not Recommended & Excluded** sub-tab; point at the memory panel | "Only feasible measures, ranked by value for money. The smart thermostat comes first at a 12-year payback, then loft and cavity wall. The heat pump barely moves the bill, but it cuts 846 kilos of CO2 a year. The customer says the loft is already done: the home updates, the score lifts to 26, and loft is never suggested again. They don't want solar: it drops to Low with the date. Excluded measures say why: solid wall insulation doesn't apply to cavity walls, and the windows are already double glazed. All of this is stored in the customer's memory." |
| **2:10–2:25** | Tab **Lloyds Green Banking & Grants** | "Grants and green finance are matched to this home: the £7,500 heat-pump grant, insulation schemes for its F rating, and cashback, with the monthly cash flow of borrowing against the energy savings." |
| **2:25–3:00** | Tab **AI Green Advisor**. Type `What if I add cavity wall insulation and a heat pump?` and send. Expand **Grounding Audit Details**. Then type `My budget is £5,000.` and send | "The assistant never computes numbers. A what-if reruns the engine first: the score goes from 26 to 48, and electricity rises by about 1,500 while gas falls by 8,068. The grounding shield checks every number in the answer against the engine's output. Tell it a budget, and the memory updates: the heat pump drops to Low as over budget." |
| **3:00–3:30** | Sidebar **Lender Portfolio (Bank View)**. Point at the four cards. Drag **Target Retrofit Uptake** from 50 to 80. Scroll to **Top Priority Borrowers** | "The bank's view: 200 mortgaged homes, 37 percent rated Poor or Very poor, and 210.7 tonnes of financed emissions. Retrofits could cut 58.8 percent of these homes' emissions. Push uptake to 80 percent: 59 homes reach band C, avoiding 169 tonnes a year, for £1.2 million of green lending. The outreach list ranks who to call first." |
| **3:30–3:50** | Sidebar **AI Grounding Inspector**. Pick **2. Adversarial Trap: Fabricated Cost (£3,750)**, click **Run Grounding Verification Shield**; point at the claim-by-claim table | "And here's the gate on a fabricated answer: an invented £3,750 cost is blocked, claim by claim. On our 50-scenario benchmark, every trap was caught." |
| **3:50–4:00** | Back to **Customer Experience**, the house on screen | "One report per home: every number computed by the engine, sourced, and checked against real homes. Thank you." |

About 535 words of voice-over (about 3:50 at a steady pace). Keep each segment within its time; if running long, cut the grants segment (2:10–2:25) first.

## Optional extras (only if time allows)

- In the chat, ask `What's the best pizza place nearby?`: the domain guardrail refocuses the conversation.
- Toggle **2D Architecture Cross-Section (SVG)** next to the 3D view.
- In the inspector, the **1. Valid Engine-Grounded Report** preset passes with every number verified.
