# Green Home Score: scoring method

Deliverable D. Code: `src/greenhome/score/score.py`. Config: `config/score_mappings.yaml`.
Tests: `tests/test_score.py`.

## 1. What the score is

A number from 0 to 100 describing how green a home is today. It is built from seven
sub-scores, each between 0 and 100, combined with fixed weights. The weights and the five
bands come from the problem statement. The dashboard shows how many points each part
contributed, so a customer can see why the home got its score.

    total = round( sum over 7 components of weight x sub-score )

Every sub-score, weight, band and default lives in `config/score_mappings.yaml`. No scoring
number is hard-coded in the Python.

## 2. Components and weights

| Component | Weight | Inputs used |
|---|---|---|
| Energy efficiency (EPC) | 25% | `sap_score`, else `energy_rating` |
| Consumption intensity | 20% | `annual_elec_kwh`, `annual_gas_kwh`, `floor_area_m2` |
| Insulation | 15% | `loft_insulation_mm`, `wall_insulation` |
| Heating | 15% | `heating` |
| Glazing | 10% | `glazing` |
| Renewables | 10% | `solar_kwp`, `annual_elec_kwh` |
| Smart controls | 5% | `smart_controls` |

The weights sum to 1.0 (checked by a test).

## 3. How each sub-score is calculated

**Energy efficiency (25%).** If a SAP score exists it is used directly, clipped to 1-100.
Otherwise the EPC band midpoint is used:

| A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|
| 96 | 86 | 75 | 61 | 46 | 30 | 10 |

**Consumption intensity (20%).** Purchased energy (annual electricity + annual gas) **relative to a typical
similar home**: ratio = actual / expected, where expected is the shipped energy model's (V1) typical (median) use
for a home with the same description (`models/energy.similar_home_kwh`, trained on real NEED use); without the
trained models it falls back to the NEED log-linear table (`models/expected_use.py`). Any home with a gas meter is
compared with a similar gas-heated home's gas, so a heat pump isn't compared with NEED's non-gas homes (many
burn unmetered oil). A ratio of 0.5 or less scores 100,
1.0 (typical) scores 75, 2.0 or more scores 0, with straight lines between. Using more never scores higher.
kWh per m2 is still reported in `inputs_used`, but it isn't scored: energy doesn't grow in proportion to floor
area (hot water, cooking and appliances don't double when the house does), so on real NEED use raw kWh/m2 scored
small homes 56 and large homes 79 at the median, a 23.5-point gap that penalised flats and smaller (often
lower-income) homes for their size. Relative to similar homes the gap is 3.6 points with V1 (6.6 with the
log-linear table; widest spread between any two size bands 6.8). Homes heated by oil, LPG or
district heat (flag `heating_fuel_unmetered`) get the default score and a flag, because their heating fuel isn't
in the bills; scoring only their electricity would make them look highly efficient. Homes the NEED model can't
place get the default and `expected_use_unavailable`.

**Insulation (15%).** Half loft, half walls.
- Loft depth: 0 mm = 0, 100 mm = 50, 270 mm or more = 100, straight lines between.
- Walls: none 0, partial 50, full 100.

**Heating (15%).**

| Heat pump | District | Condensing gas | Non-condensing gas | Storage heaters | Oil / LPG | Direct electric |
|---|---|---|---|---|---|---|
| 100 | 70 | 55 | 35 | 30 | 25 | 20 |

**Glazing (10%).**

| Triple | Double (post-2002) | Double (older) | Partial | Single |
|---|---|---|---|---|
| 100 | 80 | 60 | 35 | 0 |

**Renewables (10%).** Estimated solar generation = `solar_kwp` x 900 kWh/kWp/year. The score
is 100 x generation / electricity demand, capped at 100. No solar (0 kWp) scores 0.

**Smart controls (5%).** None 0, programmer/timer 30, smart thermostat 70, smart thermostat plus
radiator valves 100.

## 4. Bands

| Total | Classification |
|---|---|
| 80-100 | Highly efficient |
| 65-79 | Good |
| 50-64 | Moderate |
| 30-49 | Poor |
| 0-29 | Very poor |

Homes classed Poor or Very poor are flagged for the efficiency detector.

## 5. Missing and unrecognised data

Rule 4 of the project: missing data is never silently 0.
- If an input is missing, that component gets a documented default of **40** and a flag
  such as `missing_heating`. The flag travels with the component into the report and the UI.
- If a category value is not in the mapping, the component also gets the default and an
  `unknown_<field>` flag, so mismatches between data and config are visible, not hidden.
- "No solar" (`solar_kwp` = 0) is a real value and scores 0. Only a missing `solar_kwp` is
  treated as missing data.

The default of 40 is a deliberate, conservative choice: it sits in the Poor band, so missing
data cannot push a home into a better band than evidence supports. It is a team decision,
not a published standard.

## 6. Guarantees and how they are tested

- **Weights sum to 1.** Tested.
- **Monotonic:** a better input never gives a lower sub-score. Tested for EPC rating,
  heating, glazing, smart controls, wall insulation, loft depth and consumption intensity.
- **Bands:** boundary values (79/80, 64/65, 49/50, 29/30) classify correctly. Tested.
- **Breakdown adds up:** component contributions sum to the total (within rounding). Tested.
- **Missing data** is flagged and never scored as 0. Tested.
- **Sanity:** a hand-built poor home is Very poor, a hand-built good home scores 80+.

## 7. Known overlap between components

The EPC rating is a modelled whole-home figure, so it already reflects insulation, heating,
glazing and controls. Those components are observed features, so the same upgrade can
influence two parts of the score. We keep the weights fixed as the problem statement
requires, and document this overlap here instead of hiding it.

## 8. Limitations and open items

- The mapping values are team drafts based on the project architecture document. Items
  marked VERIFY in `score_mappings.yaml` still need a published source (EPC band ranges,
  NEED benchmarks, loft depth target, regional solar yield).
- Solar yield uses one flat 900 kWh/kWp, not a regional figure.
- Consumption intensity adds the electricity and gas figures given on the property. It
  matches "purchased energy" only if `annual_elec_kwh` is grid import. Bills are compared with NEED's
  weather-corrected gas without weather-normalising them first, so a mild billing year scores slightly better.
- Category names in the config (for example `gas_condensing`) must match the Property schema names in the
  data. Mismatches show up as `unknown_<field>` flags.
- The data is synthetic, so the score shows how the method behaves, not real-world accuracy.
- The architecture document expects example property P10001 to score about 38 (Poor).
  This check is still to be done on the generated data.
