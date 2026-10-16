# V2 circuit-category check: 2024–2026

Section pace remains disabled. This check uses published qualifying and Grand
Prix classifications, not car/position telemetry. It does not change the model's
training years or session influence.

## Coverage and sources

Jolpi/Ergast API classifications cover 24 weekends in 2024, 24 in 2025 and the
first 17 in 2026: 65 qualifying sessions and 65 Grands Prix, with 2,659 published
driver results. Missing qualifying entries are omitted, not assigned a position.
Sprints are excluded from this check so a Sprint weekend does not count twice.
721 available positions from the uploaded 2026 summaries also matched the
corresponding published API positions, with zero differences.

Sources, with pagination (the API caps pages at 100 entries):
- https://api.jolpi.ca/ergast/f1/2024/results/
- https://api.jolpi.ca/ergast/f1/2024/qualifying/
- https://api.jolpi.ca/ergast/f1/2025/results/
- https://api.jolpi.ca/ergast/f1/2025/qualifying/
- https://api.jolpi.ca/ergast/f1/2026/results/
- https://api.jolpi.ca/ergast/f1/2026/qualifying/

All 26 tracks across the current list and these historical seasons have at
least one category. Sakhir, Jeddah and Imola were added for historical coverage.
The categories remain broad manual assumptions, not official similarity ratings.

Madrid's official layout page, https://www.madring.com/en/circuit, describes
an 837m longest straight, heavy braking zones, slow exits, fast linked corners
and a banked Turn 12. Those support its straight, corner, braking and traction
tags. Tyre-demand membership stays blank because this source does not establish
an appropriate tyre category. Every track having a category does not require
inventing values for every characteristic.

## What was compared

For each driver, predict a later result using only earlier results in the same
season, same session family and same team. Require at least three earlier
appearances and two matching tracks. Normalise positions for 20- versus
22-car fields. Both methods use the same earlier records and the existing
0.85-per-round recency factor:

1. Reference: recent results, without preferring any circuit category.
2. Category check: the same results, additionally weighted by track similarity.

This isolates the information added by categories instead of merely noticing
that a strong driver often finishes well everywhere. A separate descriptive
CSV compares each driver's category results with their own earlier form.
Cross-track agreement also requires the same driver, season and team. Race
checks were repeated on finishers; excluding retirements did not reverse the
main finding. Published race positions can still reflect strategy, weather,
penalties, damage and other factors.

Seven category-weight configurations were checked: current 25/30/15/15/15,
equal weights, and five alternatives each emphasising one characteristic.
A candidate was selected using 2024 only, then checked unchanged on 2025 and
2026. Singapore 2026 is excluded from the weight-comparison holdout, which ends
at round 16. No weights were chosen to reproduce the liked Singapore ranking.
Current category definitions are applied retrospectively; this is not a claim
that these exact profiles existed before the historical events.

## Results

Average position error: lower is better. These are checks of simple historical
averages, not the full XGBoost prediction model or its winner-pick accuracy.

| Season | Session | Without circuit preference | Current category weights |
|---|---|---:|---:|
| 2025 | Qualifying | 3.15 | 3.19 |
| 2025 | Race | 3.69 | 3.72 |
| 2026, through round 16 | Qualifying | 2.17 | 2.23 |
| 2026, through round 16 | Race | 3.96 | 4.10 |

Equal weights were the best of the category candidates on 2024, but they did
not beat the reference on either later season. Differences between equal
weights and the current weights were small and mixed. Resampling whole
weekends also gives no evidence of improvement over the reference. There is
no defensible accuracy-based weight change from this check, so the current
manual weights remain provisional and unchanged.

Drivers do not reliably reproduce the same positions on tracks in one broad
category. For example, these 2026 qualifying results are all in the broad
slow-technical-corner group:

| Driver | Monaco | Hungary | Singapore |
|---|---:|---:|---:|
| Hamilton | 3 | 2 | 3 |
| Leclerc | 4 | 3 | 2 |
| Norris | 8 | 1 | 5 |
| Russell | 6 | 7 | 6 |
| Verstappen | 2 | 6 | 1 |

Some patterns are reasonably consistent; others vary considerably. Finishing
positions alone cannot prove or disprove a physical track characteristic, and
these examples do not establish a stable driver-specific advantage.

## Full-model replay and limits

After completing Madrid's profile, the separate Singapore replay still uses
462 examples from 16 earlier 2026 weekends and FP1/SQ/S/Q as current inputs,
with 25,000 simulations. Section features are excluded. It picks 2 of the same
4 historical winners correctly. Four tests are too few to establish accuracy.
The earlier output is retained separately; Madrid's newly available historical
profile changes the fitted model and therefore the latest ranking.

No Russell grid penalty was applied. The model still defaults to qualifying
order unless a complete verified grid is supplied with --grid. Grid sources
and session influence remain separate tasks.

Raw API responses, analysis scripts, complete driver/category tables and the
paired comparisons are stored outside Git in this cloud workspace. Uploaded
ZIPs and the deferred telemetry handoff are also archived outside Git for v3.
No raw caches, telemetry samples or model outputs were added to this repository.
