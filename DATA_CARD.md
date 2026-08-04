# MetroPT-3 Data Card

## Source and license

- **Dataset:** MetroPT-3
- **Repository:** UCI Machine Learning Repository
- **DOI:** https://doi.org/10.24432/C5VW3R
- **Creators:** Narjes Davari, Bruno Veloso, Rita Ribeiro, and João Gama
- **License:** Creative Commons Attribution 4.0 International

MetroGuard downloads the UCI archive directly and verifies its pinned SHA-256 digest. The raw
archive and prepared feature table are excluded from Git. The compact `data/demo/replay.csv`
extract is redistributed with attribution under the source license.

## Contents

The released CSV contains 1,516,948 observations from one Air Production Unit on a Metro do
Porto train, spanning 1 February to 1 September 2020. Seven analog signals measure pressure,
oil temperature, and motor current. Eight digital signals represent compressor and valve state,
pressure switches, oil level, and airflow impulses.

UCI metadata contains both 1 Hz and 0.1 Hz descriptions. Inspection of the published timestamps
shows an approximately 10-second cadence with jitter and longer gaps, so the pipeline measures
coverage and never assumes a perfect clock.

## Consolidated incident annotations

| ID | Start | End | Reported condition |
|---|---|---|---|
| 1 | 2020-04-18 00:00 | 2020-04-18 23:59 | Air leak / high stress |
| 2 | 2020-05-29 23:30 | 2020-05-30 06:00 | Air leak / high stress |
| 3 | 2020-06-05 10:00 | 2020-06-07 14:30 | Air leak / high stress |
| 4 | 2020-07-15 14:30 | 2020-07-15 19:00 | Air leak / high stress |

The DSAA study describes more expert-defined windows, while UCI publishes these four
consolidated events. MetroGuard uses the UCI record as the canonical annotation and documents
the difference instead of silently combining definitions.

## Processing and leakage controls

- Sort and deduplicate timestamps; retain small negative analog readings as measurement noise.
- Aggregate into right-closed, causal 5-minute bins without interpolation or backfill.
- Reject bins below 80% of the expected 30 readings.
- Build 12-bin windows only across exactly consecutive bins.
- Fit scaling and representations only on 1–21 February.
- Lock score normalization and thresholds only on 22–29 February.
- Preserve March–September for holdout evaluation, with a 60-minute purge at boundaries.

## Known limitations

- One train/APU cannot represent fleet, route, climate, or hardware variation.
- Only four consolidated events are available and all concern air leakage.
- “No reported failure” is not proof of healthy operation.
- Maintenance records are limited, so post-maintenance effects cannot be isolated confidently.
- Pointwise samples are highly autocorrelated and must not be treated as independent evidence.

