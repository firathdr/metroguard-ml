# MetroGuard v1 Model Card

## Intended use

MetroGuard is a reproducible research and portfolio demonstration of early-warning anomaly
detection on MetroPT-3. It is intended for offline historical analysis, model comparison, and
discussion of time-series validation design.

It is **not** intended for live railway maintenance, safety decisions, root-cause diagnosis, or
remaining-useful-life estimation.

## Pre-registered primary model

The primary model is a small temporal convolutional autoencoder. It receives 12 consecutive
five-minute feature vectors, uses Conv1D channels 32 → 16 → an 8-channel bottleneck, and then a
symmetric decoder. It minimizes mean squared reconstruction error and stops using only the
February calibration loss.

Three fixed comparators are included: top-three robust-z deviation, PCA reconstruction at 95%
explained variance, and a 300-tree Isolation Forest. Test results never alter model parameters
or the primary-model designation.

## Alert policy

Calibration scores are normalized by median/MAD, filtered by causal EWMA with alpha 0.2, and
thresholded at calibration q=0.995. An alert requires three exceedances in four five-minute
scores. Episodes within 30 minutes merge and a six-hour cooldown follows.

## Evaluation

- Early event recall as `k/4` for alarms 24–2 hours before incident start.
- Late detection for alarms in the final two hours or during the incident.
- False alarm episodes per normal exposure day.
- Lead time, alarm precision, and percentage of time in alert.
- PR-AUC on official incident intervals; ROC-AUC is secondary.
- 6/12/24/48-hour sensitivity and per-event results.

The exact generated metrics are stored in `reports/metrics.json` and copied into the release
artifact metadata.

## Explanations

The autoencoder reports sensor-group reconstruction-error contributions. Isolation Forest can
be inspected with TreeSHAP on sampled windows. These are signals associated with a high anomaly
score, not causal or mechanical root-cause statements.

## Limitations and risks

- Four events imply extremely uncertain event-level performance.
- Threshold performance may change across other trains and operating modes.
- False negatives could create unjustified reassurance; false positives could waste maintenance
  resources. The UI therefore avoids “safe/unsafe” language.
- The reference period may contain unreported degradation.
- The release artifact is for repeatability, not operational deployment.

