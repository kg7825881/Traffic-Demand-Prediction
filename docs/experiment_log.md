# Experiment Log

## Baseline

Score:
87.33782

Summary:
Original LightGBM pipeline.

---

## Exp1 – Geohash Hierarchy

Score:
91.15397

Change:
Added geohash, geo4, geo5, geo6.

Result:
SUCCESS

Improvement:
+3.81615

---

## Exp2 – Frequency Encoding

Score:
91.18758

Change:
Added geohash_freq, geo4_freq, geo5_freq, geo6_freq.

Result:
SUCCESS

Improvement:
+0.03361

---

## Exp3 – Location × Time Interactions

Score:
90.70221

Result:
FAILED

Reason:
Strong CV importance but poorer leaderboard generalization.

---

## Exp4 – Day48 Historical Lookups

Score:
90.81696

Result:
FAILED

Reason:
CV improved significantly but leaderboard declined.

---

## Exp5 – Spatial Density Features

Score:
91.19004

Result:
SUCCESS

Improvement:
+0.00246

Current Best.
