# Decider label wording log

Rules (docs/PLAN.md §4, approved by the project owner on 2026-10-06):
- Versions are scored on the **dev** split only, at batch size 1 (`decider_run.py --split dev --batch-size 1`).
- Wording changes only. No example items and no text from any abstract.
- **Selection rule (fixed before v2 was run):** at most three versions (v1, v2, v3). The version with the highest dev
  accuracy wins. If two are tied (same accuracy to 4 d.p.), the earlier version wins. The test set is then run once,
  with the winning version only.

| Version | Written | Basis | Dev accuracy | Dev macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| v1 | 2026-10-06 | Blind: plain names + general-knowledge one-liners, written before any Decider output | 0.6643 [0.6434, 0.6848] | 0.6667 | Main confusions: Biochem→Medical 118/286, Psych→Medical 67/286, Civil→ME 56/286 |
| v2 | 2026-10-06 | Taxonomy-informed prose from the WOS category docs (Domain → area names) | 0.7013 [0.6808, 0.7213] | 0.7024 | Approved by the owner as "taxonomy-informed". Main confusions: Civil→ME 80, Bio→Med 73, Psych→Med 62. 10.4 items/s vs 13.2 for v1 (longer prompt) |
| v3 | 2026-10-06 | v2 with only the Civil and ME descriptions reworded ("materials" overlap) | 0.7078 [0.6878, 0.7283] | 0.7096 | Civil→ME 80→56. ECE 0.045 |

**Selected: v3** (highest dev accuracy, by the rule above). The v2→v3 gain (+0.65 pt) is inside the CI. Picking the best of 3 on dev carries a small optimistic bias, which is noted in the report's threats to validity. No further versions are allowed.
