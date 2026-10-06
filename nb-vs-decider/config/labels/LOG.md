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
| v2 | 2026-10-06 | Taxonomy-informed prose from the WOS category docs (Domain → area names) | _pending_ | | Approved by the owner as "taxonomy-informed" |
