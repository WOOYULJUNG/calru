# H-C local-grid CA comparison

The rank sum is a diagnostic ordering, not an automatic scientific decision. Only cells with all model seeds blank-stable and all full analyses complete are ranked.

| Cell | Stable | Task NMSE | Blank error | H=2048 memory | Tangent gain | Radial gain | H=4096 recovery | Uniform flow | Rank sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a0p05_bm0p10 | 3/3 | 0.00109034 | 0.20777 | 0.197737 | 0.999458 | 0.806362 | 0.0223705 | 0.000483009 | 13 |
| a0p05_bm0p05 | 3/3 | 0.00375354 | 0.30146 | 0.311429 | 0.99941 | 0.774966 | 0.00400415 | 0.000442688 | 17 |
| a0p035_bm0p10 | 3/3 | 0.000887463 | 0.226846 | 0.227851 | 0.99923 | 0.840993 | 0.0524534 | 0.00102955 | 18 |
| a0p035_bm0p15 | 3/3 | 0.00261967 | 0.336606 | 0.387556 | 0.994805 | 0.94714 | 0.00460124 | 0.00777405 | 26 |
| a0p065_bm0p05 | 3/3 | 0.00343934 | 0.367823 | 0.300426 | 0.992806 | 0.808463 | 0.0129007 | 0.00686207 | 29 |
| a0p065_bm0p15 | 3/3 | 0.0130879 | 0.354542 | 0.363532 | 0.993602 | 0.941777 | 0.00347906 | 0.022806 | 31 |
| a0p065_bm0p10 | 3/3 | 0.0122304 | 0.34252 | 0.36546 | 0.983432 | 0.938315 | 0.0104454 | 0.0127593 | 34 |
| a0p035_bm0p05 | 2/3 | 0.000729548 | 0.331957 | NA | NA | NA | NA | NA | NA |
| a0p05_bm0p15 | 2/3 | 0.00696124 | 0.444143 | NA | NA | NA | NA | NA | NA |

Selection should prioritize: all-seed stability, near-neutral tangent dynamics, finite normal recovery, bounded finite-time memory drift, and acceptable task error. Effective basin count is reported descriptively and is not treated as a monotone score.
