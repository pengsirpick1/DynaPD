# DMMPv3 fixed attack evaluation

- transfer mixed: 0
- adaptive source run: results\dmmpv3_legacydirect_fused_fullcw_seed0_20260719_134324
- fresh target run: results\dmmpv3_legacydirect_fused_fullcw_seed0_20260719_134324
- source profiles: clean only
- target profile: test_000
- budget / keep ratio: 0.3000 / 1.0000
- adaptive source budget / keep ratio: 0.3000 / 1.0000
- raw real-packet retention: 1.000000
- visible dummy overhead: 0.289779
- hard gate: clean >= 0.8500, defended <= 0.4000

| protocol | attacker | source users | overlap | clean acc | defended acc | gate |
|---|---:|---:|---:|---:|---:|---:|
| fixed | DF | 0 | 0.0000 | 0.973968 | 0.184495 | 1 |
| fixed | RF | 0 | 0.0000 | 0.978607 | 0.448252 | 0 |