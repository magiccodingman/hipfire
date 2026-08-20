---
title: Qwen continuous batch long-context FWHT3 dispatch
date: 2026-08-20
tags: [qwen35,continuous-batch,fwht3,attention,LDS,gfx1201]
---

Qwen35 single-GPU continuous batching used to allocate a separate Q8 lane-major
KV cache even when the loaded cache was FWHT3. It then dispatched
`attention_q8_0_kv_independent`, whose dynamic LDS is
`(lane_capacity + min(next_pow2(max(lane_capacity, head_dim)), 256) + head_dim) * 4`.
At head_dim 256 and gfx1201's 64 KiB/block limit, 15,872 is the exact largest
lane: `(15872 + 256 + 256) * 4 = 65536`.

Do not remove that kernel's admission limit. For long Q8 lanes and all FWHT3
continuous-batch lanes, use the existing bounded-LDS tile+reduce attention.
FWHT3 needs slot-descriptor support in both its batched K writer and tile kernel
because K and Q8 V have distinct per-position strides/bases. Preserve the Q8
independent kernel only as the short-lane fast path.
