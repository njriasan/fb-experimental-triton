# Subtile Barrier Movement Safety Analysis

This document analyzes what channel properties must hold to safely move
barrier ops relative to the SubtiledRegionOp's tile body operations.

## Current placement

`doTokenLowering` places barriers **outside** the SubtiledRegionOp:

```
wait_barrier(empty)          ← producer_acquire: before SubtiledRegionOp
SubtiledRegionOp {
  setup { ... }
  tile {
    truncf → local_store     ← repeated N times by lowering
  }
  teardown { ... }
}
arrive_barrier(full)         ← producer_commit: after SubtiledRegionOp
```

After `LowerSubtiledRegionPass` expands the tiles, this becomes:

```
wait_barrier(empty)
  [setup ops]
  truncf0 → local_store0    ← tile 0
  truncf1 → local_store1    ← tile 1
  ...
  truncfN → local_storeN    ← tile N-1
  [teardown ops]
arrive_barrier(full)
```

All N stores complete before the arrive signals the consumer.

---

## Case 1: Moving arrive_barrier later (into the tile body)

**Goal:** Signal the consumer after *each* subtile's store completes, not
after all N stores. This enables pipelining — the consumer can start
TMA-copying subtile 0 while the producer is still storing subtile 1.

**Desired placement:**

```
wait_barrier(empty)
  truncf0 → local_store0
  arrive_barrier(full)       ← after tile 0's store
  truncf1 → local_store1
  arrive_barrier(full)       ← after tile 1's store
  ...
```

### Required channel properties

1. **Independent SMEM buffers per subtile.** Each subtile must write to
   a different SMEM buffer. If subtiles share a buffer (buffer.copy=1),
   the consumer can't start reading subtile 0's buffer while subtile 1
   is overwriting it. With `EPILOGUE_SUBTILE=N`, `GenerateSubtiledRegion`
   creates N separate SMEM allocs — this is already satisfied.

2. **Per-subtile barrier or multi-buffered barrier.** A single barrier
   can only track one arrive-wait cycle at a time (phase flips on each
   arrive). For per-subtile arrives to work, either:
   - N separate barriers (one per subtile), or
   - One barrier with `numBuffers=N` and per-subtile index/phase, or
   - One barrier with arrive count=1 and the consumer does N waits
     with incrementing phases.
   Currently we have a single barrier with `numBuffers=1`. Moving the
   arrive inside the tile body requires changing to `numBuffers=N`.

3. **Consumer must wait per-subtile.** The consumer's tile body must also
   have per-subtile waits with matching phases:
   ```
   wait_barrier(full, phase=0) → TMA subtile 0
   wait_barrier(full, phase=1) → TMA subtile 1
   ...
   ```
   The phase computation `(accumCnt + tileIdx) / numBuffers & 1` in
   `emitBarrierOp` already supports this — but the barrier must be
   multi-buffered (`numBuffers=N`).

4. **No cross-subtile data dependency in the producer.** If subtile 1's
   `truncf` depends on subtile 0's result (e.g., through a shared
   accumulator), the arrive after subtile 0 would signal "data ready"
   before the dependency is resolved. In practice, subtiles are
   independent slices of the accumulator — this is satisfied by
   construction from the split tree.

5. **Consumer release must be per-subtile.** The consumer's
   `arrive_barrier(empty)` must also fire per-subtile, not once for all:
   ```
   TMA subtile 0 → arrive_barrier(empty)
   TMA subtile 1 → arrive_barrier(empty)
   ...
   ```
   Otherwise the producer's per-subtile wait(empty) for subtile 1+ would
   deadlock waiting for a release that only fires once.

### Summary for Case 1

Safe to move arrive later (into tile body) if:
- Subtiles use independent SMEM buffers ✓ (by construction)
- Barrier is multi-buffered with numBuffers = numTiles
- Consumer has matching per-subtile waits and releases
- No cross-subtile data dependency in producer ✓ (by construction)

---

## Case 2: Moving wait_barrier earlier (into the tile body)

**Goal:** The producer waits for the consumer to release *each* subtile's
buffer before overwriting it, rather than waiting for all buffers at once.
This enables pipelining in the backward direction — the producer can start
storing subtile 0 while subtile N-1's buffer is still being read.

**Desired placement:**

```
  [setup ops]
  wait_barrier(empty, phase=0)  ← wait for subtile 0's buffer
  truncf0 → local_store0
  wait_barrier(empty, phase=1)  ← wait for subtile 1's buffer
  truncf1 → local_store1
  ...
arrive_barrier(full)
```

### Required channel properties

1. **Independent SMEM buffers per subtile.** Same as Case 1 — each
   subtile's buffer must be independently releasable. ✓

2. **Per-subtile barrier or multi-buffered barrier.** Same as Case 1 —
   the wait phases must distinguish subtiles. Requires `numBuffers=N`.

3. **Consumer releases per-subtile.** The consumer must arrive on the
   empty barrier after finishing each subtile's TMA copy, not once for
   all. Otherwise the per-subtile waits in the producer deadlock.

4. **No ordering constraint between subtile waits.** The producer must
   be able to wait for subtile i's buffer without first having consumed
   subtile i-1's release. With multi-buffered barriers and independent
   phases, this is automatic — each wait checks its own phase slot.

5. **First-iteration semantics.** On the first iteration, no consumer
   has released any buffer. The empty barrier must be pre-arrived for
   ALL subtile phases, not just phase 0. With `numBuffers=N`, the
   pre-arrive needs to cover N slots.

### Summary for Case 2

Safe to move wait earlier (into tile body) if:
- Subtiles use independent SMEM buffers ✓
- Barrier is multi-buffered with numBuffers = numTiles
- Consumer releases per-subtile
- Empty barrier pre-arrived for all N subtile phases
- No cross-subtile ordering constraint in producer ✓

---

## Case 3: Moving arrive past the setup region

**Goal:** Instead of arriving after the entire SubtiledRegionOp (after
all tiles and teardown), arrive after the setup region — before any
tile body runs. This would mean "setup is done, tiles haven't started."

**Desired placement:**

```
wait_barrier(empty)
SubtiledRegionOp {
  setup { ... }
  arrive_barrier(full)       ← after setup, before tiles
  tile {
    truncf → local_store
  }
  teardown { ... }
}
```

After lowering:

```
wait_barrier(empty)
  [setup ops]
  arrive_barrier(full)       ← consumer can start reading... what?
  truncf0 → local_store0
  truncf1 → local_store1
  [teardown ops]
```

### Analysis: This is NEVER safe for the epilogue channel

The arrive signals "data is ready in the SMEM buffer." But the setup
region doesn't write data to the SMEM buffer — it only computes the
subtile values (tmem_load, reshape, split). The actual SMEM writes happen
in the tile body (local_store). Arriving after setup would tell the
consumer to start TMA-copying from SMEM buffers that haven't been
written yet — guaranteed data corruption.

### When COULD this be safe?

Only if the arrive signals something other than "SMEM data ready":
- **"Setup computation done"**: If the consumer needs to wait for setup
  values (e.g., tmem_subslice results) before proceeding, an arrive
  after setup could signal this. But this is a different channel — not
  the SMEM data channel.
- **"Accumulator read complete"**: If the arrive signals that the
  producer finished reading the TMEM accumulator (so the MMA can start
  the next iteration), this could go after tmem_load in setup. But
  this is the TMEM commit backward barrier, not the subtile channel.

### Required channel properties (if somehow applicable)

1. **The channel's data must be produced in the setup region.** The setup
   must write to the channel's SMEM buffer before the arrive. This is
   NOT the case for the epilogue channel — setup only computes tensor
   values, tile body does the SMEM stores.

2. **Consumer must not read SMEM before tile body runs.** If the consumer
   only uses data from the setup (not the SMEM buffer), this could work.
   But the consumer SubtiledRegionOp's tile body does TMA copies from
   SMEM — it reads SMEM data.

3. **No tile body ops that modify the signaled resource.** If the arrive
   claims "resource X is ready" but the tile body subsequently modifies
   resource X, the consumer would see stale/partial data.

### Summary for Case 3

Moving arrive past setup is **unsafe** for the epilogue→TMA store channel
because:
- The signaled resource (SMEM buffer) is written by the tile body, not setup
- The consumer reads the SMEM buffer, which isn't populated until after tiles run
- Arriving early would cause the consumer to TMA-copy uninitialized SMEM data

This would only be safe for a hypothetical channel where the data is fully
produced in the setup region and the consumer doesn't need tile body results.

---

## Implications for the current design

The current "barriers outside SubtiledRegionOp" approach is equivalent to
the most conservative placement: wait before everything, arrive after
everything. This is correct for any number of subtiles but sacrifices
pipelining between subtiles.

To enable per-subtile pipelining (Cases 1 and 2), the key changes would be:
1. Increase `numBuffers` on the subtile token from 1 to N
2. Place barriers inside the tile body (before/after each tile's ops)
3. Handle per-subtile phase computation in the barrier ops
4. Pre-arrive the empty barrier for all N phases on the first iteration

This is the "Option 1" approach discussed earlier. It requires passing
barrier values through the SubtiledRegionOp's inputs (since
IsolatedFromAbove prevents direct reference) and computing per-tile phases
inside the tile body using the tile index argument.
