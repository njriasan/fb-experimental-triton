# WSBarrier Op Design

## Motivation

The current subtile barrier placement puts `wait_barrier`/`arrive_barrier`
ops **outside** the SubtiledRegionOp. This works for correctness but
prevents per-subtile pipelining — the consumer can't start processing
subtile 0 until the producer finishes all N subtiles.

To enable per-subtile pipelining, barriers need to live **inside** the
tile body. But `wait_barrier`/`arrive_barrier` reference barrier memdescs
defined outside the SubtiledRegionOp, which violates `IsolatedFromAbove`.

A dedicated `WSBarrier` op solves this by encoding barrier semantics as
attributes rather than SSA value references. The barrier memdesc is
resolved at lowering time from the SubtiledRegionOp's `barriers` operand
list, not from an SSA capture.

## Op Definition

### `ttng.ws_wait_barrier`

```
ttng.ws_wait_barrier {
  barrierIdx = <int>,           // index into SubtiledRegionOp's barriers()
  loweringMask = <array<i32>>   // optional: which tiles emit this op
} : ()
```

Semantics: wait for the barrier at `barriers[barrierIdx]` to reach the
expected phase. The phase is computed automatically from the tile index
and accumCnt during lowering.

### `ttng.ws_arrive_barrier`

```
ttng.ws_arrive_barrier {
  barrierIdx = <int>,           // index into SubtiledRegionOp's barriers()
  count = <int>,                // arrive count (default 1)
  loweringMask = <array<i32>>   // optional: which tiles emit this op
} : ()
```

Semantics: arrive on the barrier at `barriers[barrierIdx]` with the given
count.

### Key properties

- **No SSA operands** — barrier and phase info are encoded as attributes
  referencing the enclosing SubtiledRegionOp's `barriers`/`accumCnts`.
  This avoids `IsolatedFromAbove` violations.

- **Self-contained** — the op carries everything needed for lowering. No
  external `subtile_op_id`, `targetOpIdx`, or barrier annotation matching.

- **Pure within the tile body** — no memory effects (the actual barrier
  op is emitted at lowering time). This means CSE and canonicalization
  won't strip or reorder it (it's a marker, not a real memory op).

## `loweringMask` attribute

The `loweringMask` is an optional `DenseI32ArrayAttr` that controls which
tiles emit the barrier op during `LowerSubtiledRegionPass` expansion.

- **Absent or empty**: emit for ALL tiles (default behavior).
- **`[1, 0]`**: emit only for tile 0, skip tile 1.
- **`[0, 1]`**: emit only for tile 1, skip tile 0.
- **`[1, 1, 0, 1]`**: emit for tiles 0, 1, 3; skip tile 2.

The mask length must equal the number of tiles (from `tileMappings`).
The verifier checks this.

### Use cases

**Wait before first tile only:**
```
tile {
  ttng.ws_wait_barrier { barrierIdx = 0, loweringMask = [1, 0] }
  truncf → local_store
}
```
Lowers to:
```
wait_barrier(bar0, phase=0)  // tile 0
truncf0 → local_store0
                             // tile 1: no wait
truncf1 → local_store1
```

**Arrive after last tile only:**
```
tile {
  truncf → local_store
  ttng.ws_arrive_barrier { barrierIdx = 0, loweringMask = [0, 1] }
}
```
Lowers to:
```
truncf0 → local_store0
                             // tile 0: no arrive
truncf1 → local_store1
arrive_barrier(bar0, 1)      // tile 1
```

**Per-subtile pipelining (all tiles):**
```
tile {
  ttng.ws_wait_barrier { barrierIdx = 0 }   // no mask = all tiles
  truncf → local_store
  ttng.ws_arrive_barrier { barrierIdx = 1 }  // no mask = all tiles
}
```
Lowers to:
```
wait_barrier(bar0, phase=0)
truncf0 → local_store0
arrive_barrier(bar1, 1)
wait_barrier(bar0, phase=1)
truncf1 → local_store1
arrive_barrier(bar1, 1)
```

## Lowering

`LowerSubtiledRegionPass` handles `ws_wait_barrier`/`ws_arrive_barrier`
during tile body expansion:

```python
for tileIdx in range(numTiles):
    for op in tile_body:
        if isinstance(op, WSWaitBarrierOp):
            if is_tile_enabled(op.loweringMask, tileIdx):
                barrier = barriers[op.barrierIdx]
                accumCnt = accumCnts[op.barrierIdx]
                phase = compute_phase(accumCnt, tileIdx, numBuffers)
                emit wait_barrier(barrier, phase)
        elif isinstance(op, WSArriveBarrierOp):
            if is_tile_enabled(op.loweringMask, tileIdx):
                barrier = barriers[op.barrierIdx]
                emit arrive_barrier(barrier, op.count)
        else:
            clone(op, tile_mapping)
```

The phase computation uses the same `(accumCnt + tileIdx) / numBuffers & 1`
formula as the existing `emitBarrierOp`.

## Interaction with existing barriers

The `ws_wait_barrier`/`ws_arrive_barrier` ops replace the current approach
where `doTokenLowering` places real `wait_barrier`/`arrive_barrier` ops
outside the SubtiledRegionOp. Instead:

1. `doTokenLowering` creates `ws_wait_barrier`/`ws_arrive_barrier` ops
   inside the tile body, with appropriate `barrierIdx` and `loweringMask`.
2. The barrier memdescs are added to the SubtiledRegionOp's `barriers()`
   and `accumCnts()` operand lists (same as the old annotation approach).
3. `LowerSubtiledRegionPass` expands them into real barrier ops.

Since `ws_wait_barrier`/`ws_arrive_barrier` have no SSA operands (only
attribute references), they satisfy `IsolatedFromAbove`. They also have
no memory effects, so CSE won't eliminate them (they're markers that
only produce real ops during lowering).

## Migration path

Phase 1 (current): barriers outside SubtiledRegionOp. Works for arbitrary
subtile counts. No pipelining between subtiles.

Phase 2: add `ws_wait_barrier`/`ws_arrive_barrier` ops. `doTokenLowering`
creates them inside the tile body with `loweringMask = all tiles`.
`LowerSubtiledRegionPass` emits per-tile barriers. Functionally equivalent
to Phase 1 but barriers are inside the op.

Phase 3: use `loweringMask` for selective barrier placement. Wait before
first tile only, arrive after last tile only — same as current Phase 1
behavior but with the infrastructure to support per-subtile pipelining.

Phase 4: enable per-subtile pipelining. Remove loweringMask restrictions,
use multi-buffered barriers (`numBuffers=N`), compute per-tile phases.

## Verifier checks

1. `barrierIdx` must be in range of the enclosing SubtiledRegionOp's
   `barriers()` list.
2. If `loweringMask` is present, its length must equal the number of tiles
   (from `tileMappings`).
3. `ws_wait_barrier`/`ws_arrive_barrier` must only appear inside a
   SubtiledRegionOp's tile region (not setup or teardown).
4. For `ws_wait_barrier`, `accumCnts[barrierIdx]` must exist.

## Example: full epilogue subtile pattern

```mlir
ttng.subtiled_region
    inputs(%lhs, %rhs, %smem0, %smem1 : ...)
    barriers(%full_bar, %empty_bar : ...)
    accumCnts(%accum, %c0 : i64, i64)
    tile_mappings = [array<i32: 0, 2>, array<i32: 1, 3>]
  setup {
    ^bb0(%a, %b, %s0, %s1):
      ttng.subtiled_region_yield %a, %b, %s0, %s1 : ...
  } tile(%t0: tensor, %t1: memdesc) {
    ttng.ws_wait_barrier { barrierIdx = 1 }  // wait empty
    %trunc = arith.truncf %t0 : f32 to f16
    ttg.local_store %trunc, %t1
    ttng.ws_arrive_barrier { barrierIdx = 0 } // arrive full
    ttng.subtiled_region_yield
  } teardown {
    ttng.subtiled_region_yield
  }
```

After lowering with 2 tiles:
```mlir
// setup
// tile 0:
wait_barrier(%empty_bar, phase=0)
%trunc0 = arith.truncf %lhs
ttg.local_store %trunc0, %smem0
arrive_barrier(%full_bar, 1)
// tile 1:
wait_barrier(%empty_bar, phase=1)
%trunc1 = arith.truncf %rhs
ttg.local_store %trunc1, %smem1
arrive_barrier(%full_bar, 1)
// teardown
```
