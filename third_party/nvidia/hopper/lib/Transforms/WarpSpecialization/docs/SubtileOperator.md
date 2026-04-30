# Subtile Operator — Design & Implementation Overview

## Motivation

In warp-specialized GEMM epilogues with `EPILOGUE_SUBTILE > 1`, the
accumulator is split into N subtiles (e.g., 128×256 → 2×128×128). Each
subtile flows through the same computation (truncf, convert, store) but with
different data and offsets. The **subtile operator** (`ttng.subtiled_region`)
captures this structure so that per-tile barrier placement, memory planning,
and code generation can reason about the repetition rather than seeing N
copies of inlined code.

## Architecture

### Op Definition

`SubtiledRegionOp` (`ttng.subtiled_region`) is `IsolatedFromAbove`: all
values from the enclosing scope must be passed explicitly via `inputs`.
The setup region receives these as block arguments.

It has three regions:

- **setup**: Receives `inputs` as block arguments. Computes subtile values
  (tmem_load → reshape → trans → split). Terminated by
  `subtiled_region_yield` whose values are indexed by tile mappings.
- **tile**: Per-tile body, replicated during lowering. Block arguments are
  substituted from setup outputs via `tileMappings`. An optional trailing
  i32 argument receives the tile index (0, 1, …).
- **teardown**: Runs once after all tiles. Its yield values become the op's
  results.

Key operands:
- `inputs: Variadic<AnyType>` — all values captured from the enclosing scope
- `tokenValues: Variadic<AnyType>` — NVWS token values for token annotations

Key attributes:
- `tileMappings: ArrayAttr` — one `DenseI32ArrayAttr` per tile mapping tile
  block args to setup yield indices
- `tokenAnnotations: ArrayAttr` — NVWS token-layer annotations, resolved to
  inline barrier ops during token lowering

Key methods:
- `addInputToTileBody(Value)` — threads a value through inputs → setup yield
  → tile mappings → tile block argument. Used by doTokenLowering and
  lowerTokenAnnotations to make barrier values accessible inside the tile
  body while respecting IsolatedFromAbove.

Defined in `include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td`.

### Passes

#### 1. GenerateSubtiledRegion
**File:** `lib/Dialect/TritonNvidiaGPU/Transforms/GenerateSubtiledRegion.cpp`
**Pass:** `triton-nvidia-gpu-test-generate-subtiled-region`

Finds `tmem_load → reshape → trans{[0,2,1]} → split` patterns and wraps the
per-tile chains into `SubtiledRegionOp`s.

Key capabilities:
- **2-tile and N-tile** (4, 8, …) via nested split tree walking
  (`collectSplitTreeLeaves`). Both paths share the same N-tile build
  functions (`buildSingleSubtiledRegionN`, `buildMultiTaskSubtiledRegionsN`)
  — the 2-tile path is a thin wrapper that converts inputs.
- **Auxiliary collection** (`collectPerTileChain`): always recursively
  captures ops needed by the chain but not depending on the split result
  (e.g., `descriptor_load`, `arith.addi` for address computation). For
  N-tile nested splits, inner setup ops are excluded via `excludeOps`.
- **Identity insertion** for asymmetric chains (e.g., one tile has an extra
  `arith.addi` for column offset). The "canonical identity set" approach in
  `checkStructuralEquivalenceN` compares the template against the shortest
  chain first to discover all identity ops, then re-compares other chains
  with `forcedIdentityOps` for consistent identity counts.
- **Multi-task segmentation** for chains crossing async task boundaries.
  Each segment becomes a separate `SubtiledRegionOp` with SMEM transitions
  (implicit buffer via `local_store`/`local_load`). Allocs are assumed to
  be pre-hoisted by the memory planner.
- **Multi-chain support** (addmm): when task IDs are non-contiguous
  (e.g., task 2 → 3 → 2 → 1), segments are merged by task ID and
  topologically sorted by data dependency, producing contiguous regions
  (e.g., task 3 → 2 → 1).

Structural equivalence (`checkStructuralEquivalence`) compares per-tile
chains pairwise, recording differing operands and identity-compatible ops.
`checkStructuralEquivalenceN` wraps this for N chains with consistent
identity handling.

#### 2. OptimizeTMemLayouts (+ PushSharedSetupToTile)
**File:** `lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp`
**Pass:** `triton-nvidia-optimize-tmem-layouts`

This pass serves dual purposes:

1. **TMem layout optimization** (pattern-based): Converts
   `tmem_load → reshape → trans → split` chains into
   `tmem_subslice → tmem_load` pairs, eliminating reshape/trans overhead.
   Also handles `tmem_store + join` patterns and layout selection for
   vectorization.

2. **SubtiledRegionOp setup push** (imperative, after patterns fire): Walks
   all `SubtiledRegionOp`s and calls `pushSubtiledRegionSetupToTile()`, which
   runs three transformations:
   - `addSubsliceRangeToSetup` — extracts per-tile N offsets from
     `tmem_subslice` ops as i32 tile args
   - `pushTmemLoadsToTile` — moves per-tile `tmem_load` chains from setup
     into tile body, interleaving loads with compute
   - `pushSharedSetupToTile` — sinks "shared" tile arguments (uniform across
     tiles) into the tile body. Only constants defined inside setup can be
     pushed; pass-through input args stay as tile args (IsolatedFromAbove
     prevents referencing the original input from inside the tile body)

The push logic lives in `PushSharedSetupToTile.cpp` and is exposed via the
`pushSubtiledRegionSetupToTile()` entry point declared in `Dialect.h`.

#### 3. LowerSubtiledRegion
**File:** `lib/Dialect/TritonNvidiaGPU/Transforms/LowerSubtiledRegion.cpp`
**Pass:** `triton-nvidia-gpu-lower-subtiled-region`

Expands each `SubtiledRegionOp` into flat IR:
1. Inlines setup ops
2. Replicates tile body N times with value substitution from tile mappings
3. Inlines teardown ops

Also exported as a public function `lowerSubtiledRegion(SubtiledRegionOp)`
for use by other passes (e.g., WSCodePartition for multi-task fallback).

### Pipeline Integration

The subtile pipeline spans two compilation phases: the WS mega-pass generates
and annotates the SubtiledRegionOps, then the main TTGIR pipeline optimizes
and lowers them.

**Inside `NVGPUWarpSpecialization` pass** (`WarpSpecialization.cpp`):

```
doTaskIdPropagate
doBufferAllocation
doHoistLoopInvariantTMEMStore
doMemoryPlanner
doGenerateSubtiledRegion          ← only runs GenerateSubtiledRegion pass
doAnnotateTMAStoreWaits
doValidateTMAStoreAnnotations
doCodePartitionPost               ← adds token annotations on SubtiledRegionOps;
                                    multi-task SubtiledRegionOps lowered here
doTokenLowering                   ← resolves tokens → inline barrier ops
scheduleLoops                       (SubtiledRegionOps survive with inline barriers)
```

**In the main TTGIR pipeline** (`compiler.py`), after the WS pass:

```
...
add_optimize_tmem_layouts         ← pattern rewrites (split → tmem_subslice)
                                    + pushSubtiledRegionSetupToTile()
add_lower_subtiled_region         ← expands tile bodies
add_tma_lowering
...
```

This separation is critical: `doGenerateSubtiledRegion` only creates the
SubtiledRegionOps (no tmem optimization, no setup push). The SubtiledRegionOps
survive through the WS pass where token annotations are resolved to inline
barrier ops. Only after the WS pass completes does `add_optimize_tmem_layouts`
transform the setup chains (both inside SubtiledRegionOps and bare splits
elsewhere), and `add_lower_subtiled_region` expands the tile bodies.

This avoids the earlier problem where `OptimizeTMemLayouts` ran inside
`doGenerateSubtiledRegion` and transformed bare (non-SubtiledRegionOp) splits
into `tmem_subslice` ops lacking `async_task_id`, crashing `createChannelPost`.

Multi-task SubtiledRegionOps (tile body spanning multiple tasks) are still
lowered as a fallback inside `doCodePartitionPost` before `specializeRegion`.

### Compiler Option

- Kernel kwarg: `generate_subtiled_region=True`
- Knob: `triton.knobs.nvidia.generate_subtiled_region = True`
- Env var: `TRITON_GENERATE_SUBTILED_REGION=1`
- Autotuning config option: `generate_subtiled_region`

Default: `False`.

### Token Annotations

`TokenAnnotationAttr` describes NVWS token-layer synchronization. During
`doTokenLowering`, token annotations are resolved to inline
`WaitBarrierOp`/`ArriveBarrierOp` ops placed inside the tile body. The
barrier memdesc and phase values are threaded through `addInputToTileBody`
so they're accessible inside the tile body despite `IsolatedFromAbove`.

### Test Coverage

| Test file | Coverage |
|-----------|----------|
| `test/TritonNvidiaGPU/lower_subtiled_region.mlir` | Basic lowering, inline barriers, tile index, teardown results |
| `test/TritonNvidiaGPU/lower_subtiled_region_barriers.mlir` | Lowering with inline wait/arrive barrier ops in tile body |
| `test/TritonNvidiaGPU/generate_subtiled_region_dp1.mlir` | DP=1 epilogue subtiling |
| `test/TritonNvidiaGPU/generate_subtiled_region_multi_task.mlir` | Multi-task, identity, addmm patterns |
| `test/TritonNvidiaGPU/generate_subtiled_region_ntile.mlir` | 4-tile, 8-tile nested splits |
| `test/TritonNvidiaGPU/generate_subtiled_region_tmem_split.mlir` | tmem_subslice + push-to-tile optimization |
| `test/TritonNvidiaGPU/push_shared_setup_to_tile.mlir` | Setup-to-tile push transformations |
| `test/TritonNvidiaGPU/ops.mlir` | Round-trip parse/print |
| `test/TritonNvidiaGPU/invalid.mlir` | Verifier error cases |
| `test/Hopper/WarpSpecialization/ws_token_lowering_subtiled_region.mlir` | Token lowering with SubtiledRegionOps inside warp_specialize |
| `test/Hopper/WarpSpecialization/ws_code_partition_subtiled_region.mlir` | Code partition with SMEM channels between SubtiledRegionOps |
| `python/test/unit/language/test_tutorial09_warp_specialization.py` | Blackwell GEMM e2e (parametrized with `generate_subtiled_region`) |
| `python/test/unit/language/test_autows_addmm.py` | Addmm e2e (parametrized with `generate_subtiled_region`) |
| `test_subtile_gemm.py` | Standalone addmm + subtile e2e |
