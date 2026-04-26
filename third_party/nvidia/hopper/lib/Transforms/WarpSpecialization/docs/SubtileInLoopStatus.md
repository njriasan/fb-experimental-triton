# Subtile Operator for In-Loop Splits — Status

## Goal

Enable the SubtiledRegionOp to work for epilogue splits inside persistent
matmul loop bodies (not just flat function bodies), reducing register pressure
by processing accumulator subtiles sequentially.

## What Works

1. **SubtiledRegionOp generation inside loops** — The pass walker now finds
   SplitOps at any nesting depth (not just function top-level blocks).

2. **Two SubtiledRegionOps generated** — One for the epilogue (truncf →
   convert_layout → local_store) and one for the TMA store
   (async_tma_copy_local_to_global → token_wait).

3. **Explicit `inputs` operand** — All outer-scope values are threaded through
   `inputs` as setup block arguments. No implicit captures of loop-body values
   that would break when `specializeRegion` clones the enclosing block.

4. **Shared operands** — Values used by all tiles identically (e.g.
   `offs_am_c`) are detected and passed as shared tile args with a single
   yield slot mapped by all tiles.

5. **WSCodePartition transparency** — `isAinNestedRegion` and
   `getSameLevelOp` use `getEffectiveParentOp` to skip SubtiledRegionOp when
   walking parent chains. The `TCGen5MMAOp` assertion is relaxed for ops
   inside SubtiledRegionOp.

6. **Channel discovery** — `createChannelPost` and `getSrcOp` look through
   SubtiledRegionOp inputs to find local_store producers and TMA copy
   consumers inside tile regions.

7. **LowerSubtiledRegion** — Maps setup block args to `op.getInputs()` before
   cloning setup ops to flat IR.

8. **PushSharedSetupToTile** — Maps setup block args back to
   `op.getInputs()` when replacing shared tile args with external values.

9. **Full compilation pipeline runs** — No crashes, no verification errors.
   The kernel compiles to PTX and runs on GPU.

10. **All existing LIT tests pass** (generate_subtiled_region_tmem_split,
    ntile, multi_task, dp1, ops, push_shared_setup_to_tile).

## Remaining Issue

**Runtime deadlock** — The kernel hangs on GPU. The SMEM epilogue-to-store
barrier (mbarrier) is created and placed correctly as `wait_barrier` /
`arrive_barrier` ops around the SubtiledRegionOps. The `doTokenLowering` pass
converts `nvws.producer_acquire/commit` → `wait_barrier/arrive_barrier`
correctly. However, the runtime synchronization fails — likely a phase
initialization or index computation mismatch.

The barrier pattern after `doTokenLowering`:

```
// Partition 0 (epilogue, task 1):
wait_barrier %mbar, %phase   // producer wait (acquire)
ttng.subtiled_region { ... local_store ... }
arrive_barrier %mbar, 1      // producer commit

// Partition 1 (store, task 2):
wait_barrier %mbar, %phase   // consumer wait
ttng.subtiled_region { ... async_tma_copy ... }
arrive_barrier %mbar, 1      // consumer release
```

## Files Changed

| File | Change |
|------|--------|
| `GenerateSubtiledRegion.cpp` | Walk fix, inputs collection, shared operands, TMA store subtile, chain exclusion |
| `TritonNvidiaGPUOps.td` | Added `inputs` operand to SubtiledRegionOp |
| `Ops.cpp` | Parser/printer/verifier for `inputs` and setup block args |
| `LowerSubtiledRegion.cpp` | Map setup block args to inputs before cloning |
| `PushSharedSetupToTile.cpp` | Map setup block args back to inputs for shared arg replacement |
| `WSCodePartition.cpp` | `getEffectiveParentOp`, nesting transparency, relaxed assertion, async_task_id propagation |
| `CodePartitionUtility.cpp` | `createChannelPost` and `getSrcOp` look through SubtiledRegionOp |
| `generate_subtiled_region_dp1.mlir` | LIT test for flat and in-loop cases |
| `test_tutorial09_warp_specialization.py` | Updated test infrastructure |

## Next Steps

Debug the barrier deadlock by comparing the mbarrier phase/index IR against
the known-working non-subtiled case. The barrier is structurally correct
(wait before SubtiledRegionOp, arrive after) but the runtime synchronization
fails. Possible causes:

1. Phase initialization mismatch between producer and consumer mbarriers
2. Buffer index computation using wrong accumulation counter
3. Single-buffered token (`numBuffers = 1`) not handling the persistent loop
   phase flipping correctly
4. Mbarrier arrive count mismatch (producer arrives once, consumer expects
   different count)

The barrier visualization skill can be used to audit the arrive/wait pattern
in the final TTGIR.
