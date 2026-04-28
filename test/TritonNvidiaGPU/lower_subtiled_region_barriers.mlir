// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-lower-subtiled-region | FileCheck %s

// Test: Lowering a SubtiledRegionOp with inline barrier ops in the tile body.
// Barrier ops are injected by doTokenLowering before this pass runs.
// The lowering pass just clones them for each tile.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @lower_epilogue_with_inline_barriers
  //
  // The tile body has wait_barrier before truncf and arrive_barrier after
  // local_store. With 2 tiles, each gets its own copy:
  //
  // CHECK: ttng.wait_barrier
  // CHECK: arith.truncf
  // CHECK: ttg.local_store
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK: arith.truncf
  // CHECK: ttg.local_store
  // CHECK: ttng.arrive_barrier
  // CHECK-NOT: ttng.subtiled_region
  tt.func @lower_epilogue_with_inline_barriers(
      %lhs: tensor<128x64xf32, #linear>,
      %rhs: tensor<128x64xf32, #linear>,
      %smem0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %smem1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %bar: !ttg.memdesc<1xi64, #shared1, #smem, mutable>,
      %phase: i32) {
    ttng.subtiled_region
        inputs(%lhs, %rhs, %smem0, %smem1 : tensor<128x64xf32, #linear>, tensor<128x64xf32, #linear>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>)
        tile_mappings = [array<i32: 0, 2>, array<i32: 1, 3>]
        barrier_annotations = []
      setup {
      ^bb0(%arg0: tensor<128x64xf32, #linear>, %arg1: tensor<128x64xf32, #linear>, %arg2: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %arg3: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>):
        ttng.subtiled_region_yield %arg1, %arg0, %arg3, %arg2 : tensor<128x64xf32, #linear>, tensor<128x64xf32, #linear>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      } tile(%t0: tensor<128x64xf32, #linear>, %t1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %tidx: i32) {
        ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %trunc = arith.truncf %t0 : tensor<128x64xf32, #linear> to tensor<128x64xf16, #linear>
        ttg.local_store %trunc, %t1 : tensor<128x64xf16, #linear> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Test: TMA store SubtiledRegionOp with inline barrier ops.

#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1_b = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem_b = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @lower_tma_store_with_inline_barriers
  //
  // 2 tiles, each with wait_barrier + TMA copy + arrive_barrier:
  // CHECK: ttng.wait_barrier
  // CHECK: ttng.async_tma_copy_local_to_global
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK: ttng.async_tma_copy_local_to_global
  // CHECK: ttng.arrive_barrier
  // CHECK-NOT: ttng.subtiled_region
  tt.func @lower_tma_store_with_inline_barriers(
      %smem0: !ttg.memdesc<128x64xf16, #shared_b, #smem_b, mutable>,
      %smem1: !ttg.memdesc<128x64xf16, #shared_b, #smem_b, mutable>,
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared_b>>,
      %off0: i32, %off1: i32,
      %bar: !ttg.memdesc<1xi64, #shared1_b, #smem_b, mutable>,
      %phase: i32) {
    ttng.subtiled_region
        inputs(%smem0, %smem1, %off0, %off1 : !ttg.memdesc<128x64xf16, #shared_b, #smem_b, mutable>, !ttg.memdesc<128x64xf16, #shared_b, #smem_b, mutable>, i32, i32)
        tile_mappings = [array<i32: 0, 2>, array<i32: 1, 3>]
        barrier_annotations = []
      setup {
      ^bb0(%arg0: !ttg.memdesc<128x64xf16, #shared_b, #smem_b, mutable>, %arg1: !ttg.memdesc<128x64xf16, #shared_b, #smem_b, mutable>, %arg2: i32, %arg3: i32):
        ttng.subtiled_region_yield %arg0, %arg1, %arg2, %arg3 : !ttg.memdesc<128x64xf16, #shared_b, #smem_b, mutable>, !ttg.memdesc<128x64xf16, #shared_b, #smem_b, mutable>, i32, i32
      } tile(%t0: !ttg.memdesc<128x64xf16, #shared_b, #smem_b, mutable>, %t1: i32, %tidx: i32) {
        ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared1_b, #smem_b, mutable>
        ttng.async_tma_copy_local_to_global %desc[%off0, %t1] %t0 : !tt.tensordesc<tensor<128x64xf16, #shared_b>>, !ttg.memdesc<128x64xf16, #shared_b, #smem_b, mutable>
        ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1_b, #smem_b, mutable>
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}
