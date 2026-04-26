// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-test-generate-subtiled-region | FileCheck %s

// Test: Multi-task epilogue subtiling with barrier annotations.
// When the local_store (task 1) and TMA copy (task 2) have different
// async_task_ids, the generation pass creates two SubtiledRegionOps:
//   1. Epilogue (task 1): truncf → local_store per tile
//   2. TMA store (task 2): async_tma_copy_local_to_global per tile
// Both SubtiledRegionOps should have the correct structure for
// per-tile barrier annotation during code partition.

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3d = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3d_perm = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked2d = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @epilogue_and_tma_store_subtile
  //
  // First SubtiledRegionOp: epilogue (task 1)
  // Setup yields leaf values and SMEM buffers.
  // Tile body has truncf → local_store.
  // CHECK: ttng.subtiled_region
  // CHECK-SAME: {async_task_id = array<i32: 1>}
  // CHECK:   setup
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   tile
  // CHECK:     arith.truncf
  // CHECK:     ttg.local_store
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   teardown
  // CHECK:     ttng.subtiled_region_yield
  //
  // Second SubtiledRegionOp: TMA store (task 2)
  // Setup yields SMEM buffers and offsets.
  // Tile body has async_tma_copy_local_to_global.
  // CHECK: ttng.subtiled_region
  // CHECK-SAME: {async_task_id = array<i32: 2>}
  // CHECK:   setup
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   tile
  // CHECK:     ttng.async_tma_copy_local_to_global
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   teardown
  // CHECK:     ttng.subtiled_region_yield
  tt.func @epilogue_and_tma_store_subtile(
      %tmem_buf: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token,
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %off0: i32, %off1: i32, %off2: i32) {
    %smem0 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %smem1 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    %loaded:2 = ttng.tmem_load %tmem_buf[%acc_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %reshaped = tt.reshape %loaded#0 : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked3d>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3d> -> tensor<128x64x2xf32, #blocked3d_perm>
    %lhs, %rhs = tt.split %transposed : tensor<128x64x2xf32, #blocked3d_perm> -> tensor<128x64xf32, #blocked2d>

    // Epilogue: task 1
    %trunc0 = arith.truncf %lhs {async_task_id = array<i32: 1>} : tensor<128x64xf32, #blocked2d> to tensor<128x64xf16, #blocked2d>
    ttg.local_store %trunc0, %smem0 {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked2d> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    // TMA store: task 2
    ttng.async_tma_copy_local_to_global %desc[%off0, %off1] %smem0 {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    // Epilogue: task 1
    %trunc1 = arith.truncf %rhs {async_task_id = array<i32: 1>} : tensor<128x64xf32, #blocked2d> to tensor<128x64xf16, #blocked2d>
    ttg.local_store %trunc1, %smem1 {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked2d> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    // TMA store: task 2
    ttng.async_tma_copy_local_to_global %desc[%off0, %off2] %smem1 {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    tt.return
  }
}

// -----

// Test: Same pattern inside scf.for loop body.
// The SubtiledRegionOps should be generated inside the loop with
// proper inputs (no implicit captures of loop-body values).

#blocked_b = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3d_b = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3d_perm_b = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked2d_b = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem_b = #ttg.shared_memory
#tmem_b = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @epilogue_and_tma_store_in_loop
  // CHECK: scf.for
  //
  // Epilogue SubtiledRegionOp with inputs and local_store in tile body.
  // CHECK:   ttng.subtiled_region inputs(
  // CHECK-SAME: {async_task_id = array<i32: 1>}
  // CHECK:     setup
  // CHECK:       ttng.subtiled_region_yield
  // CHECK:     tile
  // CHECK:       arith.truncf
  // CHECK:       ttg.local_store
  // CHECK:       ttng.subtiled_region_yield
  //
  // TMA store SubtiledRegionOp with inputs including shared operand.
  // CHECK:   ttng.subtiled_region inputs(
  // CHECK-SAME: {async_task_id = array<i32: 2>}
  // CHECK:     setup
  // CHECK:       ttng.subtiled_region_yield
  // CHECK:     tile
  // CHECK:       ttng.async_tma_copy_local_to_global
  // CHECK:       ttng.subtiled_region_yield
  tt.func @epilogue_and_tma_store_in_loop(
      %tmem_buf: !ttg.memdesc<128x128xf32, #tmem_b, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token,
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared_b>>,
      %smem0: !ttg.memdesc<128x64xf16, #shared_b, #smem_b, mutable>,
      %smem1: !ttg.memdesc<128x64xf16, #shared_b, #smem_b, mutable>,
      %off0: i32, %off1: i32, %off2: i32,
      %lb: i32, %ub: i32, %step: i32) {

    scf.for %iv = %lb to %ub step %step  : i32 {
      %loaded:2 = ttng.tmem_load %tmem_buf[%acc_tok] : !ttg.memdesc<128x128xf32, #tmem_b, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked_b>
      %reshaped = tt.reshape %loaded#0 : tensor<128x128xf32, #blocked_b> -> tensor<128x2x64xf32, #blocked3d_b>
      %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3d_b> -> tensor<128x64x2xf32, #blocked3d_perm_b>
      %lhs, %rhs = tt.split %transposed : tensor<128x64x2xf32, #blocked3d_perm_b> -> tensor<128x64xf32, #blocked2d_b>

      %trunc0 = arith.truncf %lhs {async_task_id = array<i32: 1>} : tensor<128x64xf32, #blocked2d_b> to tensor<128x64xf16, #blocked2d_b>
      ttg.local_store %trunc0, %smem0 {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked2d_b> -> !ttg.memdesc<128x64xf16, #shared_b, #smem_b, mutable>
      ttng.async_tma_copy_local_to_global %desc[%off0, %off1] %smem0 {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared_b>>, !ttg.memdesc<128x64xf16, #shared_b, #smem_b, mutable>

      %trunc1 = arith.truncf %rhs {async_task_id = array<i32: 1>} : tensor<128x64xf32, #blocked2d_b> to tensor<128x64xf16, #blocked2d_b>
      ttg.local_store %trunc1, %smem1 {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked2d_b> -> !ttg.memdesc<128x64xf16, #shared_b, #smem_b, mutable>
      ttng.async_tma_copy_local_to_global %desc[%off0, %off2] %smem1 {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared_b>>, !ttg.memdesc<128x64xf16, #shared_b, #smem_b, mutable>
    }

    tt.return
  }
}
