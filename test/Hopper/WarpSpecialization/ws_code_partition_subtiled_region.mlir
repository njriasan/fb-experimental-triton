// RUN: triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=1 post-channel-creation=1" | FileCheck %s

// Test: Code partition with SubtiledRegionOps for epilogue subtiling.
// The epilogue SubtiledRegionOp (task 1, local_store) and TMA store
// SubtiledRegionOp (task 2, async_tma_copy) share SMEM buffers.
// The code partition pass should:
//   1. Create an SMEM channel between the two SubtiledRegionOps
//   2. Place token annotations (producer_acquire/commit on task 1,
//      consumer_wait/release on task 2)
//   3. specializeRegion should clone both into separate partitions

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // After code partition, we should get a warp_specialize with:
  //   partition0 (task 1): SubtiledRegionOp with local_store + token annotations
  //   partition1 (task 2): SubtiledRegionOp with TMA copy + token annotations
  //
  // CHECK-LABEL: @subtiled_smem_channel
  // CHECK: ttg.warp_specialize
  //
  // Partition 0 (epilogue): SubtiledRegionOp with producer token annotations
  // CHECK: partition0
  // CHECK:   ttng.subtiled_region
  // CHECK-SAME: token_annotations
  // CHECK-SAME: producer_acquire
  // CHECK-SAME: producer_commit
  // CHECK:     ttg.local_store
  //
  // Partition 1 (store): SubtiledRegionOp with consumer token annotations
  // CHECK: partition1
  // CHECK:   ttng.subtiled_region
  // CHECK-SAME: token_annotations
  // CHECK-SAME: consumer_wait
  // CHECK-SAME: consumer_release
  // CHECK:     ttng.async_tma_copy_local_to_global
  tt.func @subtiled_smem_channel(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %off0: i32, %off1: i32, %off2: i32,
      %lhs: tensor<128x64xf32, #linear>,
      %rhs: tensor<128x64xf32, #linear>) {
    %smem0 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %smem1 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index

    scf.for %iv = %c0 to %c10 step %c1 {
      // Dummy task 0 op (MMA/compute placeholder).
      %dummy = arith.constant {async_task_id = array<i32: 0>} 0 : i32

      // Epilogue SubtiledRegionOp (task 1): truncf → local_store
      ttng.subtiled_region
          inputs(%rhs, %lhs, %smem0, %smem1 :
                 tensor<128x64xf32, #linear>, tensor<128x64xf32, #linear>,
                 !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
                 !ttg.memdesc<128x64xf16, #shared, #smem, mutable>)
          tile_mappings = [array<i32: 0, 2>, array<i32: 1, 3>]
          barrier_annotations = []
          {async_task_id = array<i32: 1>}
        setup {
        ^bb0(%a0: tensor<128x64xf32, #linear>, %a1: tensor<128x64xf32, #linear>,
             %a2: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
             %a3: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>):
          ttng.subtiled_region_yield %a0, %a1, %a2, %a3 :
            tensor<128x64xf32, #linear>, tensor<128x64xf32, #linear>,
            !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
            !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        } tile(%t0: tensor<128x64xf32, #linear>,
               %t1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
               %tidx: i32) {
          %trunc = arith.truncf %t0 {async_task_id = array<i32: 1>}
            : tensor<128x64xf32, #linear> to tensor<128x64xf16, #linear>
          ttg.local_store %trunc, %t1 {async_task_id = array<i32: 1>}
            : tensor<128x64xf16, #linear> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          ttng.subtiled_region_yield
        } teardown {
          ttng.subtiled_region_yield
        }

      // TMA store SubtiledRegionOp (task 2): async_tma_copy
      ttng.subtiled_region
          inputs(%smem0, %smem1, %off1, %off2 :
                 !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
                 !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
                 i32, i32)
          tile_mappings = [array<i32: 0, 2>, array<i32: 1, 3>]
          barrier_annotations = []
          {async_task_id = array<i32: 2>}
        setup {
        ^bb0(%a0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
             %a1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
             %a2: i32, %a3: i32):
          ttng.subtiled_region_yield %a0, %a1, %a2, %a3 :
            !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
            !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
            i32, i32
        } tile(%t0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
               %t1: i32, %tidx: i32) {
          ttng.async_tma_copy_local_to_global %desc[%off0, %t1] %t0
            {async_task_id = array<i32: 2>}
            : !tt.tensordesc<tensor<128x64xf16, #shared>>,
              !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          ttng.subtiled_region_yield
        } teardown {
          ttng.subtiled_region_yield
        }
    } {async_task_id = array<i32: 0, 1, 2>, tt.warp_specialize,
       tt.separate_epilogue_store = true,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["compute", "epilogue", "epilogue_store"]}

    tt.return
  }
}
