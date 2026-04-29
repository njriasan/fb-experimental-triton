// RUN: triton-opt %s -split-input-file | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @ws_barrier_basic
  // CHECK: ttng.ws_wait_barrier {barrierIdx = 0 : i32}
  // CHECK: ttng.ws_arrive_barrier {barrierIdx = 1 : i32}
  tt.func @ws_barrier_basic(
      %bar0: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
      %bar1: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
      %accum: i64) {
    ttng.subtiled_region
        barriers(%bar0, %bar1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>,
                                !ttg.memdesc<1xi64, #shared, #smem, mutable>)
        accum_cnts(%accum, %accum : i64, i64)
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        ttng.ws_wait_barrier {barrierIdx = 0 : i32}
        %sum = arith.addi %arg0, %arg0 : i32
        ttng.ws_arrive_barrier {barrierIdx = 1 : i32}
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @ws_barrier_with_mask
  // CHECK: ttng.ws_wait_barrier {barrierIdx = 0 : i32, loweringMask = array<i32: 1, 0>}
  // CHECK: ttng.ws_arrive_barrier {barrierIdx = 0 : i32, loweringMask = array<i32: 0, 1>}
  tt.func @ws_barrier_with_mask(
      %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
      %accum: i64) {
    ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared, #smem, mutable>)
        accum_cnts(%accum : i64)
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        ttng.ws_wait_barrier {barrierIdx = 0 : i32, loweringMask = array<i32: 1, 0>}
        %sum = arith.addi %arg0, %arg0 : i32
        ttng.ws_arrive_barrier {barrierIdx = 0 : i32, loweringMask = array<i32: 0, 1>}
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @ws_barrier_custom_count
  // CHECK: ttng.ws_arrive_barrier {barrierIdx = 0 : i32, count = 4 : i32}
  tt.func @ws_barrier_custom_count(
      %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
      %accum: i64) {
    ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared, #smem, mutable>)
        accum_cnts(%accum : i64)
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %sum = arith.addi %arg0, %arg0 : i32
        ttng.ws_arrive_barrier {barrierIdx = 0 : i32, count = 4 : i32}
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}
