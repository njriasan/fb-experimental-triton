// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-lower-subtiled-region | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // Test basic lowering: two tiles, no barriers.
  // CHECK-LABEL: @basic_two_tiles
  tt.func @basic_two_tiles() {
    // Setup ops should be inlined:
    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    // CHECK: %[[C1:.*]] = arith.constant 1 : i32
    // Tile 0 (arg0 = c0):
    // CHECK: arith.index_cast %[[C0]]
    // Tile 1 (arg0 = c1):
    // CHECK: arith.index_cast %[[C1]]
    // CHECK-NOT: ttng.subtiled_region
    ttng.subtiled_region
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %idx = arith.index_cast %arg0 : i32 to index
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }

  // Test lowering with inline arrive_barrier in tile body.
  // The barrier is cloned for each tile.
  // CHECK-LABEL: @arrive_inline
  tt.func @arrive_inline(
      %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
      %row: i32) {
    // Tile 0:
    // CHECK: arith.addi
    // CHECK: ttng.arrive_barrier
    // Tile 1:
    // CHECK: arith.addi
    // CHECK: ttng.arrive_barrier
    // CHECK-NOT: ttng.subtiled_region
    ttng.subtiled_region
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c128 = arith.constant 128 : i32
        ttng.subtiled_region_yield %c0, %c128 : i32, i32
      } tile(%arg0: i32) {
        %off = arith.addi %arg0, %row : i32
        ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }

  // Test lowering with inline wait_barrier in tile body.
  // CHECK-LABEL: @wait_inline
  tt.func @wait_inline(
      %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
      %phase: i32) {
    // Tile 0:
    // CHECK: ttng.wait_barrier
    // CHECK: arith.addi
    // Tile 1:
    // CHECK: ttng.wait_barrier
    // CHECK: arith.addi
    // CHECK-NOT: ttng.subtiled_region
    ttng.subtiled_region
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %res = arith.addi %arg0, %arg0 : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }

  // Test with multiple block args per tile.
  // CHECK-LABEL: @multi_arg_tiles
  tt.func @multi_arg_tiles() {
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
    // CHECK-DAG: %[[C10:.*]] = arith.constant 10 : i32
    // CHECK-DAG: %[[C20:.*]] = arith.constant 20 : i32
    // Tile 0: addi c0, c10
    // CHECK: arith.addi %[[C0]], %[[C10]]
    // Tile 1: addi c1, c20
    // CHECK: arith.addi %[[C1]], %[[C20]]
    // CHECK-NOT: ttng.subtiled_region
    ttng.subtiled_region
        tile_mappings = [array<i32: 0, 2>, array<i32: 1, 3>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        %c10 = arith.constant 10 : i32
        %c20 = arith.constant 20 : i32
        ttng.subtiled_region_yield %c0, %c1, %c10, %c20 : i32, i32, i32, i32
      } tile(%a: i32, %b: i32) {
        %sum = arith.addi %a, %b : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }

  // Test with both wait and arrive inline in tile body.
  // CHECK-LABEL: @wait_and_arrive_inline
  tt.func @wait_and_arrive_inline(
      %bar_wait: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
      %bar_arrive: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
      %phase: i32) {
    // Tile 0:
    // CHECK: ttng.wait_barrier
    // CHECK: arith.muli
    // CHECK: ttng.arrive_barrier
    // Tile 1:
    // CHECK: ttng.wait_barrier
    // CHECK: arith.muli
    // CHECK: ttng.arrive_barrier
    // CHECK-NOT: ttng.subtiled_region
    ttng.subtiled_region
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = []
      setup {
        %c3 = arith.constant 3 : i32
        %c5 = arith.constant 5 : i32
        ttng.subtiled_region_yield %c3, %c5 : i32, i32
      } tile(%arg0: i32) {
        ttng.wait_barrier %bar_wait, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %res = arith.muli %arg0, %arg0 : i32
        ttng.arrive_barrier %bar_arrive, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }

  // Test with a single tile (degenerate case).
  // CHECK-LABEL: @single_tile
  tt.func @single_tile(
      %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
      %phase: i32) {
    // CHECK: ttng.wait_barrier
    // CHECK-NEXT: arith.addi
    // CHECK-NEXT: ttng.arrive_barrier
    // CHECK-NOT: ttng.subtiled_region
    ttng.subtiled_region
        tile_mappings = [array<i32: 0>]
        barrier_annotations = []
      setup {
        %c42 = arith.constant 42 : i32
        ttng.subtiled_region_yield %c42 : i32
      } tile(%arg0: i32) {
        ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %res = arith.addi %arg0, %arg0 : i32
        ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }

  // Test capturing values from the outer scope.
  // CHECK-LABEL: @capture_outer_value
  // CHECK-SAME: %[[OUTER:arg0]]: i32
  tt.func @capture_outer_value(%outer: i32) {
    // CHECK: arith.constant 0 : i32
    // Tile 0: addi c0, %outer
    // CHECK: arith.addi %{{.*}}, %[[OUTER]]
    // Tile 1: addi c1, %outer
    // CHECK: arith.addi %{{.*}}, %[[OUTER]]
    // CHECK-NOT: ttng.subtiled_region
    ttng.subtiled_region
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %res = arith.addi %arg0, %outer : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }

  // Test no barriers, no phases.
  // CHECK-LABEL: @no_barriers
  tt.func @no_barriers() {
    // CHECK: arith.constant 0 : i32
    // CHECK: arith.constant 1 : i32
    // CHECK: arith.index_cast
    // CHECK: arith.index_cast
    // CHECK-NOT: ttng.subtiled_region
    ttng.subtiled_region
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %idx = arith.index_cast %arg0 : i32 to index
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }

  // Test teardown region with results.
  // CHECK-LABEL: @teardown_with_results
  tt.func @teardown_with_results() -> i32 {
    // CHECK: arith.constant 0 : i32
    // CHECK: arith.constant 1 : i32
    // Tiles:
    // CHECK: arith.addi
    // CHECK: arith.addi
    // Teardown:
    // CHECK: %[[RESULT:.*]] = arith.constant 42 : i32
    // CHECK: tt.return %[[RESULT]]
    // CHECK-NOT: ttng.subtiled_region
    %result = ttng.subtiled_region
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %res = arith.addi %arg0, %arg0 : i32
        ttng.subtiled_region_yield
      } teardown {
        %c42 = arith.constant 42 : i32
        ttng.subtiled_region_yield %c42 : i32
      } -> (i32)
    tt.return %result : i32
  }

  // Test tile index argument: the trailing i32 arg is substituted with
  // the tile index constant (0, 1, ...) during lowering.
  // CHECK-LABEL: @tile_index_arg
  tt.func @tile_index_arg() {
    // Setup:
    // CHECK: %[[C10:.*]] = arith.constant 10 : i32
    // CHECK: %[[C20:.*]] = arith.constant 20 : i32
    // Tile 0: arg0 = c10, tileIdx = 0
    // CHECK: %[[T0:.*]] = arith.constant 0 : i32
    // CHECK: arith.addi %[[C10]], %[[T0]]
    // Tile 1: arg0 = c20, tileIdx = 1
    // CHECK: %[[T1:.*]] = arith.constant 1 : i32
    // CHECK: arith.addi %[[C20]], %[[T1]]
    // CHECK-NOT: ttng.subtiled_region
    ttng.subtiled_region
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = []
      setup {
        %c10 = arith.constant 10 : i32
        %c20 = arith.constant 20 : i32
        ttng.subtiled_region_yield %c10, %c20 : i32, i32
      } tile(%arg0: i32, %tileIdx: i32) {
        %sum = arith.addi %arg0, %tileIdx : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}
