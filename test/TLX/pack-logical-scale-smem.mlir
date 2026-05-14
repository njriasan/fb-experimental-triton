// RUN: triton-opt -split-input-file --tlx-pack-logical-scale-smem %s | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pack_store_in_default_copy_in_partition
  tt.func public @pack_store_in_default_copy_in_partition(
      %scale: tensor<128x8xi8, #blocked>,
      %smem: !ttg.memdesc<128x8xi8, #shared, #smem, mutable>,
      %tmem: !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>) {
    ttg.warp_specialize(%smem, %tmem)
    default {
      // CHECK: %[[PACKED_VIEW_STORE:.*]] = ttg.memdesc_reshape {{.*}} : !ttg.memdesc<128x8xi8, {{.*}}> -> !ttg.memdesc<1x2x32x16xi8, {{.*}}>
      // CHECK: %[[RESHAPE_5D:.*]] = tt.reshape {{.*}} : tensor<128x8xi8, {{.*}}> -> tensor<1x4x32x2x4xi8, {{.*}}>
      // CHECK: %[[TRANS:.*]] = tt.trans %[[RESHAPE_5D]] {order = array<i32: 0, 3, 2, 1, 4>} : tensor<1x4x32x2x4xi8, {{.*}}> -> tensor<1x2x32x4x4xi8, {{.*}}>
      // CHECK: %[[PACKED_TENSOR:.*]] = tt.reshape %[[TRANS]] : tensor<1x2x32x4x4xi8, {{.*}}> -> tensor<1x2x32x16xi8, {{.*}}>
      // CHECK: ttg.local_store %[[PACKED_TENSOR]], %[[PACKED_VIEW_STORE]] : tensor<1x2x32x16xi8, {{.*}}> -> !ttg.memdesc<1x2x32x16xi8, {{.*}}>
      ttg.local_store %scale, %smem : tensor<128x8xi8, #blocked> -> !ttg.memdesc<128x8xi8, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<128x8xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>) num_warps(1) {
      // CHECK: %[[PACKED_VIEW_COPY:.*]] = ttg.memdesc_reshape {{.*}} : !ttg.memdesc<128x8xi8, {{.*}}> -> !ttg.memdesc<1x2x32x16xi8, {{.*}}>
      // CHECK: ttng.tmem_copy %[[PACKED_VIEW_COPY]], {{.*}} : !ttg.memdesc<1x2x32x16xi8, {{.*}}>, !ttg.memdesc<128x8xi8, {{.*}}>
      ttng.tmem_copy %arg0, %arg1 : !ttg.memdesc<128x8xi8, #shared, #smem, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<128x8xi8, #shared, #smem, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared_5d = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8, rank = 5}>
#smem = #ttg.shared_memory
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pack_logical_store_to_5d_scale_smem
  tt.func public @pack_logical_store_to_5d_scale_smem(
      %scale: tensor<128x4xi8, #blocked>,
      %smem: !ttg.memdesc<1x1x1x2x256xi8, #shared_5d, #smem, mutable>,
      %tmem: !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>) {
    ttg.warp_specialize(%smem, %tmem)
    default {
      // CHECK: %[[PACKED_VIEW_STORE:.*]] = ttg.memdesc_reshape {{.*}} : !ttg.memdesc<1x1x1x2x256xi8, {{.*}}> -> !ttg.memdesc<1x1x32x16xi8, {{.*}}>
      // CHECK: %[[RESHAPE_5D:.*]] = tt.reshape {{.*}} : tensor<128x4xi8, {{.*}}> -> tensor<1x4x32x1x4xi8, {{.*}}>
      // CHECK: %[[TRANS:.*]] = tt.trans %[[RESHAPE_5D]] {order = array<i32: 0, 3, 2, 1, 4>} : tensor<1x4x32x1x4xi8, {{.*}}> -> tensor<1x1x32x4x4xi8, {{.*}}>
      // CHECK: %[[PACKED_TENSOR:.*]] = tt.reshape %[[TRANS]] : tensor<1x1x32x4x4xi8, {{.*}}> -> tensor<1x1x32x16xi8, {{.*}}>
      // CHECK: ttg.local_store %[[PACKED_TENSOR]], %[[PACKED_VIEW_STORE]] : tensor<1x1x32x16xi8, {{.*}}> -> !ttg.memdesc<1x1x32x16xi8, {{.*}}>
      ttg.local_store %scale, %smem : tensor<128x4xi8, #blocked> -> !ttg.memdesc<1x1x1x2x256xi8, #shared_5d, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1x1x1x2x256xi8, #shared_5d, #smem, mutable>, %arg1: !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>) num_warps(1) {
      // CHECK: %[[PACKED_VIEW_COPY:.*]] = ttg.memdesc_reshape {{.*}} : !ttg.memdesc<1x1x1x2x256xi8, {{.*}}> -> !ttg.memdesc<1x1x32x16xi8, {{.*}}>
      // CHECK: ttng.tmem_copy %[[PACKED_VIEW_COPY]], {{.*}} : !ttg.memdesc<1x1x32x16xi8, {{.*}}>, !ttg.memdesc<128x4xi8, {{.*}}>
      ttng.tmem_copy %arg0, %arg1 : !ttg.memdesc<1x1x1x2x256xi8, #shared_5d, #smem, mutable>, !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1x1x1x2x256xi8, #shared_5d, #smem, mutable>, !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared_5d = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8, rank = 5}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory
#dummy_tmem_layout = #tlx.dummy_tmem_layout<>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pack_logical_store_to_dummy_tmem_alias
  tt.func public @pack_logical_store_to_dummy_tmem_alias(
      %scale: tensor<128x4xi8, #blocked>,
      %smem: !ttg.memdesc<1x1x1x2x256xi8, #shared_5d, #smem, mutable>,
      %tmem: !ttg.memdesc<97x128x4xi8, #dummy_tmem_layout, #tmem, mutable>) {
    ttg.warp_specialize(%smem, %tmem)
    default {
      // CHECK: %[[PACKED_VIEW_STORE:.*]] = ttg.memdesc_reshape {{.*}} : !ttg.memdesc<1x1x1x2x256xi8, {{.*}}> -> !ttg.memdesc<1x1x32x16xi8, {{.*}}>
      // CHECK: %[[RESHAPE_5D:.*]] = tt.reshape {{.*}} : tensor<128x4xi8, {{.*}}> -> tensor<1x4x32x1x4xi8, {{.*}}>
      // CHECK: %[[TRANS:.*]] = tt.trans %[[RESHAPE_5D]] {order = array<i32: 0, 3, 2, 1, 4>} : tensor<1x4x32x1x4xi8, {{.*}}> -> tensor<1x1x32x4x4xi8, {{.*}}>
      // CHECK: %[[PACKED_TENSOR:.*]] = tt.reshape %[[TRANS]] : tensor<1x1x32x4x4xi8, {{.*}}> -> tensor<1x1x32x16xi8, {{.*}}>
      // CHECK: ttg.local_store %[[PACKED_TENSOR]], %[[PACKED_VIEW_STORE]] : tensor<1x1x32x16xi8, {{.*}}> -> !ttg.memdesc<1x1x32x16xi8, {{.*}}>
      ttg.local_store %scale, %smem : tensor<128x4xi8, #blocked> -> !ttg.memdesc<1x1x1x2x256xi8, #shared_5d, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1x1x1x2x256xi8, #shared_5d, #smem, mutable>, %arg1: !ttg.memdesc<97x128x4xi8, #dummy_tmem_layout, #tmem, mutable>) num_warps(1) {
      // CHECK: %[[PACKED_VIEW_COPY:.*]] = ttg.memdesc_reshape {{.*}} : !ttg.memdesc<1x1x1x2x256xi8, {{.*}}> -> !ttg.memdesc<1x1x32x16xi8, {{.*}}>
      // CHECK: ttng.tmem_copy %[[PACKED_VIEW_COPY]], {{.*}} : !ttg.memdesc<1x1x32x16xi8, {{.*}}>, !ttg.memdesc<97x128x4xi8, {{.*}}>
      ttng.tmem_copy %arg0, %arg1 : !ttg.memdesc<1x1x1x2x256xi8, #shared_5d, #smem, mutable>, !ttg.memdesc<97x128x4xi8, #dummy_tmem_layout, #tmem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1x1x1x2x256xi8, #shared_5d, #smem, mutable>, !ttg.memdesc<97x128x4xi8, #dummy_tmem_layout, #tmem, mutable>) -> ()
    tt.return
  }
}
