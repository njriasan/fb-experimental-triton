// RUN: triton-opt --tlx-pack-logical-scale-smem --verify-diagnostics %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_non_copy_read(
      %scale: tensor<128x8xi8, #blocked>,
      %smem: !ttg.memdesc<128x8xi8, #shared, #smem, mutable>,
      %tmem: !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>) {
    ttg.warp_specialize(%smem, %tmem)
    default {
      ttg.local_store %scale, %smem : tensor<128x8xi8, #blocked> -> !ttg.memdesc<128x8xi8, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<128x8xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>) num_warps(4) {
      ttng.tmem_copy %arg0, %arg1 : !ttg.memdesc<128x8xi8, #shared, #smem, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>
      // expected-error @+1 {{uses a SMEM slot that would be packed for scale tmem_copy}}
      %0 = ttg.local_load %arg0 : !ttg.memdesc<128x8xi8, #shared, #smem, mutable> -> tensor<128x8xi8, #blocked>
      ttg.warp_return
    } : (!ttg.memdesc<128x8xi8, #shared, #smem, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>) -> ()
    tt.return
  }
}
