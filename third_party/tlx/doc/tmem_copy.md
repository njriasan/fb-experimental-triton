# TLX TMEM Copy

## Scope

This document describes TLX `tlx.tmem_copy` and the compiler paths that
lower it to NVIDIA Blackwell `tcgen05.cp`. It covers:

- regular shared-memory to tensor-memory copies,
- scale copies into `tensor_memory_scales_encoding`, and
- TLX logical 2D scale SMEM stores that are packed by the compiler before
  `tmem_copy`.

The Python frontend API is:

```python
tlx.tmem_copy(src_smem, dst_tmem)
```

The generated IR operation is:

```mlir
ttng.tmem_copy %src, %dst : (!ttg.memdesc<... #smem>, !ttg.memdesc<... #tmem>)
```

The operation can also carry an optional shared-memory barrier in IR. The TLX
Python builtin currently exposes the source and destination operands.

## Hardware Model

`ttng.tmem_copy` lowers to `tcgen05.cp`. The instruction copies from SMEM into
TMEM without routing the payload through registers. The compiler emits the
instruction from one elected warp. In paired or two-CTA mode, it also predicates
the instruction so only the leader CTA issues it.

If a barrier operand is present, lowering emits a tcgen05 commit/arrive sequence
after the copy instructions. That barrier can be used to observe completion of
the asynchronous copy.

## IR Verification

The verifier enforces common requirements before lowering:

- the source must be a shared-memory memdesc,
- the destination must be a mutable tensor-memory memdesc,
- an optional barrier must be a shared-memory memdesc,
- the source shared layout must have at least 16-byte alignment,
- only one CTA is accepted by the verifier today,
- transposed or FP4-padded NVMMAShared sources are rejected, and
- dummy TMEM layout is accepted early because later passes resolve it.

After that, verification splits by destination encoding.

For a regular TMEM destination:

- the source and destination shapes must match,
- the destination must use `TensorMemoryEncodingAttr`,
- `blockM` must be 128,
- an NVMMAShared source must be swizzled, and
- the source element type must be 32-bit.

For a scale TMEM destination:

- the destination must use `TensorMemoryScalesEncodingAttr`, and
- an NVMMAShared source must not be swizzled.

The scale path intentionally does less shape verification because it supports
multiple physical SMEM shapes that all represent the same logical scale tensor.

## Regular TMEM Copy

Regular copies preserve the logical element shape between SMEM and TMEM. The
lowering is in `copySharedToTmem` in
`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`.

The lowering computes the linear layout conversion from the source SMEM memdesc
to the destination TMEM memdesc:

```text
cvt = tmem_layout^-1 o smem_layout
```

That conversion is used to select a `tcgen05.cp` atom with the right row count,
column bit width, and multicast mode. The compiler then builds an SMEM matrix
descriptor, iterates over the TMEM columns covered by the copy atom, and emits
one or more `tcgen05.cp` instructions.

This path is for normal TMEM data movement. It expects 32-bit source elements
and a source layout that can be described by the tcgen05 SMEM descriptor.

## Scale TMEM Copy

Scale copies are different. The destination TMEM memdesc has the logical scale
shape:

```text
(BLOCK_MN, BLOCK_K / scale_vec_size)
```

For MXFP scale tensors, `scale_vec_size` is 32. The destination encoding is
`tensor_memory_scales_encoding`, which means the hardware writes a duplicated
scale layout suitable for Blackwell scaled MMA.

The source SMEM does not have to have the same logical shape as the destination.
Instead, it must physically contain packed 32 x 128-bit chunks. Each chunk is
copied by a warp4 `tcgen05.cp` atom and duplicated over the TMEM rows and warp
lanes expected by scaled MMA.

Conceptually, the packed scale SMEM layout is:

```text
(rep_m_or_n, rep_k, 32, 4, 4B)
```

where:

```text
rep_m_or_n = BLOCK_MN / 128
rep_k      = (BLOCK_K / scale_vec_size) / 4
```

Each `(32, 4, 4B)` tile is one 32 x 128-bit hardware block.

The scale lowering accepts several equivalent physical SMEM shapes as long as it
can recover `rep_m_or_n` and `rep_k`, including:

- `(rep_m_or_n * 32, 16B)` for unit-test style inputs,
- `(rep_m_or_n, rep_k * 32 * 4 * 4B)` for 2D cp.async loads,
- `(rep_m_or_n, rep_k, 32, 16B)` for TMA-style packed loads,
- `(1, rep_m_or_n, rep_k, 2, 256B)` for TMA-style 5D loads, and
- `(rep_m_or_n, rep_k, 32, 4, 4B)` for 5D cp.async-style loads.

For each recovered `(i, j)` packed block, the compiler computes the SMEM address
through the source linear layout, creates a blocked-scale SMEM descriptor, and
emits:

```text
tcgen05.cp.cta_group::<1 or 2>.warpx4.32x128b
```

The TMEM address advances in groups of four columns, with blocks ordered by
M/N first and K second.

## Why Logical 2D Scale SMEM Needs Packing

A logical scale tensor may naturally be computed as:

```text
(BLOCK_MN, BLOCK_K / scale_vec_size)
```

For example, users may want to write:

```python
tlx.local_store(tlx.local_view(ds_scale_smem, tmem_buf_id), ds_scale_dq)
tlx.tmem_copy(tlx.local_view(ds_scale_smem, tmem_buf_id), ds_scale_dq_tmem)
```

That is only correct if the SMEM slot is physically packed in the 32 x 128-bit
chunk order expected by the scale `tmem_copy` lowering. A plain rank-2 store
into a rank-5 TMA-shaped SMEM allocation does not, by itself, establish that
physical order. If the source bytes are logically correct but physically laid
out differently, `tcgen05.cp` will copy the wrong scale values into TMEM and the
kernel can fail only by accuracy drift.

Previously, kernels made the physical packing explicit in TLX:

```python
packed = (
    scales
    .reshape([rep_rows, 4, 32, rep_cols, 4])
    .permute(0, 3, 2, 1, 4)
    .reshape([1, rep_rows, rep_cols, 2, 256])
)
tlx.local_store(tlx.local_view(scale_smem, tmem_buf_id), packed)
```

The final shape is convenient for a 5D TMA-shaped allocation, but the
`tmem_copy` lowering only needs the packed 32 x 128-bit chunk order.

## Logical 2D Scale SMEM Compiler Rewrite

TLX has a pass named `tlx-pack-logical-scale-smem` for this case. It runs after
warp specialization, because TLX kernels that use this path are warp
specialized and the pass must see the real producer and consumer partitions. It
also resolves warp-specialize captures, local aliases, memdesc indexes,
subslice views, reinterpret views, transposes, and reshapes when proving which
physical SMEM slot is being used.

When the pass finds a scale `ttng.tmem_copy`, it groups all copies that read the
same SMEM slot in the same warp-specialize op. It then proves the slot is only
used in ways that are compatible with scale packing:

- every read of that SMEM slot, or an alias of it, must be a scale
  `ttng.tmem_copy`,
- writes must be compatible `ttg.local_store` producers or compatible TMA
  producers,
- logical local stores must be rank-2 i8 tensors with the same shape as the
  logical scale TMEM destination,
- already packed local stores are left alone, and
- packed TMA producers are left alone.

If the pass cannot prove those conditions, it rejects the IR instead of silently
rewriting a buffer that may have non-copy consumers.

For a compatible logical rank-2 store, the pass rewrites the producer to the
minimum required packing:

```text
reshape   [rep_rows, 4, 32, rep_cols, 4]
transpose [0, 3, 2, 1, 4]
reshape   [rep_rows, rep_cols, 32, 16]
```

It does not perform the old final reshape to:

```text
[1, rep_rows, rep_cols, 2, 256]
```

Instead, it creates a rank-4 packed memdesc view of the SMEM slot:

```text
[rep_rows, rep_cols, 32, 16]
```

The rewritten local store writes the rank-4 packed tensor into that rank-4 view,
and each matching `ttng.tmem_copy` is rewritten to read from the same rank-4
packed view. The underlying storage can still be the original 5D TMA-shaped
SMEM allocation. The view only changes how the producer store and copy consumer
address the bytes.

## Aliasing and Warp Specialization

The pass keys SMEM slots by the resolved underlying storage plus index and
subslice information. This is why it can handle a buffer passed as a
warp-specialize argument even when the original allocation is in the default
partition.

The pass treats these as view-like aliases of the same slot:

- `ttg.memdesc_index`,
- `ttg.memdesc_subslice`,
- `ttg.memdesc_reinterpret`,
- `ttg.memdesc_reshape`,
- `ttg.memdesc_trans`, and
- TLX `local_alias`.

Constant indexes are normalized by value, so two independently materialized
`%c0_i32` constants still identify the same buffer slot.

The proof is intentionally conservative. If the same physical slot has any load
or non-copy consumer, the compiler cannot rewrite the producer layout without
also changing that consumer's interpretation of the bytes. In that case the pass
fails and asks the kernel to keep an explicit packed producer or to separate the
storage.

## Common Failure Modes

Accuracy issues usually come from a mismatch between logical shape and physical
SMEM packing:

- a logical rank-2 scale store feeds a scale `tmem_copy` without being packed,
- a packed scale SMEM slot is also consumed by a normal local load,
- two copies from the same SMEM slot disagree on the logical scale shape,
- the source is swizzled on the scale path, or
- a regular TMEM copy is accidentally used where a scale TMEM copy was intended.

The logical packing pass is designed to make the first two cases fail loudly or
be rewritten correctly. The verifier and lowerer cover the remaining layout
requirements for the specific TMEM copy mode.

## Practical Guidance

Use regular `tlx.tmem_copy` for normal 32-bit TMEM payloads when the source and
destination have the same logical shape.

Use scale `tlx.tmem_copy` when the destination allocation uses
`tensor_memory_scales_encoding`. In that mode, think of the source SMEM as a
physical packed-scale staging buffer, not as an ordinary logical 2D tensor.

For TLX kernels that compute scales logically in registers, prefer a logical
rank-2 `local_store` into the scale SMEM slot and let
`tlx-pack-logical-scale-smem` insert the minimal packing. Keep every consumer of
that slot as a scale `tmem_copy`, or allocate a separate SMEM buffer for other
uses.
