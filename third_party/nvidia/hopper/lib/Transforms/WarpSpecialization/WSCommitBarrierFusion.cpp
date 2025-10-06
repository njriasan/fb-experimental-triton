
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace tt = mlir::triton;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

// Find all ttng::TCGen5CommitOp that could be theoritically
// fused together if the consumers are compatible.
SmallVector<ttng::TCGen5CommitOp>
collectCommitGroup(ttng::TCGen5CommitOp &commitOp,
                   DenseSet<ttng::TCGen5CommitOp> &seenCommits) {
  SmallVector<ttng::TCGen5CommitOp> commitGroup;
  auto block = commitOp->getBlock();
  auto startit = mlir::Block::iterator(commitOp);
  for (auto it = startit; it != block->end(); it++) {
    if (auto op = dyn_cast<ttng::TCGen5CommitOp>(*it)) {
      if (!seenCommits.count(op)) {
        seenCommits.insert(op);
        commitGroup.push_back(op);
      }
    } else {
      // We currently only support all ttng::TCGen5CommitOp
      // being grouped together.
      break;
    }
  }
  return commitGroup;
}

// Fuse together the barriers used by repeated
// tcgen05.commit operations. This works with the following
// setup:
// 1, Collect all tcgen05.commit operations that logically occur
// "concurrently" and especially without any intermediate mma ops.
// Right now we only support commit operations that are placed next
// to each other in the IR, but in theory this can be extended.
//
// 2. For each candidate group, group together barriers based on the underlying
// consumer(s). We will form a subgroup if the barrier:
//    a. Has identical pipelining state.
//    b. Will occur with the same frequency. This requires an equivalent
//    "nesting" with the same underlying condition.
//    c. Has the same expected phase value.
//
// 3. For each subgroup, update the barriers based on the consumer's location.
//    a. Within the same partition remove all waits but the earliest location in
//    program order.
//    b. Within different partitions unify all barriers to use the same source
//    barrier buffer.
//
// 4. Cleanup the code to remove the unused barriers.
void fuseTcgen05CommitBarriers(tt::FuncOp &funcOp) {
  DenseSet<ttng::TCGen5CommitOp> seenCommits;
  SmallVector<SmallVector<ttng::TCGen5CommitOp>> commitGroups;
  funcOp.walk([&](ttng::TCGen5CommitOp &commitOp) {
    if (seenCommits.count(commitOp)) {
      return;
    }
    auto commitGroup = collectCommitGroup(commitOp, seenCommits);
    if (commitGroup.size() > 1) {
      commitGroups.push_back(commitGroup);
    }
  });
  for (auto commitGroup : commitGroups) {
  }
}

} // namespace mlir
