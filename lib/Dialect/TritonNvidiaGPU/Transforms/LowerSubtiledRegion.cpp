#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPULOWERSUBTILEDREGIONPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

void lowerSubtiledRegion(SubtiledRegionOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();

  // 1. Clone setup region ops (except yield).
  // Map setup block args to the corresponding inputs (IsolatedFromAbove).
  Block &setupBlock = op.getSetupRegion().front();
  IRMapping setupMapping;
  for (auto [blockArg, input] :
       llvm::zip(setupBlock.getArguments(), op.getInputs()))
    setupMapping.map(blockArg, input);
  for (Operation &setupOp : setupBlock.without_terminator())
    builder.clone(setupOp, setupMapping);

  // 2. Collect remapped setup outputs from the cloned yield operands.
  auto yieldOp = cast<SubtiledRegionYieldOp>(setupBlock.getTerminator());
  SmallVector<Value> setupOutputs;
  for (Value v : yieldOp.getResults())
    setupOutputs.push_back(setupMapping.lookupOrDefault(v));

  ArrayAttr tileMappings = op.getTileMappings();
  unsigned numTiles = tileMappings.size();
  Block &tileBlock = op.getTileRegion().front();

  // Detect optional tile index argument: present when tile block has one more
  // arg than the tile mapping entries.
  unsigned numTileArgs = tileBlock.getNumArguments();
  unsigned mappingSize =
      cast<DenseI32ArrayAttr>(tileMappings[0]).asArrayRef().size();
  bool hasTileIndex = (numTileArgs == mappingSize + 1);

  // 3. For each tile, clone tile region ops with substitution.
  for (unsigned tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
    auto indices = cast<DenseI32ArrayAttr>(tileMappings[tileIdx]);
    IRMapping tileMapping;

    for (auto [j, idx] : llvm::enumerate(indices.asArrayRef()))
      tileMapping.map(tileBlock.getArgument(j), setupOutputs[idx]);

    if (hasTileIndex) {
      Value tileIdxConst = arith::ConstantOp::create(
          builder, loc, builder.getI32IntegerAttr(tileIdx));
      tileMapping.map(tileBlock.getArgument(numTileArgs - 1), tileIdxConst);
    }

    for (Operation &tileOp : tileBlock.without_terminator())
      builder.clone(tileOp, tileMapping);
  }

  // 4. Clone teardown region ops (except terminator).
  Block &teardownBlock = op.getTeardownRegion().front();
  IRMapping teardownMapping;
  for (Operation &teardownOp : teardownBlock.without_terminator())
    builder.clone(teardownOp, teardownMapping);

  // 5. Replace op results with teardown yield values.
  auto teardownTerminator =
      cast<SubtiledRegionYieldOp>(teardownBlock.getTerminator());
  for (auto [opResult, teardownVal] :
       llvm::zip(op.getResults(), teardownTerminator.getResults()))
    opResult.replaceAllUsesWith(teardownMapping.lookupOrDefault(teardownVal));

  // 6. Erase the SubtiledRegionOp.
  op.erase();
}

namespace {

class TritonNvidiaGPULowerSubtiledRegionPass
    : public impl::TritonNvidiaGPULowerSubtiledRegionPassBase<
          TritonNvidiaGPULowerSubtiledRegionPass> {
public:
  using TritonNvidiaGPULowerSubtiledRegionPassBase::
      TritonNvidiaGPULowerSubtiledRegionPassBase;

  void runOnOperation() override {
    SmallVector<SubtiledRegionOp> ops;
    getOperation().walk([&](SubtiledRegionOp op) { ops.push_back(op); });

    for (auto op : ops)
      lowerSubtiledRegion(op);
  }
};

} // namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
