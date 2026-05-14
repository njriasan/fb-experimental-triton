#include "IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "tlx-pack-logical-scale-smem"

using namespace mlir;
namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace tlx {

#define GEN_PASS_DEF_TLXPACKLOGICALSCALESMEM
#include "tlx/dialect/include/Transforms/Passes.h.inc"

namespace {

struct ScaleCopyGroup {
  std::string key;
  ttg::WarpSpecializeOp ws;
  int64_t rows = 0;
  int64_t cols = 0;
  bool hasPackedScaleSmemCopySource = false;
  SmallVector<ttng::TMEMCopyOp> logicalCopies;
};

static ScaleCopyGroup *findGroup(SmallVectorImpl<ScaleCopyGroup> &groups,
                                 StringRef key, ttg::WarpSpecializeOp ws) {
  for (ScaleCopyGroup &group : groups) {
    if (group.key == key && group.ws == ws)
      return &group;
  }
  return nullptr;
}

static Value resolveWarpSpecializeCapture(Value value) {
  while (auto arg = dyn_cast<BlockArgument>(value)) {
    auto *parentOp = arg.getOwner()->getParentOp();
    auto partitions =
        dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(parentOp);
    if (!partitions || arg.getArgNumber() >= partitions.getNumOperands())
      break;
    value = partitions.getExplicitCaptures()[arg.getArgNumber()];
  }
  return value;
}

static void appendPtrKey(llvm::raw_ostream &os, const void *ptr) {
  os << "0x";
  os.write_hex(reinterpret_cast<uintptr_t>(ptr));
}

static std::string getValueKey(Value value) {
  value = resolveWarpSpecializeCapture(value);
  std::string storage;
  llvm::raw_string_ostream os(storage);
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    os << "arg:";
    appendPtrKey(os, blockArg.getOwner());
    os << ":" << blockArg.getArgNumber();
    return os.str();
  }
  os << "val:";
  appendPtrKey(os, value.getDefiningOp());
  os << ":" << cast<OpResult>(value).getResultNumber();
  return os.str();
}

static std::string getIndexKey(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIntOp>())
    return ("const:" + Twine(constant.value())).str();
  return getValueKey(value);
}

static void appendArrayKey(llvm::raw_ostream &os, ArrayRef<int32_t> values) {
  os << "[";
  llvm::interleaveComma(values, os);
  os << "]";
}

static std::string getSmemSlotKey(Value value) {
  value = resolveWarpSpecializeCapture(value);

  std::string storage;
  llvm::raw_string_ostream os(storage);
  if (auto op = value.getDefiningOp<ttg::MemDescIndexOp>()) {
    os << getSmemSlotKey(op.getSrc()) << ".idx(" << getIndexKey(op.getIndex())
       << ")";
    return os.str();
  }
  if (auto op = value.getDefiningOp<ttg::MemDescSubsliceOp>()) {
    os << getSmemSlotKey(op.getSrc()) << ".subslice";
    appendArrayKey(os, op.getOffsets());
    return os.str();
  }
  if (auto op = value.getDefiningOp<ttg::MemDescReinterpretOp>())
    return getSmemSlotKey(op.getSrc());
  if (auto op = value.getDefiningOp<ttg::MemDescReshapeOp>())
    return getSmemSlotKey(op.getSrc());
  if (auto op = value.getDefiningOp<ttg::MemDescTransOp>())
    return getSmemSlotKey(op.getSrc());
  if (auto op = value.getDefiningOp<LocalAliasOp>())
    return getSmemSlotKey(op.getSrc());

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    os << "arg:";
    appendPtrKey(os, blockArg.getOwner());
    os << ":" << blockArg.getArgNumber();
  } else {
    os << "root:";
    appendPtrKey(os, value.getDefiningOp());
    os << ":" << cast<OpResult>(value).getResultNumber();
  }
  return os.str();
}

static bool isRank2I8MemDesc(Value value, int64_t &rows, int64_t &cols) {
  auto type = dyn_cast<ttg::MemDescType>(value.getType());
  if (!type || type.getRank() != 2)
    return false;
  if (!type.getElementType().isIntOrFloat() ||
      type.getElementType().getIntOrFloatBitWidth() != 8)
    return false;
  rows = type.getDimSize(0);
  cols = type.getDimSize(1);
  return true;
}

static bool isPackedScaleSmemDesc(Value value, int64_t rows, int64_t cols) {
  auto type = dyn_cast<ttg::MemDescType>(value.getType());
  if (!type || type.getRank() < 5)
    return false;
  if (!type.getElementType().isIntOrFloat() ||
      type.getElementType().getIntOrFloatBitWidth() != 8)
    return false;
  ArrayRef<int64_t> shape = type.getShape();
  if (shape[type.getRank() - 2] != 2 || shape[type.getRank() - 1] != 256)
    return false;
  int64_t numElements = 1;
  for (int64_t dim : shape)
    numElements *= dim;
  return numElements == rows * cols;
}

static bool getScaleTmemDims(ttng::TMEMCopyOp copy, int64_t &rows,
                             int64_t &cols) {
  auto type = dyn_cast<ttg::MemDescType>(copy.getDst().getType());
  if (!type || !type.getElementType().isIntOrFloat() ||
      type.getElementType().getIntOrFloatBitWidth() != 8)
    return false;
  if (isa<ttng::TensorMemoryScalesEncodingAttr>(type.getEncoding())) {
    if (type.getRank() != 2)
      return false;
    rows = type.getDimSize(0);
    cols = type.getDimSize(1);
    return true;
  }
  if (isa<DummyTMEMLayoutAttr>(type.getEncoding())) {
    if (type.getRank() != 2 && type.getRank() != 3)
      return false;
    rows = type.getDimSize(type.getRank() - 2);
    cols = type.getDimSize(type.getRank() - 1);
    return true;
  }
  return false;
}

static bool hasScaleTmemDst(ttng::TMEMCopyOp copy) {
  int64_t rows = 0;
  int64_t cols = 0;
  return getScaleTmemDims(copy, rows, cols);
}

static bool isLogicalScaleCopy(ttng::TMEMCopyOp copy, int64_t &rows,
                               int64_t &cols) {
  if (!getScaleTmemDims(copy, rows, cols))
    return false;
  int64_t srcRows = 0;
  int64_t srcCols = 0;
  if (isRank2I8MemDesc(copy.getSrc(), srcRows, srcCols)) {
    if (srcRows != rows || srcCols != cols)
      return false;
  } else if (!isPackedScaleSmemDesc(copy.getSrc(), rows, cols)) {
    return false;
  }
  return rows % 128 == 0 && cols % 4 == 0;
}

static bool isCompatiblePackedStore(ttg::LocalStoreOp store,
                                    ArrayRef<int64_t> packedShape) {
  auto srcType = dyn_cast<RankedTensorType>(store.getSrc().getType());
  if (!srcType)
    return false;
  if (srcType.getRank() == static_cast<int64_t>(packedShape.size()) &&
      llvm::equal(srcType.getShape(), packedShape))
    return true;
  auto dstType = cast<ttg::MemDescType>(store.getDst().getType());
  return srcType.getShape() == dstType.getShape();
}

static bool isCompatibleLogicalStore(ttg::LocalStoreOp store, int64_t rows,
                                     int64_t cols) {
  auto srcType = dyn_cast<RankedTensorType>(store.getSrc().getType());
  if (!srcType || srcType.getRank() != 2)
    return false;
  return srcType.getDimSize(0) == rows && srcType.getDimSize(1) == cols &&
         srcType.getElementType().isIntOrFloat() &&
         srcType.getElementType().getIntOrFloatBitWidth() == 8;
}

static bool isViewLikeMemDescOp(Operation *op) {
  return isa<ttg::MemDescIndexOp, ttg::MemDescSubsliceOp,
             ttg::MemDescReinterpretOp, ttg::MemDescReshapeOp,
             ttg::MemDescTransOp, LocalAliasOp>(op);
}

static bool opUsesSlot(Operation *op, StringRef key) {
  for (Value operand : op->getOperands()) {
    if (!isa<ttg::MemDescType>(operand.getType()))
      continue;
    if (getSmemSlotKey(operand) == key)
      return true;
  }
  return false;
}

static ttg::MemDescReshapeOp createPackedView(OpBuilder &builder, Location loc,
                                              Value smem,
                                              ArrayRef<int64_t> shape) {
  return ttg::MemDescReshapeOp::create(builder, loc, smem, shape);
}

static Value createPackedTensor(OpBuilder &builder, Location loc, Value src,
                                int64_t repRows, int64_t repCols) {
  SmallVector<int64_t> reshape5d = {repRows, 4, 32, repCols, 4};
  Value reshaped = tt::ReshapeOp::create(builder, loc, reshape5d, src);
  SmallVector<int32_t> order = {0, 3, 2, 1, 4};
  Value transposed = tt::TransOp::create(builder, loc, reshaped, order);
  SmallVector<int64_t> packedShape = {repRows, repCols, 32, 16};
  return tt::ReshapeOp::create(builder, loc, packedShape, transposed);
}

struct TLXPackLogicalScaleSmemPass
    : public impl::TLXPackLogicalScaleSmemBase<TLXPackLogicalScaleSmemPass> {
  using impl::TLXPackLogicalScaleSmemBase<
      TLXPackLogicalScaleSmemPass>::TLXPackLogicalScaleSmemBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<ScaleCopyGroup> groups;

    bool collectionFailed = false;
    module.walk([&](ttng::TMEMCopyOp copy) {
      if (collectionFailed)
        return;
      int64_t rows = 0;
      int64_t cols = 0;
      if (!isLogicalScaleCopy(copy, rows, cols))
        return;
      ttg::WarpSpecializeOp ws = copy->getParentOfType<ttg::WarpSpecializeOp>();
      if (!ws)
        return;
      std::string key = getSmemSlotKey(copy.getSrc());
      ScaleCopyGroup *existingGroup = findGroup(groups, key, ws);
      if (!existingGroup) {
        ScaleCopyGroup group;
        group.key = key;
        group.ws = ws;
        group.rows = rows;
        group.cols = cols;
        group.hasPackedScaleSmemCopySource =
            isPackedScaleSmemDesc(copy.getSrc(), rows, cols);
        group.logicalCopies.push_back(copy);
        groups.push_back(std::move(group));
        return;
      }
      ScaleCopyGroup &group = *existingGroup;
      if (group.rows != rows || group.cols != cols) {
        copy.emitOpError("conflicting logical scale SMEM shapes for the same "
                         "physical SMEM slot");
        collectionFailed = true;
        return;
      }
      group.hasPackedScaleSmemCopySource |=
          isPackedScaleSmemDesc(copy.getSrc(), rows, cols);
      group.logicalCopies.push_back(copy);
    });

    if (collectionFailed) {
      signalPassFailure();
      return;
    }
    if (groups.empty())
      return;

    for (ScaleCopyGroup &group : groups) {
      if (failed(processGroup(group))) {
        signalPassFailure();
        return;
      }
    }
  }

  LogicalResult processGroup(ScaleCopyGroup &group) {
    SmallVector<int64_t> packedShape = {group.rows / 128, group.cols / 4, 32,
                                        16};
    SmallVector<ttg::LocalStoreOp> storesToRewrite;
    SmallVector<ttng::TMEMCopyOp> copiesToRewrite(group.logicalCopies.begin(),
                                                  group.logicalCopies.end());

    WalkResult walk = group.ws.walk([&](Operation *op) -> WalkResult {
      if (!opUsesSlot(op, group.key))
        return WalkResult::advance();

      if (isViewLikeMemDescOp(op) || isa<ttg::WarpSpecializePartitionsOp>(op))
        return WalkResult::advance();

      if (auto store = dyn_cast<ttg::LocalStoreOp>(op)) {
        if (isCompatibleLogicalStore(store, group.rows, group.cols)) {
          storesToRewrite.push_back(store);
          return WalkResult::advance();
        }
        if (isCompatiblePackedStore(store, packedShape))
          return WalkResult::advance();
        store.emitOpError("writes a SMEM slot used by a logical rank-2 scale "
                          "tmem_copy, but the store source is not a compatible "
                          "logical or packed scale shape");
        return WalkResult::interrupt();
      }

      if (auto tmaLoad = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
        if (isPackedScaleSmemDesc(tmaLoad.getResult(), group.rows, group.cols))
          return WalkResult::advance();
        tmaLoad.emitOpError("writes a SMEM slot used by a logical rank-2 "
                            "scale tmem_copy, but the TMA destination is not "
                            "a compatible packed scale shape");
        return WalkResult::interrupt();
      }

      if (auto copy = dyn_cast<ttng::TMEMCopyOp>(op)) {
        if (!hasScaleTmemDst(copy)) {
          copy.emitOpError("reads a SMEM slot rewritten for scale tmem_copy, "
                           "but the destination is not tensor-memory scales");
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }

      op->emitOpError("uses a SMEM slot that would be packed for scale "
                      "tmem_copy; all reads of that slot must be scale "
                      "ttng.tmem_copy and all writes must be compatible "
                      "ttg.local_store");
      return WalkResult::interrupt();
    });

    if (walk.wasInterrupted())
      return failure();
    if (storesToRewrite.empty() && group.hasPackedScaleSmemCopySource)
      return success();
    if (storesToRewrite.empty())
      return group.logicalCopies.front().emitOpError(
          "could not find a compatible logical local_store producer for "
          "logical rank-2 scale SMEM tmem_copy");

    IRRewriter rewriter(&getContext());
    for (ttg::LocalStoreOp store : storesToRewrite) {
      rewriter.setInsertionPoint(store);
      Location loc = store.getLoc();
      Value packedView =
          createPackedView(rewriter, loc, store.getDst(), packedShape);
      Value packedTensor = createPackedTensor(rewriter, loc, store.getSrc(),
                                              group.rows / 128, group.cols / 4);
      auto newStore =
          ttg::LocalStoreOp::create(rewriter, loc, packedTensor, packedView);
      newStore->setAttrs(store->getAttrs());
      rewriter.eraseOp(store);
    }

    for (ttng::TMEMCopyOp copy : copiesToRewrite) {
      rewriter.setInsertionPoint(copy);
      Location loc = copy.getLoc();
      Value packedView =
          createPackedView(rewriter, loc, copy.getSrc(), packedShape);
      auto newCopy = ttng::TMEMCopyOp::create(rewriter, loc, packedView,
                                              copy.getDst(), copy.getBarrier());
      newCopy->setAttrs(copy->getAttrs());
      rewriter.eraseOp(copy);
    }
    return success();
  }
};

} // namespace

} // namespace tlx
} // namespace triton
} // namespace mlir
