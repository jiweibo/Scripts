#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
struct FnArgCnt : public FunctionPass {
  static char ID;
  FnArgCnt() : FunctionPass(ID) {}
  virtual bool runOnFunction(Function &F) {
    errs() << "FnArgCnt --- ";
    errs() << F.getName() << ": ";
    errs() << F.arg_size() << '\n';
    return false;
  }
};
} // namespace

char FnArgCnt::ID = 0;
static RegisterPass<FnArgCnt> X("fnargcnt", "Function Argument Count Pass",
                                false, false);
