#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/ToolOutputFile.h"

#include <memory>

using namespace llvm;

int main() {
  LLVMContext context;
  // Create sum module to put out function into it.
  auto owner = std::make_unique<Module>("sum", context);
  Module *m = owner.get();

  // Create the sum function entry and insert this entry into module m.
  // The function will have a return type of "int" and take two arguments of
  // "int".
  Function *sum = Function::Create(
      FunctionType::get(Type::getInt32Ty(context),
                        {Type::getInt32Ty(context), Type::getInt32Ty(context)},
                        false),
      Function::ExternalLinkage, "sum", m);

  // Add a basic block to the function. As before, it automatically
  // inserts after the last parameter.
  BasicBlock *bb = BasicBlock::Create(context, "EntryBlock", sum);

  // Create a basic block builder. The builder will automatically append
  // instructions to the basic block.
  IRBuilder<> builder(bb);

  Argument *arg_x = sum->arg_begin();
  arg_x->setName("x");
  Argument *arg_y = sum->arg_begin() + 1;
  arg_y->setName("y");

  // Create the add instruction, inserting in into the end of bb.
  Value *add = builder.CreateAdd(arg_x, arg_y);
  // Create the return instruction and add it to the basic block.
  builder.CreateRet(add);

  // Validate the generatord code.
  verifyFunction(*sum);

  std::error_code error_code;
  outs() << "Write bitcode to file: sum.bc\n\n";
  std::unique_ptr<ToolOutputFile> out(
      new ToolOutputFile("./sum.bc", error_code, sys::fs::F_None));
  WriteBitcodeToFile(*m, out->os());
  out->keep();
  return 0;
}
