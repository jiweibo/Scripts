LLVM_CONFIG?=llvm-config

ifndef VERBOSE
#QUIET:=@
endif

SRC_DIR?=$(PWD)
LDFLAGS+=$(shell $(LLVM_CONFIG) --ldflags)
COMMON_FLAGS=-Wall -Wextra
CXXFLAGS+=$(COMMON_FLAGS) $(shell $(LLVM_CONFIG) --cxxflags) $(shell $(LLVM_CONFIG) --system-libs)
CPPFLAGS+=$(shell $(LLVM_CONFIG) --cppflags) -I$(SRC_DIR)

LLVMLIBS=$(shell $(LLVM_CONFIG) --libs)

default: generator.bin FnArgCnt.so

%.o : $(SRC_DIR)/%.cc
	@echo Compiling $*.cc
	$(QUIET)$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) -fPIC $<

%.bin : %.o
	@echo Linking $@
	$(QUIET)$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(CLANGLIBS) $(LLVMLIBS) $(CXXFLAGS)

%.so : %.o
	@echo Linking $@
	$(QUIET)$(CXX) -o $@ -shared -fPIC -Wl,-O1 $(CXXFLAGS) $(LDFLAGS) $^

.PHONY: clean

clean:
	$(QUIET)rm -f *.o *.bin FnArgCnt.so
