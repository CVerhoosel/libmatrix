LIBMATRIX_METHODS := $(shell grep -Po '(?<=^  void ).*(?=\(\))' libmatrix.cpp)

null :=
space := $(null) #
comma := ,

USER_CXX_FLAGS := \
	-std=c++11 \
	-DFUNCNAMES='$(subst $(space),$(comma) ,$(foreach token,$(LIBMATRIX_METHODS),"$(token)"))' \
	-DFUNCS='$(subst $(space),$(comma) ,$(foreach token,$(LIBMATRIX_METHODS),&LibMatrix::$(token)))'

include $(TRILINOSPATH)/include/Makefile.export.Trilinos

# Make sure to use same compilers and flags as Trilinos
CXX=$(Trilinos_CXX_COMPILER)
CXX_FLAGS=$(Trilinos_CXX_COMPILER_FLAGS) $(USER_CXX_FLAGS)

INCLUDE_DIRS=$(Trilinos_INCLUDE_DIRS) $(Trilinos_TPL_INCLUDE_DIRS)
LIBRARY_DIRS=$(Trilinos_LIBRARY_DIRS) $(Trilinos_TPL_LIBRARY_DIRS)
LIBRARIES=$(Trilinos_LIBRARIES) $(Trilinos_TPL_LIBRARIES)

LINK_FLAGS=$(Trilinos_EXTRA_LD_FLAGS)

default: print_info libmatrix.mpi

# Echo trilinos build info just for fun
print_info:
	@echo "\nFound Trilinos!  Here are the details: "
	@echo "   Trilinos_VERSION = $(Trilinos_VERSION)"
	@echo "   Trilinos_PACKAGE_LIST = $(Trilinos_PACKAGE_LIST)"
	@echo "   Trilinos_LIBRARIES = $(Trilinos_LIBRARIES)"
	@echo "   Trilinos_INCLUDE_DIRS = $(Trilinos_INCLUDE_DIRS)"
	@echo "   Trilinos_LIBRARY_DIRS = $(Trilinos_LIBRARY_DIRS)"
	@echo "   Trilinos_TPL_LIST = $(Trilinos_TPL_LIST)"
	@echo "   Trilinos_TPL_INCLUDE_DIRS = $(Trilinos_TPL_INCLUDE_DIRS)"
	@echo "   Trilinos_TPL_LIBRARIES = $(Trilinos_TPL_LIBRARIES)"
	@echo "   Trilinos_TPL_LIBRARY_DIRS = $(Trilinos_TPL_LIBRARY_DIRS)"
	@echo "   Trilinos_BUILD_SHARED_LIBS = $(Trilinos_BUILD_SHARED_LIBS)"
	@echo "End of Trilinos details\n"

%.o: %.cpp
	$(CXX) -c $(CXX_FLAGS) $< -o $@ $(INCLUDE_DIRS)

libmatrix.mpi: libmatrix.o utils.o
	$(CXX) $(CXX_FLAGS) $^ -o $@ $(LINK_FLAGS) $(INCLUDE_DIRS) $(LIBRARY_DIRS) $(LIBRARIES)

clean:
	rm -f libmatrix.mpi *.o *.pyc

test: libmatrix.mpi
	python test.py

.PHONY: default clean test
	
# vim:noexpandtab
