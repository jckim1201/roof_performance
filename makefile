# Makefile for Roofline Performance Suite
# Supports multiple compilers: nvfortran, gfortran, ifx
# Organized directory structure with src/ and build/

# Default compiler
FC ?= gfortran

# Detect compiler and set appropriate flags
ifeq ($(FC),nvfortran)
    FFLAGS = -O3 -mp=multicore -Minfo=all
    MODFLAGS = -module           # where to WRITE .mod
    INCFLAG  = -I                # where to FIND  .mod
    COMPILER_NAME = NVFORTRAN
else ifeq ($(FC),gfortran)
    FFLAGS = -O3 -fopenmp -march=native -mtune=native
    MODFLAGS = -J                # where to WRITE .mod
    INCFLAG  = -I                # where to FIND  .mod
    COMPILER_NAME = GNU_Fortran
else ifeq ($(FC),ifx)
    FFLAGS = -O3 -qopenmp -xHost
    MODFLAGS = -module           # where to WRITE .mod
    INCFLAG  = -I                # where to FIND  .mod
    COMPILER_NAME = Intel_Fortran
else ifeq ($(FC),ifort)
    FFLAGS = -O3 -qopenmp -xHost
    MODFLAGS = -module
    INCFLAG  = -I
    COMPILER_NAME = Intel_Fortran_Classic
else
    FFLAGS = -O3 -fopenmp -march=native -mtune=native
    MODFLAGS = -J
    INCFLAG  = -I
    COMPILER_NAME = Unknown
endif

# Additional optimization flags (can be overridden)
OPT_FLAGS ?=
DEBUG_FLAGS = -g -O0 -Wall

# Directory structure
SRC_DIR     = src
BUILD_DIR   = obj
MOD_DIR     = mod
BIN_DIR     = ./
RESULTS_DIR = results
SCRIPTS_DIR = scripts
DOCS_DIR    = docs

# Ensure directories exist early (now includes MOD_DIR)
$(shell mkdir -p $(SRC_DIR) $(BUILD_DIR) $(BIN_DIR) $(RESULTS_DIR) $(SCRIPTS_DIR) $(DOCS_DIR) $(MOD_DIR))

# Source files
MAIN_SOURCE     = $(SRC_DIR)/roofline_performance_suite_nvfortran.f90
ORIGINAL_SOURCE = $(SRC_DIR)/roofline_microbench_nvfortran.f90

# Module and object directories
MODDIR := $(abspath $(MOD_DIR))
OBJDIR := $(BUILD_DIR)

# Output binaries
TARGET          = $(BIN_DIR)/roofline_suite

# Scripts
RUN_SCRIPT = $(SCRIPTS_DIR)/run_roofline_benchmark.sh
VIZ_SCRIPT = $(SCRIPTS_DIR)/roofline_visualizer.py

# Timestamp for results
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

# Set module directory flags
MODULE_FLAGS  = $(MODFLAGS) $(MODDIR)   # write .mod into MODDIR
MODULE_INCS   = $(INCFLAG) $(MODDIR)    # find  .mod from MODDIR

# Default target
.PHONY: all
all: setup $(TARGET)
	@echo "================================================"
	@echo "Build completed with $(COMPILER_NAME)"
	@echo "Binary:  $(TARGET)"
	@echo "Modules: $(MODDIR)/"
	@echo "Objects: $(OBJDIR)/"
	@echo "================================================"
	@echo "Run with: $(TARGET)"

# Setup directory structure and move source files if needed
.PHONY: setup
setup:
	@echo "Setting up directory structure..."
	@mkdir -p $(SRC_DIR) $(BUILD_DIR) $(BIN_DIR) $(RESULTS_DIR) $(SCRIPTS_DIR) $(DOCS_DIR) $(MOD_DIR)
	@# Move source files to src/ if they exist in root
	@if [ -f "roofline_performance_suite_nvfortran.f90" ]; then \
		echo "Moving source files to $(SRC_DIR)/..."; \
		mv -f roofline_performance_suite_nvfortran.f90 $(SRC_DIR)/ 2>/dev/null || true; \
	fi
	@if [ -f "roofline_microbench_nvfortran.f90" ]; then \
		mv -f roofline_microbench_nvfortran.f90 $(SRC_DIR)/ 2>/dev/null || true; \
	fi
	@# Move scripts if they exist in root
	@if [ -f "run_roofline_benchmark.sh" ]; then \
		echo "Moving scripts to $(SCRIPTS_DIR)/..."; \
		mv -f run_roofline_benchmark.sh $(SCRIPTS_DIR)/ 2>/dev/null || true; \
		chmod +x $(SCRIPTS_DIR)/run_roofline_benchmark.sh; \
	fi
	@if [ -f "roofline_visualizer.py" ]; then \
		mv -f roofline_visualizer.py $(SCRIPTS_DIR)/ 2>/dev/null || true; \
	fi
	@# Create README if it doesn't exist
	@if [ ! -f "README.md" ]; then \
		echo "# Roofline Performance Suite" > README.md; \
		echo "" >> README.md; \
		echo "## Directory Structure" >> README.md; \
		echo "- src/     : Source code files" >> README.md; \
		echo "- build/   : Object files" >> README.md; \
		echo "- mod/     : Fortran module files (.mod)" >> README.md; \
		echo "- bin/     : Compiled binaries" >> README.md; \
		echo "- scripts/ : Helper scripts" >> README.md; \
		echo "- results/ : Benchmark results" >> README.md; \
		echo "- docs/    : Documentation" >> README.md; \
	fi

# Main build target
$(TARGET): $(MAIN_SOURCE) | setup
	@echo "Building with $(COMPILER_NAME)..."
	@echo "Compiler: $(FC)"
	@echo "Flags:    $(FFLAGS) $(OPT_FLAGS)"
	@echo "Mod dir:  $(MODDIR)"
	cd $(BUILD_DIR) && $(FC) $(FFLAGS) $(OPT_FLAGS) $(MODULE_FLAGS) $(MODULE_INCS) ../$< -o ../$@
	@echo "Build successful!"

# Debug build
.PHONY: debug
debug: FFLAGS = $(DEBUG_FLAGS)
debug: $(TARGET)
	@echo "Debug build completed"

# Intel-specific build with MKL
.PHONY: intel-mkl
intel-mkl: FC = ifx
intel-mkl: FFLAGS = -O3 -qopenmp -xHost -qmkl=parallel
intel-mkl: $(TARGET)
	@echo "Intel MKL build completed"

# NVIDIA HPC SDK build with GPU support (experimental)
.PHONY: nvidia-gpu
nvidia-gpu: FC = nvfortran
nvidia-gpu: FFLAGS = -O3 -mp=gpu -gpu=cc80 -Minfo=all
nvidia-gpu: $(TARGET)
	@echo "NVIDIA GPU build completed (experimental)"

# Run quick test
.PHONY: test
test: $(TARGET)
	@echo "Running quick test..."
	$(TARGET) --N=1000000 --iters=2 --warmup=1 --stats=2

# Run full benchmark
.PHONY: benchmark
benchmark: $(TARGET)
	@echo "Running full benchmark suite..."
	@mkdir -p $(RESULTS_DIR)/bench_$(TIMESTAMP)
	$(TARGET) --output=csv --outfile=$(RESULTS_DIR)/bench_$(TIMESTAMP)/results.csv
	@echo "Results saved to $(RESULTS_DIR)/bench_$(TIMESTAMP)/"

# Run scaling study
.PHONY: scaling
scaling: $(TARGET)
	@echo "Running scaling study..."
	@if [ -f "$(RUN_SCRIPT)" ]; then \
		cd $(SCRIPTS_DIR) && bash run_roofline_benchmark.sh --scaling; \
	else \
		echo "Error: $(RUN_SCRIPT) not found"; \
	fi

# Run with validation
.PHONY: validate
validate: $(TARGET)
	@echo "Running with validation..."
	$(TARGET) --validate --N=10000000

# Profile with gprof (GNU only)
.PHONY: profile
profile: FC = gfortran
profile: FFLAGS = -O3 -fopenmp -pg -march=native
profile: $(TARGET)
	@echo "Running profiled version..."
	$(TARGET) --N=10000000 --iters=5
	gprof $(TARGET) gmon.out > $(RESULTS_DIR)/profile_report.txt
	@echo "Profile report saved to $(RESULTS_DIR)/profile_report.txt"

# Check OpenMP configuration
.PHONY: check-openmp
check-openmp:
	@echo "OpenMP Configuration:"
	@echo "OMP_NUM_THREADS = ${OMP_NUM_THREADS}"
	@echo "OMP_PROC_BIND   = ${OMP_PROC_BIND}"
	@echo "OMP_PLACES      = ${OMP_PLACES}"
	@echo ""
	@echo "Available cores: $(shell nproc)"

# System information
.PHONY: sysinfo
sysinfo:
	@echo "System Information:"
	@echo "==================="
	@uname -a
	@echo ""
	@if command -v lscpu >/dev/null 2>&1; then \
		echo "CPU Information:"; \
		lscpu | grep -E "Model name|Socket|Core|Thread|MHz|Cache"; \
	fi
	@echo ""
	@free -h

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(BUILD_DIR)/*.o
	@rm -f $(MOD_DIR)/*.mod $(MOD_DIR)/*.smod
	@rm -f gmon.out 

# Clean binaries
.PHONY: clean-bin
clean-bin:
	@echo "Cleaning binaries..."
	@rm -f $(BIN_DIR)/*

# Clean everything except source
.PHONY: distclean
distclean: clean clean-bin
	@echo "Cleaning all generated files..."
	@rm -rf $(BUILD_DIR)/* $(BIN_DIR)/*
	@rm -f *.json *.csv *.txt

# Really clean everything including results
.PHONY: realclean
realclean: distclean
	@echo "Cleaning all results..."
	@rm -rf $(RESULTS_DIR)/*

# Install Python dependencies for visualization
.PHONY: install-viz
install-viz:
	@echo "Installing Python visualization dependencies..."
	pip install numpy matplotlib pandas seaborn

# Generate visualization from latest results
.PHONY: visualize
visualize:
	@if [ -z "$(CSV_FILE)" ]; then \
		echo "Looking for most recent CSV file..."; \
		CSV_FILE=$$(find $(RESULTS_DIR) -name "*.csv" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-); \
		if [ -n "$$CSV_FILE" ]; then \
			echo "Found: $$CSV_FILE"; \
			if [ -f "$(VIZ_SCRIPT)" ]; then \
				python $(VIZ_SCRIPT) "$$CSV_FILE" --output-dir=$$(dirname "$$CSV_FILE"); \
			else \
				echo "Error: $(VIZ_SCRIPT) not found"; \
			fi \
		else \
			echo "No CSV files found in $(RESULTS_DIR)"; \
			echo "Run 'make benchmark' first to generate results"; \
		fi \
	else \
		if [ -f "$(VIZ_SCRIPT)" ]; then \
			python $(VIZ_SCRIPT) $(CSV_FILE) --output-dir=$$(dirname $(CSV_FILE)); \
		else \
			echo "Error: $(VIZ_SCRIPT) not found"; \
		fi \
	fi

# Show directory structure
.PHONY: tree
tree:
	@echo "Project Structure:"
	@echo "==================="
	@echo "."
	@echo "├── Makefile"
	@echo "├── README.md"
	@echo "├── $(SRC_DIR)/"
	@echo "│   ├── roofline_performance_suite_nvfortran.f90"
	@echo "│   └── roofline_microbench_nvfortran.f90"
	@echo "├── $(BUILD_DIR)/"
	@echo "│   └── *.o   (Object files)"
	@echo "├── $(MOD_DIR)/"
	@echo "│   └── *.mod (Fortran module files)"
	@echo "├── $(BIN_DIR)/"
	@echo "│   └── roofline_suite (executable)"
	@echo "├── $(SCRIPTS_DIR)/"
	@echo "│   ├── run_roofline_benchmark.sh"
	@echo "│   └── roofline_visualizer.py"
	@echo "├── $(RESULTS_DIR)/"
	@echo "│   └── bench_*/ (benchmark results)"
	@echo "└── $(DOCS_DIR)/"
	@echo "    └── (documentation)"

# Package for distribution
.PHONY: dist
dist: clean
	@echo "Creating distribution package..."
	@mkdir -p dist
	tar czf dist/roofline_suite_$(TIMESTAMP).tar.gz \
		--exclude='dist' --exclude='$(RESULTS_DIR)' \
		--exclude='$(BUILD_DIR)' --exclude='$(MOD_DIR)' \
		--exclude='$(BIN_DIR)' \
		Makefile README.md $(SRC_DIR) $(SCRIPTS_DIR) $(DOCS_DIR)
	@echo "Distribution package created: dist/roofline_suite_$(TIMESTAMP).tar.gz"

# Help target
.PHONY: help
help:
	@echo "Roofline Performance Suite - Makefile"
	@echo "======================================"
	@echo ""
	@echo "Directory Structure:"
	@echo "  src/      - Source code files"
	@echo "  build/    - Object files"
	@echo "  mod/      - Fortran module files (.mod)"
	@echo "  bin/      - Compiled binaries"
	@echo "  scripts/  - Helper scripts"
	@echo "  results/  - Benchmark results"
	@echo "  docs/     - Documentation"
	@echo ""
	@echo "Basic targets:"
	@echo "  make              - Build with default compiler ($(FC))"
	@echo "  make setup        - Setup directory structure"
	@echo "  make FC=nvfortran - Build with NVIDIA compiler"
	@echo "  make FC=gfortran  - Build with GNU compiler"
	@echo "  make FC=ifx       - Build with Intel compiler"
	@echo ""
	@echo "Test and benchmark targets:"
	@echo "  make test         - Run quick test"
	@echo "  make benchmark    - Run full benchmark suite"
	@echo "  make scaling      - Run scaling study"
	@echo "  make validate     - Run with validation"
	@echo ""
	@echo "Analysis targets:"
	@echo "  make profile      - Build and run with profiling (gfortran)"
	@echo "  make visualize    - Generate plots from results"
	@echo ""
	@echo "Utility targets:"
	@echo "  make check-openmp - Check OpenMP configuration"
	@echo "  make sysinfo      - Display system information"
	@echo "  make tree         - Show directory structure"
	@echo "  make dist         - Create distribution package"
	@echo ""
	@echo "Cleaning targets:"
	@echo "  make clean        - Remove build artifacts (.o, .mod)"
	@echo "  make clean-bin    - Remove binaries"
	@echo "  make distclean    - Remove all generated files"
	@echo "  make realclean    - Remove everything including results"
	@echo ""
	@echo "Advanced builds:"
	@echo "  make debug        - Build with debug flags"
	@echo "  make intel-mkl    - Build with Intel MKL"
	@echo "  make nvidia-gpu   - Build with GPU support (experimental)"

# Set default goal
.DEFAULT_GOAL := all

