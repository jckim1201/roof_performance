#!/bin/bash
# run_roofline_benchmark.sh
# Automated script for running comprehensive Roofline benchmarks
# with CPU frequency control, NUMA binding, and result analysis

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR=$(pwd)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${SCRIPT_DIR}/results_${TIMESTAMP}"
SYSTEM_INFO_FILE="${RESULTS_DIR}/system_info.txt"
BENCHMARK_BINARY="./roofline_suite"

# Default parameters
VECTOR_SIZES="1000000 10000000 100000000"
THREAD_COUNTS="1 2 4 8"
KERNELS="all"
ITERATIONS=10
WARMUP=3
STATS_RUNS=5

# Functions
print_header() {
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}  Roofline Performance Benchmark${NC}"
    echo -e "${BLUE}=====================================${NC}"
}

check_requirements() {
    echo -e "${YELLOW}Checking requirements...${NC}"
    
    # Check for benchmark binary
    if [ ! -f "$BENCHMARK_BINARY" ]; then
        echo -e "${RED}Error: Benchmark binary not found at $BENCHMARK_BINARY${NC}"
        echo -e "${YELLOW}Please compile first with one of:${NC}"
        echo "  nvfortran -O3 -mp=multicore roofline_performance_suite_nvfortran.f90 -o roofline_suite"
        echo "  gfortran -O3 -fopenmp roofline_performance_suite_nvfortran.f90 -o roofline_suite"
        exit 1
    fi
    
    # Check for optional tools
    if command -v lscpu &> /dev/null; then
        HAVE_LSCPU=1
    else
        HAVE_LSCPU=0
        echo -e "${YELLOW}Warning: lscpu not found. System info will be limited.${NC}"
    fi
    
    if command -v numactl &> /dev/null; then
        HAVE_NUMACTL=1
    else
        HAVE_NUMACTL=0
        echo -e "${YELLOW}Warning: numactl not found. NUMA optimization unavailable.${NC}"
    fi
}

gather_system_info() {
    echo -e "${GREEN}Gathering system information...${NC}"
    
    mkdir -p "$RESULTS_DIR"
    
    {
        echo "========================================="
        echo "System Information"
        echo "========================================="
        echo "Date: $(date)"
        echo "Hostname: $(hostname)"
        echo ""
        
        if [ $HAVE_LSCPU -eq 1 ]; then
            echo "CPU Information:"
            echo "----------------"
            lscpu
            echo ""
        fi
        
        echo "Memory Information:"
        echo "-------------------"
        free -h
        echo ""
        
        if [ -f /proc/meminfo ]; then
            echo "Memory Details:"
            grep -E "MemTotal|MemFree|Cached|SwapTotal" /proc/meminfo
            echo ""
        fi
        
        echo "Compiler Information:"
        echo "--------------------"
        if command -v nvfortran &> /dev/null; then
            nvfortran --version 2>&1 | head -n 2
        fi
        if command -v gfortran &> /dev/null; then
            gfortran --version | head -n 1
        fi
        echo ""
        
        echo "OpenMP Information:"
        echo "------------------"
        echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-not set}"
        echo "OMP_PROC_BIND=${OMP_PROC_BIND:-not set}"
        echo "OMP_PLACES=${OMP_PLACES:-not set}"
        echo ""
        
    } > "$SYSTEM_INFO_FILE"
    
    echo -e "${GREEN}System info saved to $SYSTEM_INFO_FILE${NC}"
}

set_cpu_governor() {
    echo -e "${YELLOW}Attempting to set CPU governor to performance...${NC}"
    
    if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
        if [ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
            for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
                echo "performance" > $cpu 2>/dev/null || true
            done
            echo -e "${GREEN}CPU governor set to performance${NC}"
        else
            echo -e "${YELLOW}Cannot set CPU governor (need root). Run with sudo for best results.${NC}"
        fi
    else
        echo -e "${YELLOW}CPU frequency scaling not available on this system${NC}"
    fi
}

run_benchmark_suite() {
    local n_size=$1
    local threads=$2
    local output_file="${RESULTS_DIR}/results_N${n_size}_T${threads}.csv"
    
    echo -e "${BLUE}Running benchmark: N=$n_size, Threads=$threads${NC}"
    
    # Set OpenMP threads
    export OMP_NUM_THREADS=$threads
    
    # Build command
    CMD="$BENCHMARK_BINARY"
    CMD="$CMD --N=$n_size"
    CMD="$CMD --iters=$ITERATIONS"
    CMD="$CMD --warmup=$WARMUP"
    CMD="$CMD --stats=$STATS_RUNS"
    CMD="$CMD --kernel=$KERNELS"
    CMD="$CMD --output=csv"
    CMD="$CMD --outfile=$output_file"
    
    # Add NUMA binding if available
    if [ $HAVE_NUMACTL -eq 1 ] && [ $threads -gt 1 ]; then
        CMD="numactl --interleave=all $CMD"
    fi
    
    # Run benchmark
    echo "Command: $CMD"
    $CMD
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Benchmark completed successfully${NC}"
    else
        echo -e "${RED}Benchmark failed${NC}"
        return 1
    fi
}

run_scaling_study() {
    echo -e "${BLUE}Running scaling study...${NC}"
    
    local n_size=100000000  # Fixed size for scaling study
    
    for threads in $THREAD_COUNTS; do
        run_benchmark_suite $n_size $threads
    done
}

run_size_study() {
    echo -e "${BLUE}Running problem size study...${NC}"
    
    local threads=$(nproc)  # Use all available cores
    
    for n_size in $VECTOR_SIZES; do
        run_benchmark_suite $n_size $threads
    done
}

combine_results() {
    echo -e "${GREEN}Combining results...${NC}"
    
    local combined_file="${RESULTS_DIR}/combined_results.csv"
    local first_file=1
    
    for csv_file in ${RESULTS_DIR}/results_N*.csv; do
        if [ -f "$csv_file" ]; then
            if [ $first_file -eq 1 ]; then
                cat "$csv_file" > "$combined_file"
                first_file=0
            else
                tail -n +2 "$csv_file" >> "$combined_file"
            fi
        fi
    done
    
    echo -e "${GREEN}Combined results saved to $combined_file${NC}"
}

generate_report() {
    echo -e "${GREEN}Generating performance report...${NC}"
    
    local report_file="${RESULTS_DIR}/performance_report.md"
    
    {
        echo "# Roofline Performance Benchmark Report"
        echo ""
        echo "## Test Configuration"
        echo "- Date: $(date)"
        echo "- Vector sizes: $VECTOR_SIZES"
        echo "- Thread counts: $THREAD_COUNTS"
        echo "- Iterations: $ITERATIONS"
        echo "- Warmup: $WARMUP"
        echo "- Statistical runs: $STATS_RUNS"
        echo ""
        echo "## System Information"
        echo '```'
        head -n 20 "$SYSTEM_INFO_FILE"
        echo '```'
        echo ""
        echo "## Results"
        echo "See combined_results.csv for detailed data"
        echo ""
        echo "## Visualizations"
        echo "Run the following command to generate plots:"
        echo '```bash'
        echo "python roofline_visualizer.py ${RESULTS_DIR}/combined_results.csv --output-dir=${RESULTS_DIR}"
        echo '```'
    } > "$report_file"
    
    echo -e "${GREEN}Report saved to $report_file${NC}"
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  -s, --scaling        Run scaling study only"
    echo "  -z, --size           Run size study only"
    echo "  -a, --all            Run all studies (default)"
    echo "  -t, --threads        Thread counts (default: \"$THREAD_COUNTS\")"
    echo "  -n, --sizes          Vector sizes (default: \"$VECTOR_SIZES\")"
    echo "  -k, --kernel         Kernel to test (default: all)"
    echo "  -i, --iterations     Number of iterations (default: $ITERATIONS)"
    echo ""
    echo "Examples:"
    echo "  $0                   # Run all benchmarks"
    echo "  $0 --scaling         # Run scaling study only"
    echo "  $0 --kernel=triad    # Test only TRIAD kernel"
    echo "  $0 --threads=\"1 2 4\" # Test with 1, 2, and 4 threads"
}

# Parse command line arguments
RUN_MODE="all"

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            print_usage
            exit 0
            ;;
        -s|--scaling)
            RUN_MODE="scaling"
            shift
            ;;
        -z|--size)
            RUN_MODE="size"
            shift
            ;;
        -a|--all)
            RUN_MODE="all"
            shift
            ;;
        -t|--threads)
            THREAD_COUNTS="$2"
            shift 2
            ;;
        -n|--sizes)
            VECTOR_SIZES="$2"
            shift 2
            ;;
        -k|--kernel)
            KERNELS="$2"
            shift 2
            ;;
        -i|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Main execution
print_header
check_requirements
gather_system_info
set_cpu_governor

echo ""
echo -e "${GREEN}Starting benchmark suite...${NC}"
echo -e "${GREEN}Results will be saved to: $RESULTS_DIR${NC}"
echo ""

# Run requested benchmarks
case "$RUN_MODE" in
    scaling)
        run_scaling_study
        ;;
    size)
        run_size_study
        ;;
    all)
        run_scaling_study
        run_size_study
        ;;
esac

# Post-processing
combine_results
generate_report

echo ""
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}  Benchmark Suite Completed!${NC}"
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}Results directory: $RESULTS_DIR${NC}"
echo ""
echo "Next steps:"
echo "1. Review the performance report: ${RESULTS_DIR}/performance_report.md"
echo "2. Analyze CSV data: ${RESULTS_DIR}/combined_results.csv"
echo "3. Generate visualizations with Python script:"
echo "   python roofline_visualizer.py ${RESULTS_DIR}/combined_results.csv"
echo ""
