# Roofline Performance Analysis Suite

A comprehensive CPU performance analysis tool based on the Roofline model for evaluating computational kernels and memory bandwidth utilization.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Building from Source](#building-from-source)
4. [Usage](#usage)
5. [Benchmark Kernels](#benchmark-kernels)
6. [Performance Metrics](#performance-metrics)
7. [Output Formats](#output-formats)
8. [Post-Processing and Visualization](#post-processing-and-visualization)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)
11. [Examples](#examples)

## Overview

The Roofline Performance Suite provides a systematic approach to analyze CPU performance by measuring both computational throughput (GFLOPS) and memory bandwidth utilization. It implements multiple benchmark kernels with varying arithmetic intensities to help identify performance bottlenecks and optimization opportunities.

### Key Features

- Six different computational kernels (AXPY, TRIAD, UPDATE3, DOT, DGEMV, STENCIL5)
- Statistical analysis with multiple runs
- Cache-aware performance modeling
- Multi-threaded execution with OpenMP
- Support for multiple compilers (NVFORTRAN, GNU Fortran, Intel Fortran)
- Automated result visualization and reporting

## Installation

### Prerequisites

- Fortran compiler (one of the following):
  - GNU Fortran (gfortran) version 7.0 or later
  - NVIDIA HPC SDK (nvfortran) version 20.0 or later
  - Intel oneAPI (ifx) or Intel Fortran Classic (ifort)
- OpenMP support
- Python 3.6+ (for visualization, optional)
- Make utility

### Python Dependencies (for visualization)

```bash
pip install numpy matplotlib pandas seaborn
```

## Building from Source

### Quick Build

```bash
# Clone or extract the source code
# Navigate to the project directory

# Build with GNU Fortran (default)
make

# Build with NVIDIA Fortran
make FC=nvfortran

# Build with Intel Fortran
make FC=ifx
```

### Alternative Build Methods

#### Direct Compilation

```bash
# Create necessary directories
mkdir -p build bin

# NVFORTRAN
cd build
nvfortran -O3 -mp=multicore -module . ../src/roofline_performance_suite_nvfortran.f90 -o ../bin/roofline_suite
cd ..

# GNU Fortran
cd build
gfortran -O3 -fopenmp -march=native -J. ../src/roofline_performance_suite_nvfortran.f90 -o ../bin/roofline_suite
cd ..

# Intel Fortran
cd build
ifx -O3 -qopenmp -xHost -module . ../src/roofline_performance_suite_nvfortran.f90 -o ../bin/roofline_suite
cd ..
```

#### Using Build Script

```bash
chmod +x build.sh
./build.sh nvfortran  # or gfortran, ifx
```

### Debug Build

```bash
make debug
```

## Usage

### Basic Command Structure

```bash
./bin/roofline_suite [options]
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--N=<size>` | Vector/problem size | 100000000 |
| `--M=<size>` | Matrix dimension for DGEMV | 1000 |
| `--iters=<n>` | Number of iterations per measurement | 5 |
| `--warmup=<n>` | Number of warmup iterations | 2 |
| `--stats=<n>` | Number of statistical runs | 5 |
| `--kernel=<name>` | Kernel to run (axpy/triad/update3/dot/dgemv/stencil5/all) | all |
| `--output=<format>` | Output format (text/json/csv) | text |
| `--outfile=<file>` | Output filename | roofline_results.csv |
| `--cache=<model>` | Cache model (naive/realistic) | realistic |
| `--validate` | Enable result validation | disabled |
| `--help` | Show help message | - |

### Quick Test

```bash
# Simple functionality test
./bin/roofline_suite --N=1000000 --iters=2 --warmup=1 --stats=2

# Or using make
make test
```

### Full Benchmark

```bash
# Run all kernels with default settings
./bin/roofline_suite

# Run all kernels with CSV output
./bin/roofline_suite --output=csv --outfile=results.csv

# Or using make
make benchmark
```

## Benchmark Kernels

### 1. AXPY (Y = alpha * X + Y)
- **Operations**: 2 FLOP per element (1 multiply, 1 add)
- **Memory**: 2 loads + 1 store (24 bytes per element)
- **Arithmetic Intensity**: 0.083 FLOP/byte
- **Typical bottleneck**: Memory bandwidth

### 2. TRIAD (A = B + scalar * C)
- **Operations**: 2 FLOP per element
- **Memory**: 2 loads + 1 store (24 bytes per element)
- **Arithmetic Intensity**: 0.083 FLOP/byte
- **Note**: Classic STREAM benchmark kernel

### 3. UPDATE3 (A = alpha * B + beta * C + D)
- **Operations**: 4 FLOP per element (2 multiply, 2 add)
- **Memory**: 3 loads + 1 store (32 bytes per element)
- **Arithmetic Intensity**: 0.125 FLOP/byte
- **Note**: Higher computational intensity

### 4. DOT (result = sum(X[i] * Y[i]))
- **Operations**: 2 FLOP per element
- **Memory**: 2 loads (16 bytes per element)
- **Arithmetic Intensity**: 0.125 FLOP/byte
- **Note**: Reduction pattern, tests accumulation

### 5. DGEMV (Y = alpha * A * X + beta * Y)
- **Operations**: 2N² FLOP for N×N matrix
- **Memory**: N² + 2N elements accessed
- **Arithmetic Intensity**: ~2.0 FLOP/byte for large N
- **Note**: Matrix-vector multiplication

### 6. STENCIL5 (5-point 1D stencil)
- **Operations**: 4 FLOP per element (4 additions)
- **Memory**: Variable based on cache model
- **Arithmetic Intensity**: 0.2-0.25 FLOP/byte
- **Note**: Tests spatial locality and cache reuse

## Performance Metrics

### Primary Metrics

1. **GFLOPS (Giga Floating-Point Operations Per Second)**
   - Peak: Best observed performance
   - Mean: Average across all runs

2. **Memory Bandwidth (GB/s)**
   - Achieved memory throughput
   - Comparison with theoretical peak

3. **Arithmetic Intensity (AI)**
   - Ratio of floating-point operations to bytes transferred
   - Key parameter in Roofline model

4. **Efficiency (%)**
   - Percentage of theoretical peak performance achieved
   - Calculated based on Roofline model

### Statistical Measures

- Minimum execution time
- Maximum execution time
- Mean execution time
- Median execution time
- Standard deviation
- Coefficient of variation

## Output Formats

### 1. Text Output (Default)

Console output with formatted results:
```
------------------------------------------------------------
Running kernel: TRIAD
Min time        :     0.124536 s
Median time     :     0.125892 s
Mean time       :     0.126234 s
Std dev         :     0.001245 s
Peak GFLOPS/s   :   16.234
Mean GFLOPS/s   :   16.089
Peak BW (GB/s)  :  195.234
Mean BW (GB/s)  :  193.456
AI (FLOP/Byte)  :     0.083
------------------------------------------------------------
```

### 2. CSV Output

Structured data suitable for analysis:
```csv
kernel,N,threads,min_time,median_time,mean_time,stddev_time,gflops_peak,gflops_mean,bandwidth_peak_GB_s,bandwidth_mean_GB_s,arithmetic_intensity,flops_per_elem,bytes_per_elem
TRIAD,100000000,8,0.124536,0.125892,0.126234,0.001245,16.234,16.089,195.234,193.456,0.083,2.0,24.0
```

### 3. JSON Output

Machine-readable format for integration:
```json
{
  "kernel": "TRIAD",
  "N": 100000000,
  "num_runs": 5,
  "min_time_sec": 0.124536,
  "median_time_sec": 0.125892,
  "mean_time_sec": 0.126234,
  "stddev_time_sec": 0.001245,
  "gflops_peak": 16.234,
  "gflops_mean": 16.089,
  "bandwidth_GB_s_peak": 195.234,
  "bandwidth_GB_s_mean": 193.456,
  "arithmetic_intensity": 0.083,
  "flops_per_element": 2.0,
  "bytes_per_element": 24.0,
  "num_threads": 8
}
```

## Post-Processing and Visualization

### Using the Python Visualization Script

The suite includes a comprehensive Python script for analyzing and visualizing results.

#### Basic Visualization

```bash
# Generate Roofline plot from CSV results
python scripts/roofline_visualizer.py results.csv --peak-flops=200 --peak-bandwidth=50

# Specify system configuration
python scripts/roofline_visualizer.py results.csv --system-config=system.json
```

#### System Configuration File

Create a `system.json` file with your system specifications:
```json
{
  "name": "Intel Xeon Gold 6248",
  "peak_flops": 2457.6,
  "peak_bandwidth": 141.0,
  "cache_sizes": {
    "L1": 0.032,
    "L2": 1.0,
    "L3": 27.5
  }
}
```

#### Generated Visualizations

1. **Roofline Plot** (`roofline.png`)
   - Shows performance of each kernel relative to hardware limits
   - Identifies memory-bound vs compute-bound regions
   - Displays efficiency percentages

2. **Performance Comparison** (`performance_comparison.png`)
   - Bar charts comparing GFLOPS across kernels
   - Bandwidth utilization comparison
   - Arithmetic intensity visualization
   - Efficiency analysis

3. **Performance Report** (`performance_report.txt`)
   - Detailed text report with all metrics
   - Bottleneck analysis
   - Optimization recommendations

### Manual Analysis with Spreadsheets

CSV output can be imported into Excel or Google Sheets for custom analysis:

1. Import CSV file
2. Create pivot tables by kernel type
3. Calculate efficiency: `Achieved_GFLOPS / MIN(AI * Peak_BW, Peak_FLOPS)`
4. Plot performance trends

### Automated Batch Processing

```bash
#!/bin/bash
# batch_analysis.sh

# Run benchmarks for different problem sizes
for size in 1000000 10000000 100000000; do
    ./bin/roofline_suite --N=$size --output=csv --outfile=results_N${size}.csv
done

# Combine results
echo "kernel,N,threads,min_time,median_time,mean_time,stddev_time,gflops_peak,gflops_mean,bandwidth_peak_GB_s,bandwidth_mean_GB_s,arithmetic_intensity,flops_per_elem,bytes_per_elem" > combined_results.csv
for file in results_N*.csv; do
    tail -n +2 $file >> combined_results.csv
done

# Generate visualization
python scripts/roofline_visualizer.py combined_results.csv
```

## Advanced Features

### 1. Cache Model Selection

The suite provides two cache models for memory traffic estimation:

#### Realistic Model (Default)
- Accounts for cache line reuse
- Considers spatial locality
- More accurate for cached operations

```bash
./bin/roofline_suite --kernel=stencil5 --cache=realistic
```

#### Naive Model
- Counts all memory operations
- Ignores cache effects
- Useful for worst-case analysis

```bash
./bin/roofline_suite --kernel=stencil5 --cache=naive
```

### 2. Thread Scaling Analysis

```bash
#!/bin/bash
# scaling_study.sh

for threads in 1 2 4 8 16; do
    export OMP_NUM_THREADS=$threads
    ./bin/roofline_suite --output=csv --outfile=results_t${threads}.csv
done
```

### 3. NUMA Optimization

```bash
# Bind to specific NUMA node
numactl --cpunodebind=0 --membind=0 ./bin/roofline_suite

# Interleave memory across nodes
numactl --interleave=all ./bin/roofline_suite
```

### 4. CPU Frequency Control

```bash
# Set performance governor (requires root)
sudo cpupower frequency-set -g performance

# Run benchmark
./bin/roofline_suite

# Restore default governor
sudo cpupower frequency-set -g powersave
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Module File Creation Error (NVFORTRAN)

**Error**: `Unable to create MODULE file`

**Solution**:
```bash
# Ensure build directory exists and compile from there
mkdir -p build
cd build
nvfortran -O3 -mp=multicore -module . ../src/roofline_performance_suite_nvfortran.f90 -o ../bin/roofline_suite
```

#### 2. OpenMP Not Working

**Symptoms**: Only one thread used despite setting OMP_NUM_THREADS

**Solutions**:
```bash
# Check OpenMP support
echo $OMP_NUM_THREADS
ldd bin/roofline_suite | grep omp

# Set OpenMP environment
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true
export OMP_PLACES=cores
```

#### 3. Performance Lower Than Expected

**Possible causes and solutions**:

1. CPU frequency scaling:
   ```bash
   # Check current frequency
   grep "cpu MHz" /proc/cpuinfo
   
   # Set to maximum frequency
   sudo cpupower frequency-set -g performance
   ```

2. Memory allocation issues:
   ```bash
   # Use large pages
   export OMP_STACKSIZE=256M
   ```

3. NUMA effects:
   ```bash
   # Check NUMA topology
   numactl --hardware
   
   # Run with NUMA binding
   numactl --cpunodebind=0 --membind=0 ./bin/roofline_suite
   ```

#### 4. Validation Failures

**Error**: `AXPY validation failed`

**Causes**:
- Numerical precision issues with large N
- Compiler optimization bugs

**Solutions**:
```bash
# Try with smaller problem size
./bin/roofline_suite --validate --N=1000000

# Compile with lower optimization
make clean
make FC=gfortran OPT_FLAGS="-O2"
```

## Examples

### Example 1: Basic Performance Assessment

```bash
# Run all kernels with moderate problem size
./bin/roofline_suite --N=50000000 --stats=10 --output=csv --outfile=baseline.csv

# Visualize results
python scripts/roofline_visualizer.py baseline.csv --peak-flops=100 --peak-bandwidth=50
```

### Example 2: Memory Bandwidth Focus

```bash
# Test memory-bound kernels with large arrays
./bin/roofline_suite --kernel=triad --N=500000000 --stats=5
./bin/roofline_suite --kernel=axpy --N=500000000 --stats=5
```

### Example 3: Compute-Intensive Analysis

```bash
# Test compute-bound kernel (DGEMV) with various matrix sizes
for size in 500 1000 2000 4000; do
    ./bin/roofline_suite --kernel=dgemv --M=$size --output=csv --outfile=dgemv_${size}.csv
done
```

### Example 4: Cache Effects Study

```bash
# Compare cache models for stencil operation
./bin/roofline_suite --kernel=stencil5 --cache=realistic --output=json --outfile=stencil_realistic.json
./bin/roofline_suite --kernel=stencil5 --cache=naive --output=json --outfile=stencil_naive.json

# Compare results
echo "Realistic model:"
grep "bandwidth_GB_s_peak" stencil_realistic.json
echo "Naive model:"
grep "bandwidth_GB_s_peak" stencil_naive.json
```

### Example 5: Full System Characterization

```bash
#!/bin/bash
# full_characterization.sh

# Create results directory
RESULTS_DIR="results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

# System information
echo "System Information" > $RESULTS_DIR/system_info.txt
lscpu >> $RESULTS_DIR/system_info.txt
free -h >> $RESULTS_DIR/system_info.txt

# Run comprehensive benchmark
problem_sizes=(1000000 10000000 100000000)
thread_counts=(1 2 4 8)

for size in ${problem_sizes[@]}; do
    for threads in ${thread_counts[@]}; do
        export OMP_NUM_THREADS=$threads
        ./bin/roofline_suite --N=$size --stats=10 \
            --output=csv --outfile=$RESULTS_DIR/results_N${size}_T${threads}.csv
    done
done

# Combine all results
cat $RESULTS_DIR/results_N*_T*.csv | head -n 1 > $RESULTS_DIR/all_results.csv
for file in $RESULTS_DIR/results_N*_T*.csv; do
    tail -n +2 $file >> $RESULTS_DIR/all_results.csv
done

# Generate report
echo "Benchmark completed. Results in $RESULTS_DIR/"
echo "To visualize: python scripts/roofline_visualizer.py $RESULTS_DIR/all_results.csv"
```

## Performance Tuning Guidelines

### Compiler Optimization Flags

#### GNU Fortran
```bash
# Aggressive optimization
FFLAGS="-O3 -march=native -mtune=native -funroll-loops -ftree-vectorize"

# With profile-guided optimization
# Step 1: Generate profile
FFLAGS="-O3 -march=native -fprofile-generate"
make clean && make
./bin/roofline_suite --N=10000000

# Step 2: Use profile
FFLAGS="-O3 -march=native -fprofile-use"
make clean && make
```

#### Intel Fortran
```bash
# AVX-512 optimization
FFLAGS="-O3 -xCORE-AVX512 -qopt-zmm-usage=high"

# With Intel MKL
FFLAGS="-O3 -xHost -qmkl=parallel"
```

#### NVIDIA Fortran
```bash
# GPU offloading (experimental)
FFLAGS="-O3 -mp=gpu -gpu=cc80"

# CPU optimization with detailed feedback
FFLAGS="-O3 -mp=multicore -Minfo=all -Mvect=simd:256"
```

### OpenMP Tuning

```bash
# Thread affinity
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Schedule tuning
export OMP_SCHEDULE="static,1000"

# Stack size for large arrays
export OMP_STACKSIZE=512M
```

### System-Level Optimizations

```bash
# Disable hyperthreading for consistent performance
echo off | sudo tee /sys/devices/system/cpu/smt/control

# Set CPU frequency governor
sudo cpupower frequency-set -g performance

# Clear page cache before benchmarking
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

# Disable NUMA balancing
echo 0 | sudo tee /proc/sys/kernel/numa_balancing
```

## Contributing

To contribute to the Roofline Performance Suite:

1. Fork the repository
2. Create a feature branch
3. Add your kernel or enhancement
4. Ensure all tests pass
5. Submit a pull request

### Adding New Kernels

To add a new kernel, modify the following files:

1. Add kernel subroutine in `kernels_mod` module
2. Add kernel ID in `constants_mod`
3. Update `execute_kernel` subroutine
4. Add metrics in `get_kernel_metrics`
5. Update documentation

## References

1. Williams, S., Waterman, A., & Patterson, D. (2009). "Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures"
2. McCalpin, J. D. (1995). "Memory Bandwidth and Machine Balance in Current High Performance Computers"
3. Intel Advisor Roofline Analysis Documentation
4. NVIDIA Nsight Compute Roofline Analysis

## License

MIT License - See LICENSE file for details

## Contact

For questions, issues, or contributions, please open an issue on the project repository.
