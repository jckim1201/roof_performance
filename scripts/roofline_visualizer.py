#!/usr/bin/env python3
"""
roofline_visualizer.py
Visualization and analysis tool for Roofline performance data
Generates Roofline plots, performance comparisons, and detailed reports
"""

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

@dataclass
class SystemSpec:
    """System specifications for Roofline model"""
    peak_flops: float  # Peak FLOPS in GFLOPS
    peak_bandwidth: float  # Peak bandwidth in GB/s
    cache_sizes: Dict[str, float]  # Cache sizes in MB
    name: str = "System"
    
    @property
    def ridge_point(self) -> float:
        """Calculate the ridge point (balance point) of the system"""
        return self.peak_flops / self.peak_bandwidth

class RooflineAnalyzer:
    """Main class for Roofline analysis and visualization"""
    
    def __init__(self, system_spec: SystemSpec):
        self.system = system_spec
        self.data = []
        self.df = None
        
    def load_json_data(self, filepath: str):
        """Load benchmark data from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.data.append(data)
            
    def load_csv_data(self, filepath: str):
        """Load benchmark data from CSV file"""
        self.df = pd.read_csv(filepath)
        
    def plot_roofline(self, output_file: str = None, show_kernels: bool = True):
        """Generate the Roofline plot"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Arithmetic Intensity range
        ai_range = np.logspace(-2, 2, 1000)
        
        # Roofline boundaries
        memory_bound = self.system.peak_bandwidth * ai_range
        compute_bound = np.ones_like(ai_range) * self.system.peak_flops
        roofline = np.minimum(memory_bound, compute_bound)
        
        # Plot roofline
        ax.loglog(ai_range, roofline, 'k-', linewidth=2, label='Roofline')
        ax.fill_between(ai_range, 0.001, roofline, alpha=0.1, color='blue')
        
        # Add ridge point
        ridge_ai = self.system.ridge_point
        ax.axvline(x=ridge_ai, color='red', linestyle='--', alpha=0.5, 
                  label=f'Ridge Point (AI={ridge_ai:.2f})')
        
        # Plot kernel performance points
        if show_kernels and self.df is not None:
            for kernel in self.df['kernel'].unique():
                kernel_data = self.df[self.df['kernel'] == kernel]
                ai = kernel_data['arithmetic_intensity'].values[0]
                gflops = kernel_data['gflops_peak'].values
                ax.scatter(ai, gflops, s=100, label=kernel, marker='o', zorder=5)
                
                # Add efficiency annotation
                theoretical = min(self.system.peak_bandwidth * ai, self.system.peak_flops)
                efficiency = (gflops[0] / theoretical) * 100
                ax.annotate(f'{efficiency:.1f}%', 
                          xy=(ai, gflops[0]), 
                          xytext=(5, 5), 
                          textcoords='offset points',
                          fontsize=8)
        
        # Formatting
        ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12)
        ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
        ax.set_title(f'Roofline Model - {self.system.name}', fontsize=14, fontweight='bold')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend(loc='best')
        
        # Add system specs text
        specs_text = f'Peak FLOPS: {self.system.peak_flops:.1f} GFLOPS\n'
        specs_text += f'Peak BW: {self.system.peak_bandwidth:.1f} GB/s'
        ax.text(0.98, 0.02, specs_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Roofline plot saved to {output_file}")
        else:
            plt.show()
            
    def plot_performance_comparison(self, output_file: str = None):
        """Generate performance comparison plots"""
        if self.df is None:
            print("No data loaded for comparison")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. GFLOPS comparison
        ax = axes[0, 0]
        kernels = self.df['kernel'].values
        gflops = self.df['gflops_peak'].values
        bars = ax.bar(kernels, gflops, color='steelblue')
        ax.set_ylabel('GFLOPS', fontsize=11)
        ax.set_title('Peak Performance Comparison', fontsize=12, fontweight='bold')
        ax.axhline(y=self.system.peak_flops, color='r', linestyle='--', 
                  alpha=0.5, label='System Peak')
        
        # Add value labels on bars
        for bar, val in zip(bars, gflops):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Bandwidth utilization
        ax = axes[0, 1]
        bandwidth = self.df['bandwidth_peak_GB_s'].values
        bars = ax.bar(kernels, bandwidth, color='coral')
        ax.set_ylabel('Bandwidth (GB/s)', fontsize=11)
        ax.set_title('Memory Bandwidth Utilization', fontsize=12, fontweight='bold')
        ax.axhline(y=self.system.peak_bandwidth, color='r', linestyle='--', 
                  alpha=0.5, label='System Peak')
        
        for bar, val in zip(bars, bandwidth):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Arithmetic Intensity
        ax = axes[1, 0]
        ai = self.df['arithmetic_intensity'].values
        bars = ax.bar(kernels, ai, color='forestgreen')
        ax.set_ylabel('Arithmetic Intensity (FLOP/Byte)', fontsize=11)
        ax.set_title('Arithmetic Intensity by Kernel', fontsize=12, fontweight='bold')
        ax.axhline(y=self.system.ridge_point, color='r', linestyle='--', 
                  alpha=0.5, label='Ridge Point')
        ax.set_yscale('log')
        
        # 4. Efficiency
        ax = axes[1, 1]
        efficiency = []
        for idx, row in self.df.iterrows():
            ai = row['arithmetic_intensity']
            achieved = row['gflops_peak']
            theoretical = min(self.system.peak_bandwidth * ai, self.system.peak_flops)
            eff = (achieved / theoretical) * 100 if theoretical > 0 else 0
            efficiency.append(eff)
        
        bars = ax.bar(kernels, efficiency, color='mediumpurple')
        ax.set_ylabel('Efficiency (%)', fontsize=11)
        ax.set_title('Roofline Efficiency', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 110])
        
        for bar, val in zip(bars, efficiency):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Adjust layout
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
            if ax.get_legend():
                ax.legend()
        
        plt.suptitle(f'Performance Analysis - {self.system.name}', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {output_file}")
        else:
            plt.show()
            
    def plot_scaling_analysis(self, thread_counts: List[int], output_file: str = None):
        """Plot scaling analysis across different thread counts"""
        if self.df is None or 'threads' not in self.df.columns:
            print("No thread scaling data available")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Strong scaling plot
        ax = axes[0]
        for kernel in self.df['kernel'].unique():
            kernel_data = self.df[self.df['kernel'] == kernel]
            threads = kernel_data['threads'].values
            speedup = kernel_data['gflops_peak'].values / kernel_data['gflops_peak'].values[0]
            ax.plot(threads, speedup, marker='o', label=kernel, linewidth=2)
        
        # Ideal scaling line
        ideal_threads = np.array(thread_counts)
        ax.plot(ideal_threads, ideal_threads / ideal_threads[0], 
               'k--', label='Ideal', alpha=0.5)
        
        ax.set_xlabel('Number of Threads', fontsize=11)
        ax.set_ylabel('Speedup', fontsize=11)
        ax.set_title('Strong Scaling Analysis', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Efficiency plot
        ax = axes[1]
        for kernel in self.df['kernel'].unique():
            kernel_data = self.df[self.df['kernel'] == kernel]
            threads = kernel_data['threads'].values
            speedup = kernel_data['gflops_peak'].values / kernel_data['gflops_peak'].values[0]
            efficiency = (speedup / (threads / threads[0])) * 100
            ax.plot(threads, efficiency, marker='s', label=kernel, linewidth=2)
        
        ax.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Perfect')
        ax.set_xlabel('Number of Threads', fontsize=11)
        ax.set_ylabel('Parallel Efficiency (%)', fontsize=11)
        ax.set_title('Parallel Efficiency', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 110])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Scaling Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Scaling plot saved to {output_file}")
        else:
            plt.show()
            
    def generate_report(self, output_file: str = "performance_report.txt"):
        """Generate detailed text report"""
        if self.df is None:
            print("No data loaded for report generation")
            return
            
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ROOFLINE PERFORMANCE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # System specifications
            f.write("SYSTEM SPECIFICATIONS\n")
            f.write("-" * 40 + "\n")
            f.write(f"System Name: {self.system.name}\n")
            f.write(f"Peak FLOPS: {self.system.peak_flops:.2f} GFLOPS\n")
            f.write(f"Peak Bandwidth: {self.system.peak_bandwidth:.2f} GB/s\n")
            f.write(f"Ridge Point (AI): {self.system.ridge_point:.3f} FLOP/Byte\n")
            if self.system.cache_sizes:
                f.write("\nCache Hierarchy:\n")
                for cache, size in self.system.cache_sizes.items():
                    f.write(f"  {cache}: {size:.2f} MB\n")
            f.write("\n")
            
            # Kernel performance summary
            f.write("KERNEL PERFORMANCE SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            for kernel in self.df['kernel'].unique():
                kernel_data = self.df[self.df['kernel'] == kernel].iloc[0]
                
                f.write(f"\n{kernel}:\n")
                f.write(f"  Performance: {kernel_data['gflops_peak']:.2f} GFLOPS\n")
                f.write(f"  Bandwidth: {kernel_data['bandwidth_peak_GB_s']:.2f} GB/s\n")
                f.write(f"  Arithmetic Intensity: {kernel_data['arithmetic_intensity']:.3f} FLOP/Byte\n")
                
                # Calculate efficiency
                ai = kernel_data['arithmetic_intensity']
                achieved = kernel_data['gflops_peak']
                theoretical = min(self.system.peak_bandwidth * ai, self.system.peak_flops)
                efficiency = (achieved / theoretical) * 100 if theoretical > 0 else 0
                
                f.write(f"  Roofline Efficiency: {efficiency:.1f}%\n")
                
                # Determine bottleneck
                if ai < self.system.ridge_point:
                    bottleneck = "Memory Bandwidth"
                    utilization = (kernel_data['bandwidth_peak_GB_s'] / self.system.peak_bandwidth) * 100
                    f.write(f"  Bottleneck: {bottleneck} ({utilization:.1f}% utilization)\n")
                else:
                    bottleneck = "Compute"
                    utilization = (kernel_data['gflops_peak'] / self.system.peak_flops) * 100
                    f.write(f"  Bottleneck: {bottleneck} ({utilization:.1f}% utilization)\n")
                
                # Timing statistics
                f.write(f"  Timing (median): {kernel_data['median_time']:.6f} seconds\n")
                if 'stddev_time' in kernel_data:
                    cv = (kernel_data['stddev_time'] / kernel_data['mean_time']) * 100
                    f.write(f"  Coefficient of Variation: {cv:.2f}%\n")
            
            # Recommendations
            f.write("\n" + "=" * 80 + "\n")
            f.write("OPTIMIZATION RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            
            for kernel in self.df['kernel'].unique():
                kernel_data = self.df[self.df['kernel'] == kernel].iloc[0]
                ai = kernel_data['arithmetic_intensity']
                efficiency = (kernel_data['gflops_peak'] / 
                            min(self.system.peak_bandwidth * ai, self.system.peak_flops)) * 100
                
                f.write(f"\n{kernel}:\n")
                
                if efficiency < 50:
                    f.write("  - Low efficiency detected. Consider:\n")
                    if ai < self.system.ridge_point:
                        f.write("    * Improving data reuse and cache utilization\n")
                        f.write("    * Using data blocking/tiling techniques\n")
                        f.write("    * Prefetching and stream optimization\n")
                    else:
                        f.write("    * Vectorization improvements\n")
                        f.write("    * Loop unrolling\n")
                        f.write("    * Instruction-level parallelism\n")
                elif efficiency < 80:
                    f.write("  - Moderate efficiency. Fine-tuning opportunities:\n")
                    f.write("    * NUMA optimization\n")
                    f.write("    * Memory alignment\n")
                    f.write("    * Thread affinity tuning\n")
                else:
                    f.write("  - Good efficiency achieved!\n")
            
            f.write("\n" + "=" * 80 + "\n")
            
        print(f"Report saved to {output_file}")

def detect_system_specs() -> SystemSpec:
    """Attempt to detect system specifications"""
    # This is a simplified example. In practice, you might want to:
    # - Parse /proc/cpuinfo on Linux
    # - Use lscpu command
    # - Query CPUID instructions
    # - Use external tools like likwid-topology
    
    # Default values for a typical modern system
    return SystemSpec(
        name="Generic x86-64 System",
        peak_flops=200.0,  # 200 GFLOPS (example)
        peak_bandwidth=50.0,  # 50 GB/s (example)
        cache_sizes={
            "L1": 0.256,  # 256 KB
            "L2": 2.0,    # 2 MB
            "L3": 16.0    # 16 MB
        }
    )

def main():
    parser = argparse.ArgumentParser(description='Roofline Model Visualization and Analysis')
    parser.add_argument('input', help='Input CSV or JSON file with benchmark results')
    parser.add_argument('--system-config', help='JSON file with system specifications')
    parser.add_argument('--peak-flops', type=float, help='System peak FLOPS (GFLOPS)')
    parser.add_argument('--peak-bandwidth', type=float, help='System peak bandwidth (GB/s)')
    parser.add_argument('--system-name', default='System', help='System name for plots')
    parser.add_argument('--output-dir', default='.', help='Output directory for plots')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--report-only', action='store_true', help='Generate only text report')
    
    args = parser.parse_args()
    
    # Load or detect system specifications
    if args.system_config:
        with open(args.system_config, 'r') as f:
            config = json.load(f)
            system = SystemSpec(**config)
    elif args.peak_flops and args.peak_bandwidth:
        system = SystemSpec(
            name=args.system_name,
            peak_flops=args.peak_flops,
            peak_bandwidth=args.peak_bandwidth,
            cache_sizes={}
        )
    else:
        print("Warning: No system specs provided, using defaults")
        system = detect_system_specs()
        system.name = args.system_name
    
    # Create analyzer
    analyzer = RooflineAnalyzer(system)
    
    # Load data
    input_path = Path(args.input)
    if input_path.suffix == '.json':
        analyzer.load_json_data(args.input)
    elif input_path.suffix == '.csv':
        analyzer.load_csv_data(args.input)
    else:
        print(f"Unsupported file format: {input_path.suffix}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate outputs
    if not args.no_plots and not args.report_only:
        analyzer.plot_roofline(output_dir / "roofline.png")
        analyzer.plot_performance_comparison(output_dir / "performance_comparison.png")
        # analyzer.plot_scaling_analysis([1, 2, 4, 8, 16], output_dir / "scaling.png")
    
    # Always generate report
    analyzer.generate_report(output_dir / "performance_report.txt")
    
    print(f"\nAnalysis complete! Results saved to {output_dir}/")

if __name__ == "__main__":
    main()
