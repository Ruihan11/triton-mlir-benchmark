#!/usr/bin/env python3
"""
Visualization script for MLIR vs Triton Flash Attention benchmark results
Creates comprehensive performance charts and analysis
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BenchmarkVisualizer:
    """Visualize benchmark results with comprehensive charts"""
    
    def __init__(self, results_file: str = "results/mlir_vs_triton_benchmark_results_v3.json"):
        """Initialize visualizer with results file"""
        self.results_file = Path(results_file)
        self.results = None
        self.df = None
        self.load_results()
        self.prepare_dataframe()
        
    def load_results(self):
        """Load benchmark results from JSON file"""
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        with open(self.results_file, 'r') as f:
            self.results = json.load(f)
        
        print(f"Loaded results from: {self.results_file}")
        print(f"Device: {self.results['metadata']['device']}")
        print(f"Timestamp: {self.results['metadata']['timestamp']}")
        
    def prepare_dataframe(self):
        """Convert results to pandas DataFrame for easier analysis"""
        data = []
        
        for benchmark in self.results['benchmarks']:
            config = benchmark['configuration']
            
            for impl_name, impl_data in benchmark['implementations'].items():
                if 'mean_ms' in impl_data:
                    row = {
                        'batch_size': config['batch_size'],
                        'seq_len': config['seq_len'],
                        'num_heads': config['num_heads'],
                        'head_dim': config['head_dim'],
                        'implementation': impl_name,
                        'mean_ms': impl_data['mean_ms'],
                        'std_ms': impl_data['std_ms'],
                        'min_ms': impl_data['min_ms'],
                        'max_ms': impl_data['max_ms'],
                        'median_ms': impl_data['median_ms'],
                        'p95_ms': impl_data['p95_ms'],
                        'p99_ms': impl_data['p99_ms'],
                        'memory_mb': impl_data['peak_memory_mb'],
                        'speedup': benchmark['speedups'].get(impl_name, 1.0)
                    }
                    data.append(row)
        
        self.df = pd.DataFrame(data)
        
        # Add readable implementation names
        impl_names = {
            'standard_triton': 'Standard Triton',
            'mlir_v1_fusion': 'MLIR V1 (Loop Fusion)',
            'mlir_v2_coalescing': 'MLIR V2 (Memory Coalescing)',
            'mlir_v3_advanced': 'MLIR V3 (Advanced)'
        }
        self.df['impl_display'] = self.df['implementation'].map(impl_names)
        
    def plot_performance_comparison(self, save_path: Optional[str] = None):
        """Create comprehensive performance comparison chart"""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Average speedup by implementation
        ax1 = fig.add_subplot(gs[0, 0])
        avg_speedup = self.df.groupby('impl_display')['speedup'].mean().sort_values()
        bars = ax1.barh(avg_speedup.index, avg_speedup.values)
        
        # Color bars based on speedup
        colors = ['red' if x < 1 else 'green' if x > 1.2 else 'orange' for x in avg_speedup.values]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax1.axvline(x=1, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Average Speedup')
        ax1.set_title('Overall Speedup vs Standard Triton')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (idx, val) in enumerate(avg_speedup.items()):
            ax1.text(val + 0.02, i, f'{val:.2f}x', va='center')
        
        # 2. Performance by sequence length
        ax2 = fig.add_subplot(gs[0, 1])
        for impl in self.df['impl_display'].unique():
            impl_data = self.df[self.df['impl_display'] == impl]
            avg_by_seq = impl_data.groupby('seq_len')['mean_ms'].mean()
            ax2.plot(avg_by_seq.index, avg_by_seq.values, marker='o', label=impl, linewidth=2)
        
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Time (ms)')
        ax2.set_title('Performance vs Sequence Length')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 3. Performance by batch size
        ax3 = fig.add_subplot(gs[0, 2])
        for impl in self.df['impl_display'].unique():
            impl_data = self.df[self.df['impl_display'] == impl]
            avg_by_batch = impl_data.groupby('batch_size')['mean_ms'].mean()
            ax3.plot(avg_by_batch.index, avg_by_batch.values, marker='s', label=impl, linewidth=2)
        
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Time (ms)')
        ax3.set_title('Performance vs Batch Size')
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Memory usage comparison
        ax4 = fig.add_subplot(gs[1, 0])
        memory_data = self.df.groupby('impl_display')['memory_mb'].mean().sort_values()
        bars = ax4.barh(memory_data.index, memory_data.values)
        
        # Color based on memory efficiency
        max_mem = memory_data.max()
        colors = ['green' if x < max_mem * 0.8 else 'orange' if x < max_mem * 0.95 else 'red' 
                 for x in memory_data.values]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax4.set_xlabel('Memory Usage (MB)')
        ax4.set_title('Average Memory Usage')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (idx, val) in enumerate(memory_data.items()):
            ax4.text(val + 1, i, f'{val:.1f} MB', va='center')
        
        # 5. Speedup heatmap
        ax5 = fig.add_subplot(gs[1, 1:])
        
        # Create pivot table for best implementation speedup
        best_speedup = self.df.groupby(['batch_size', 'seq_len'])['speedup'].max().reset_index()
        pivot = best_speedup.pivot(index='batch_size', columns='seq_len', values='speedup')
        
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=1.0,
                   ax=ax5, cbar_kws={'label': 'Speedup'}, vmin=0.5, vmax=2.0)
        ax5.set_title('Maximum Speedup Heatmap (Batch Size vs Sequence Length)')
        ax5.set_xlabel('Sequence Length')
        ax5.set_ylabel('Batch Size')
        
        # 6. Performance variance (stability)
        ax6 = fig.add_subplot(gs[2, 0])
        variance_data = self.df.groupby('impl_display')['std_ms'].mean().sort_values()
        bars = ax6.barh(variance_data.index, variance_data.values)
        
        # Color based on stability
        colors = ['green' if x < variance_data.mean() else 'orange' for x in variance_data.values]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax6.set_xlabel('Std Dev (ms)')
        ax6.set_title('Performance Stability (Lower is Better)')
        ax6.grid(True, alpha=0.3)
        
        # 7. Scaling efficiency
        ax7 = fig.add_subplot(gs[2, 1])
        
        # Calculate scaling efficiency
        base_seq = 64
        for impl in self.df['impl_display'].unique():
            impl_data = self.df[self.df['impl_display'] == impl]
            scaling = []
            seq_lens = sorted(impl_data['seq_len'].unique())
            
            for seq_len in seq_lens:
                time_at_seq = impl_data[impl_data['seq_len'] == seq_len]['mean_ms'].mean()
                time_at_base = impl_data[impl_data['seq_len'] == base_seq]['mean_ms'].mean()
                theoretical_scaling = seq_len / base_seq
                actual_scaling = time_at_seq / time_at_base
                efficiency = theoretical_scaling / actual_scaling if actual_scaling > 0 else 0
                scaling.append(efficiency)
            
            ax7.plot(seq_lens, scaling, marker='o', label=impl, linewidth=2)
        
        ax7.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax7.set_xlabel('Sequence Length')
        ax7.set_ylabel('Scaling Efficiency')
        ax7.set_title('Scaling Efficiency (1.0 = Perfect Linear Scaling)')
        ax7.legend(loc='best', fontsize=8)
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim([0, 1.2])
        
        # 8. Best implementation by configuration
        ax8 = fig.add_subplot(gs[2, 2])
        
        # Find best implementation for each configuration
        best_impl_counts = {}
        for _, group in self.df.groupby(['batch_size', 'seq_len']):
            best_impl = group.loc[group['speedup'].idxmax()]['impl_display']
            best_impl_counts[best_impl] = best_impl_counts.get(best_impl, 0) + 1
        
        # Create pie chart
        if best_impl_counts:
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            wedges, texts, autotexts = ax8.pie(best_impl_counts.values(), 
                                                labels=best_impl_counts.keys(),
                                                autopct='%1.1f%%',
                                                colors=colors[:len(best_impl_counts)])
            ax8.set_title('Best Implementation Distribution\n(% of configurations where each is fastest)')
        
        # Main title
        fig.suptitle(f'MLIR vs Triton Flash Attention Performance Analysis\n'
                    f'Device: {self.results["metadata"]["device"]}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved performance comparison to: {save_path}")
        
        plt.show()
        
    def plot_detailed_speedup(self, save_path: Optional[str] = None):
        """Create detailed speedup analysis charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Speedup by sequence length for each batch size
        ax = axes[0, 0]
        batch_sizes = sorted(self.df['batch_size'].unique())
        
        for batch_size in batch_sizes:
            batch_data = self.df[self.df['batch_size'] == batch_size]
            
            # Get MLIR V3 speedup
            v3_data = batch_data[batch_data['implementation'] == 'mlir_v3_advanced']
            if not v3_data.empty:
                speedup_by_seq = v3_data.groupby('seq_len')['speedup'].mean()
                ax.plot(speedup_by_seq.index, speedup_by_seq.values, 
                       marker='o', label=f'Batch {batch_size}', linewidth=2)
        
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Speedup')
        ax.set_title('MLIR V3 Speedup vs Sequence Length')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 2. Implementation comparison at different scales
        ax = axes[0, 1]
        
        # Group by total elements (batch * seq_len)
        self.df['total_elements'] = self.df['batch_size'] * self.df['seq_len']
        
        for impl in ['mlir_v1_fusion', 'mlir_v2_coalescing', 'mlir_v3_advanced']:
            impl_data = self.df[self.df['implementation'] == impl]
            if not impl_data.empty:
                speedup_by_size = impl_data.groupby('total_elements')['speedup'].mean()
                ax.plot(speedup_by_size.index, speedup_by_size.values, 
                       marker='o', label=impl.replace('mlir_', 'MLIR ').replace('_', ' ').title(),
                       linewidth=2)
        
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Total Elements (Batch × Seq Length)')
        ax.set_ylabel('Speedup')
        ax.set_title('Speedup vs Problem Size')
        ax.set_xscale('log')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 3. Memory efficiency
        ax = axes[1, 0]
        
        # Calculate memory efficiency (throughput per MB)
        self.df['throughput_per_mb'] = (self.df['batch_size'] * self.df['seq_len'] * 
                                        self.df['num_heads'] * self.df['head_dim']) / \
                                       (self.df['mean_ms'] * self.df['memory_mb'])
        
        efficiency_data = self.df.groupby('impl_display')['throughput_per_mb'].mean().sort_values()
        bars = ax.bar(range(len(efficiency_data)), efficiency_data.values)
        
        # Color based on efficiency
        max_eff = efficiency_data.max()
        colors = ['green' if x > max_eff * 0.8 else 'orange' if x > max_eff * 0.6 else 'red' 
                 for x in efficiency_data.values]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xticks(range(len(efficiency_data)))
        ax.set_xticklabels(efficiency_data.index, rotation=45, ha='right')
        ax.set_ylabel('Elements/(ms × MB)')
        ax.set_title('Memory Efficiency (Higher is Better)')
        ax.grid(True, alpha=0.3)
        
        # 4. Performance consistency
        ax = axes[1, 1]
        
        # Calculate coefficient of variation (CV) for each implementation
        cv_data = []
        for impl in self.df['impl_display'].unique():
            impl_data = self.df[self.df['impl_display'] == impl]
            cv = (impl_data['std_ms'] / impl_data['mean_ms']).mean() * 100
            cv_data.append({'Implementation': impl, 'CV (%)': cv})
        
        cv_df = pd.DataFrame(cv_data).sort_values('CV (%)')
        bars = ax.bar(range(len(cv_df)), cv_df['CV (%)'].values)
        
        # Color based on consistency
        colors = ['green' if x < 5 else 'orange' if x < 10 else 'red' for x in cv_df['CV (%)'].values]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xticks(range(len(cv_df)))
        ax.set_xticklabels(cv_df['Implementation'].values, rotation=45, ha='right')
        ax.set_ylabel('Coefficient of Variation (%)')
        ax.set_title('Performance Consistency (Lower is Better)')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, val in enumerate(cv_df['CV (%)'].values):
            ax.text(i, val + 0.2, f'{val:.1f}%', ha='center')
        
        fig.suptitle('Detailed Speedup and Efficiency Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved detailed speedup analysis to: {save_path}")
        
        plt.show()
        
    def plot_performance_breakdown(self, save_path: Optional[str] = None):
        """Create performance breakdown charts"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Percentile comparison
        ax = axes[0, 0]
        percentiles = ['median_ms', 'p95_ms', 'p99_ms']
        x = np.arange(len(self.df['impl_display'].unique()))
        width = 0.25
        
        for i, percentile in enumerate(percentiles):
            data = self.df.groupby('impl_display')[percentile].mean().values
            label = percentile.replace('_ms', '').replace('median', 'P50').upper()
            ax.bar(x + i * width, data, width, label=label)
        
        ax.set_xlabel('Implementation')
        ax.set_xticks(x + width)
        ax.set_xticklabels(self.df['impl_display'].unique(), rotation=45, ha='right')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Latency Percentiles')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Min vs Max performance
        ax = axes[0, 1]
        impl_names = self.df['impl_display'].unique()
        min_times = self.df.groupby('impl_display')['min_ms'].mean().values
        max_times = self.df.groupby('impl_display')['max_ms'].mean().values
        
        x = np.arange(len(impl_names))
        width = 0.35
        
        ax.bar(x - width/2, min_times, width, label='Min', color='green', alpha=0.7)
        ax.bar(x + width/2, max_times, width, label='Max', color='red', alpha=0.7)
        
        ax.set_xlabel('Implementation')
        ax.set_xticks(x)
        ax.set_xticklabels(impl_names, rotation=45, ha='right')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Best vs Worst Case Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Speedup distribution
        ax = axes[0, 2]
        for impl in ['mlir_v1_fusion', 'mlir_v2_coalescing', 'mlir_v3_advanced']:
            impl_data = self.df[self.df['implementation'] == impl]
            if not impl_data.empty:
                ax.hist(impl_data['speedup'].values, bins=20, alpha=0.6, 
                       label=impl.replace('mlir_', 'MLIR ').replace('_', ' ').title())
        
        ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Speedup')
        ax.set_ylabel('Count')
        ax.set_title('Speedup Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Performance by configuration size
        ax = axes[1, 0]
        config_sizes = ['Small\n(B≤2, L≤128)', 'Medium\n(B≤4, L≤256)', 'Large\n(B>4 or L>256)']
        
        # Categorize configurations
        small_mask = (self.df['batch_size'] <= 2) & (self.df['seq_len'] <= 128)
        medium_mask = (self.df['batch_size'] <= 4) & (self.df['seq_len'] <= 256) & ~small_mask
        large_mask = ~small_mask & ~medium_mask
        
        speedup_by_size = []
        for mask in [small_mask, medium_mask, large_mask]:
            v3_speedups = self.df[mask & (self.df['implementation'] == 'mlir_v3_advanced')]['speedup']
            speedup_by_size.append(v3_speedups.mean() if not v3_speedups.empty else 0)
        
        bars = ax.bar(config_sizes, speedup_by_size)
        colors = ['green' if x > 1.2 else 'orange' if x > 1.0 else 'red' for x in speedup_by_size]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_ylabel('Average Speedup')
        ax.set_title('MLIR V3 Speedup by Configuration Size')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, val in enumerate(speedup_by_size):
            ax.text(i, val + 0.02, f'{val:.2f}x', ha='center')
        
        # 5. Time breakdown by operation
        ax = axes[1, 1]
        
        # Estimate relative costs (illustrative)
        operations = ['Memory\nAccess', 'Compute\n(GEMM)', 'Softmax', 'Other']
        standard_costs = [40, 35, 20, 5]  # Percentages for standard
        v3_costs = [25, 40, 25, 10]  # V3 optimizes memory but does more compute
        
        x = np.arange(len(operations))
        width = 0.35
        
        ax.bar(x - width/2, standard_costs, width, label='Standard', color='#FF6B6B')
        ax.bar(x + width/2, v3_costs, width, label='MLIR V3', color='#4ECDC4')
        
        ax.set_ylabel('Relative Cost (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(operations)
        ax.set_title('Estimated Operation Cost Breakdown')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Improvement summary
        ax = axes[1, 2]
        
        # Calculate improvements
        improvements = {
            'Avg Speedup': self.df[self.df['implementation'] == 'mlir_v3_advanced']['speedup'].mean(),
            'Memory\nReduction': 1 - (self.df[self.df['implementation'] == 'mlir_v3_advanced']['memory_mb'].mean() /
                                     self.df[self.df['implementation'] == 'standard_triton']['memory_mb'].mean()),
            'Stability\nImprovement': 1 - (self.df[self.df['implementation'] == 'mlir_v3_advanced']['std_ms'].mean() /
                                          self.df[self.df['implementation'] == 'standard_triton']['std_ms'].mean()),
        }
        
        # Convert to percentages where appropriate
        display_values = []
        for key, val in improvements.items():
            if 'Speedup' in key:
                display_values.append(val)
            else:
                display_values.append(val * 100)  # Convert to percentage
        
        bars = ax.bar(improvements.keys(), display_values)
        colors = ['green' if x > 0 else 'red' for x in display_values]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_ylabel('Improvement')
        ax.set_title('MLIR V3 vs Standard Summary')
        ax.grid(True, alpha=0.3)
        
        # Add value labels with appropriate formatting
        for i, (key, val) in enumerate(zip(improvements.keys(), display_values)):
            if 'Speedup' in key:
                ax.text(i, val + 0.02, f'{val:.2f}x', ha='center')
            else:
                ax.text(i, val + 1, f'{val:.1f}%', ha='center')
        
        fig.suptitle('Performance Breakdown Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved performance breakdown to: {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path: Optional[str] = None):
        """Generate a text report with key findings"""
        report = []
        report.append("=" * 80)
        report.append("MLIR vs TRITON FLASH ATTENTION - PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"\nDevice: {self.results['metadata']['device']}")
        report.append(f"CUDA Capability: {self.results['metadata']['cuda_capability']}")
        report.append(f"Timestamp: {self.results['metadata']['timestamp']}\n")
        
        # Summary statistics
        if 'summary' in self.results:
            summary = self.results['summary']
            
            report.append("EXECUTIVE SUMMARY")
            report.append("-" * 40)
            
            # Find best performing implementation
            best_impl = None
            best_speedup = 0
            for impl, perf in summary['implementation_performance'].items():
                if perf['avg_speedup'] > best_speedup:
                    best_speedup = perf['avg_speedup']
                    best_impl = impl
            
            report.append(f"✓ Best Overall Implementation: {best_impl}")
            report.append(f"  Average Speedup: {best_speedup:.2f}x")
            
            # Performance improvements
            v3_perf = summary['implementation_performance'].get('mlir_v3_advanced', {})
            if v3_perf:
                report.append(f"\n✓ MLIR V3 Performance:")
                report.append(f"  - Average Time: {v3_perf['avg_time_ms']:.2f} ms")
                report.append(f"  - Memory Usage: {v3_perf['avg_memory_mb']:.1f} MB")
                report.append(f"  - Speedup: {v3_perf['avg_speedup']:.2f}x")
            
            # Best configurations
            report.append("\nOPTIMAL CONFIGURATIONS")
            report.append("-" * 40)
            for impl, config_data in summary['best_configurations'].items():
                config = config_data['config']
                speedup = config_data['speedup']
                report.append(f"\n{impl}:")
                report.append(f"  Best at: Batch={config['batch_size']}, Seq={config['seq_len']}")
                report.append(f"  Speedup: {speedup:.2f}x")
        
        # Detailed analysis
        report.append("\nDETAILED ANALYSIS")
        report.append("-" * 40)
        
        # Speedup by configuration size
        small_speedup = self.df[(self.df['batch_size'] <= 2) & 
                                (self.df['seq_len'] <= 128) & 
                                (self.df['implementation'] == 'mlir_v3_advanced')]['speedup'].mean()
        large_speedup = self.df[(self.df['batch_size'] >= 4) & 
                                (self.df['seq_len'] >= 256) & 
                                (self.df['implementation'] == 'mlir_v3_advanced')]['speedup'].mean()
        
        report.append(f"\nPerformance Scaling:")
        report.append(f"  Small configs (B≤2, L≤128): {small_speedup:.2f}x speedup")
        report.append(f"  Large configs (B≥4, L≥256): {large_speedup:.2f}x speedup")
        
        # Memory efficiency
        std_mem = self.df[self.df['implementation'] == 'standard_triton']['memory_mb'].mean()
        v3_mem = self.df[self.df['implementation'] == 'mlir_v3_advanced']['memory_mb'].mean()
        mem_reduction = (1 - v3_mem/std_mem) * 100
        
        report.append(f"\nMemory Efficiency:")
        report.append(f"  Standard: {std_mem:.1f} MB average")
        report.append(f"  MLIR V3: {v3_mem:.1f} MB average")
        report.append(f"  Reduction: {mem_reduction:.1f}%")
        
        # Stability
        std_cv = (self.df[self.df['implementation'] == 'standard_triton']['std_ms'] / 
                 self.df[self.df['implementation'] == 'standard_triton']['mean_ms']).mean() * 100
        v3_cv = (self.df[self.df['implementation'] == 'mlir_v3_advanced']['std_ms'] / 
                self.df[self.df['implementation'] == 'mlir_v3_advanced']['mean_ms']).mean() * 100
        
        report.append(f"\nPerformance Stability:")
        report.append(f"  Standard CV: {std_cv:.1f}%")
        report.append(f"  MLIR V3 CV: {v3_cv:.1f}%")
        report.append(f"  Improvement: {std_cv - v3_cv:.1f} percentage points")
        
        # Recommendations
        report.append("\nRECOMMENDATIONS")
        report.append("-" * 40)
        
        if best_speedup > 1.5:
            report.append("✓ MLIR optimizations provide significant speedup (>1.5x)")
            report.append("  Recommended for production deployment")
        elif best_speedup > 1.2:
            report.append("✓ MLIR optimizations provide moderate speedup (>1.2x)")
            report.append("  Consider for performance-critical applications")
        else:
            report.append("⚠ MLIR optimizations provide limited speedup")
            report.append("  Evaluate based on specific use case requirements")
        
        if mem_reduction > 20:
            report.append("✓ Significant memory savings achieved")
            report.append("  Beneficial for memory-constrained environments")
        
        if v3_cv < std_cv:
            report.append("✓ Improved performance stability")
            report.append("  Better for latency-sensitive applications")
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Saved text report to: {save_path}")
        
        print(report_text)
        return report_text
    
    def create_all_visualizations(self, output_dir: str = "visualizations"):
        """Create all visualizations and save them"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Creating visualizations...")
        
        # 1. Main performance comparison
        print("1. Creating performance comparison chart...")
        self.plot_performance_comparison(
            save_path=output_path / "performance_comparison.png"
        )
        
        # 2. Detailed speedup analysis
        print("2. Creating detailed speedup analysis...")
        self.plot_detailed_speedup(
            save_path=output_path / "detailed_speedup.png"
        )
        
        # 3. Performance breakdown
        print("3. Creating performance breakdown...")
        self.plot_performance_breakdown(
            save_path=output_path / "performance_breakdown.png"
        )
        
        # 4. Generate text report
        print("4. Generating text report...")
        self.generate_report(
            save_path=output_path / "performance_report.txt"
        )
        
        print(f"\nAll visualizations saved to: {output_path}")
        
        return output_path


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize MLIR vs Triton benchmark results')
    parser.add_argument('--results', type=str, 
                       default='results/mlir_vs_triton_benchmark_results_v3.json',
                       help='Path to benchmark results JSON file')
    parser.add_argument('--output', type=str, default='visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')
    
    args = parser.parse_args()
    
    # Create visualizer
    try:
        visualizer = BenchmarkVisualizer(args.results)
    except FileNotFoundError:
        print(f"Error: Results file not found at {args.results}")
        print("Please run the benchmark first to generate results.")
        return
    
    # Create all visualizations
    output_path = visualizer.create_all_visualizations(args.output)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"Results analyzed from: {args.results}")
    print(f"Visualizations saved to: {output_path}/")
    print("\nGenerated files:")
    print("  - performance_comparison.png: Overall performance analysis")
    print("  - detailed_speedup.png: Detailed speedup breakdowns")
    print("  - performance_breakdown.png: Component-wise analysis")
    print("  - performance_report.txt: Text summary report")
    
    if args.show:
        print("\nShowing plots interactively...")
        plt.show()


if __name__ == "__main__":
    main()