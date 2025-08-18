#!/usr/bin/env python3
"""
Comprehensive Visualization for MLIR vs Triton Flash Attention Benchmarks
Creates detailed charts and analysis from JSON benchmark results
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class MLIRBenchmarkVisualizer:
    """Visualizer for MLIR vs Triton benchmark results"""
    
    def __init__(self, json_file: str = "results/mlir_vs_triton_benchmark_results.json"):
        """Initialize visualizer with JSON data"""
        self.json_file = json_file
        self.data = self.load_json_data()
        self.df = self.json_to_dataframe()
        
        # Color scheme for implementations
        self.colors = {
            'standard_triton': '#3498db',      # Blue
            'mlir_v1_fusion': '#e74c3c',       # Red
            'mlir_v2_coalescing': '#2ecc71',   # Green
            'pytorch': '#f39c12'                # Orange
        }
        
        self.labels = {
            'standard_triton': 'Standard Triton',
            'mlir_v1_fusion': 'MLIR V1 (Loop Fusion)',
            'mlir_v2_coalescing': 'MLIR V2 (Memory Coalescing)',
            'pytorch': 'PyTorch Native'
        }
    
    def load_json_data(self):
        """Load JSON benchmark data"""
        with open(self.json_file, 'r') as f:
            return json.load(f)
    
    def json_to_dataframe(self):
        """Convert JSON data to pandas DataFrame for easier analysis"""
        records = []
        
        for benchmark in self.data['benchmarks']:
            config = benchmark['configuration']
            base_record = {
                'batch_size': config['batch_size'],
                'seq_len': config['seq_len'],
                'num_heads': config['num_heads'],
                'head_dim': config['head_dim'],
                'total_elements': config['total_elements']
            }
            
            # Add implementation metrics
            for impl_name, impl_data in benchmark['implementations'].items():
                if 'mean_ms' in impl_data:
                    record = base_record.copy()
                    record['implementation'] = impl_name
                    record['mean_ms'] = impl_data['mean_ms']
                    record['std_ms'] = impl_data['std_ms']
                    record['min_ms'] = impl_data['min_ms']
                    record['max_ms'] = impl_data['max_ms']
                    record['median_ms'] = impl_data['median_ms']
                    record['p95_ms'] = impl_data['p95_ms']
                    record['p99_ms'] = impl_data['p99_ms']
                    record['mean_memory_mb'] = impl_data['mean_memory_mb']
                    record['peak_memory_mb'] = impl_data['peak_memory_mb']
                    
                    # Add speedup
                    if impl_name in benchmark['speedups']:
                        record['speedup'] = benchmark['speedups'][impl_name]
                    else:
                        record['speedup'] = 1.0
                    
                    # Add correctness info
                    for corr in benchmark['correctness']:
                        if impl_name in corr['implementation'].lower().replace(' ', '_'):
                            record['max_diff'] = corr['max_diff']
                            record['is_correct'] = corr['is_correct']
                            break
                    
                    records.append(record)
        
        return pd.DataFrame(records)
    
    def create_performance_overview(self):
        """Create main performance overview figure"""
        fig = plt.figure(figsize=(24, 14))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
        
        # Title
        fig.suptitle('MLIR vs Triton Flash Attention Performance Analysis', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        # 1. Execution Time by Sequence Length
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_time_vs_sequence(ax1)
        
        # 2. Speedup Heatmap
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_speedup_heatmap(ax2)
        
        # 3. Memory Usage Comparison
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_memory_usage(ax3)
        
        # 4. Performance Scaling
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_performance_scaling(ax4)
        
        # 5. Speedup Distribution
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_speedup_distribution(ax5)
        
        # 6. Batch Size Impact
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_batch_size_impact(ax6)
        
        # 7. Performance Percentiles
        ax7 = fig.add_subplot(gs[2, 1])
        self._plot_performance_percentiles(ax7)
        
        # 8. Efficiency Metrics
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_efficiency_metrics(ax8)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        return fig
    
    def _plot_time_vs_sequence(self, ax):
        """Plot execution time vs sequence length - simplified version"""
        # Average across batch sizes for cleaner plot
        avg_data = self.df.groupby(['seq_len', 'implementation']).agg({
            'mean_ms': 'mean',
            'std_ms': 'mean'
        }).reset_index()
        
        for impl in self.df['implementation'].unique():
            impl_data = avg_data[avg_data['implementation'] == impl]
            impl_data = impl_data.sort_values('seq_len')
            
            ax.plot(impl_data['seq_len'], impl_data['mean_ms'],
                   marker='o', label=self.labels[impl],
                   color=self.colors[impl], linewidth=2.5, markersize=8)
            
            # Add error bands
            ax.fill_between(impl_data['seq_len'],
                           impl_data['mean_ms'] - impl_data['std_ms'],
                           impl_data['mean_ms'] + impl_data['std_ms'],
                           color=self.colors[impl], alpha=0.15)
        
        ax.set_xlabel('Sequence Length', fontsize=12)
        ax.set_ylabel('Execution Time (ms)', fontsize=12)
        ax.set_title('Average Execution Time vs Sequence Length', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    def _plot_speedup_heatmap(self, ax):
        """Plot speedup heatmap"""
        # Create pivot table for MLIR V2 speedup
        mlir_v2 = self.df[self.df['implementation'] == 'mlir_v2_coalescing']
        
        if not mlir_v2.empty:
            pivot = mlir_v2.pivot_table(
                values='speedup',
                index='seq_len',
                columns='batch_size',
                aggfunc='mean'
            )
            
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                       center=1.0, ax=ax, cbar_kws={'label': 'Speedup'},
                       vmin=0.5, vmax=2.0, square=True)
            ax.set_title('MLIR V2 Speedup Heatmap', fontsize=14, fontweight='bold')
            ax.set_xlabel('Batch Size', fontsize=12)
            ax.set_ylabel('Sequence Length', fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No MLIR V2 data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('MLIR V2 Speedup Heatmap', fontsize=14, fontweight='bold')
    
    def _plot_memory_usage(self, ax):
        """Plot memory usage comparison"""
        memory_data = []
        for impl in self.df['implementation'].unique():
            impl_data = self.df[self.df['implementation'] == impl]
            memory_data.append({
                'Implementation': self.labels[impl],
                'impl_key': impl,
                'Mean': impl_data['peak_memory_mb'].mean(),
                'Std': impl_data['peak_memory_mb'].std()
            })
        
        memory_df = pd.DataFrame(memory_data)
        x = np.arange(len(memory_df))
        
        bars = ax.bar(x, memory_df['Mean'], yerr=memory_df['Std'],
                      color=[self.colors[impl] for impl in memory_df['impl_key']],
                      capsize=5, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Implementation', fontsize=12)
        ax.set_ylabel('Peak Memory (MB)', fontsize=12)
        ax.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(memory_df['Implementation'], rotation=30, ha='right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mean in zip(bars, memory_df['Mean']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.1f}', ha='center', va='bottom', fontsize=10)
    
    def _plot_performance_scaling(self, ax):
        """Plot performance scaling with sequence length"""
        for impl in self.df['implementation'].unique():
            impl_data = self.df[self.df['implementation'] == impl]
            
            # Average across batch sizes
            avg_data = impl_data.groupby('seq_len').agg({
                'mean_ms': 'mean',
                'std_ms': 'mean'
            }).reset_index()
            
            ax.plot(avg_data['seq_len'], avg_data['mean_ms'],
                   marker='s', label=self.labels[impl],
                   color=self.colors[impl], linewidth=2.5, markersize=8)
        
        ax.set_xlabel('Sequence Length', fontsize=12)
        ax.set_ylabel('Average Execution Time (ms)', fontsize=12)
        ax.set_title('Performance Scaling Analysis', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    def _plot_speedup_distribution(self, ax):
        """Plot speedup distribution for MLIR implementations"""
        mlir_v1 = self.df[self.df['implementation'] == 'mlir_v1_fusion']['speedup']
        mlir_v2 = self.df[self.df['implementation'] == 'mlir_v2_coalescing']['speedup']
        
        if not mlir_v1.empty and not mlir_v2.empty:
            ax.hist([mlir_v1, mlir_v2], bins=10, label=['MLIR V1', 'MLIR V2'],
                   color=[self.colors['mlir_v1_fusion'], self.colors['mlir_v2_coalescing']],
                   alpha=0.7, edgecolor='black', linewidth=1)
            
            ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Baseline')
            ax.axvline(x=mlir_v1.mean(), color=self.colors['mlir_v1_fusion'],
                      linestyle='-', linewidth=2, alpha=0.7)
            ax.axvline(x=mlir_v2.mean(), color=self.colors['mlir_v2_coalescing'],
                      linestyle='-', linewidth=2, alpha=0.7)
            
            # Add text annotations for means
            ax.text(mlir_v1.mean(), ax.get_ylim()[1]*0.9, f'V1: {mlir_v1.mean():.2f}x',
                   ha='center', fontsize=10, color=self.colors['mlir_v1_fusion'])
            ax.text(mlir_v2.mean(), ax.get_ylim()[1]*0.8, f'V2: {mlir_v2.mean():.2f}x',
                   ha='center', fontsize=10, color=self.colors['mlir_v2_coalescing'])
        
        ax.set_xlabel('Speedup Factor', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Speedup Distribution', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_batch_size_impact(self, ax):
        """Plot impact of batch size on performance"""
        batch_impact = self.df.groupby(['batch_size', 'implementation']).agg({
            'mean_ms': 'mean'
        }).reset_index()
        
        for impl in self.df['implementation'].unique():
            impl_data = batch_impact[batch_impact['implementation'] == impl]
            ax.plot(impl_data['batch_size'], impl_data['mean_ms'],
                   marker='o', label=self.labels[impl],
                   color=self.colors[impl], linewidth=2.5, markersize=8)
        
        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Average Execution Time (ms)', fontsize=12)
        ax.set_title('Batch Size Impact on Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(sorted(self.df['batch_size'].unique()))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
    
    def _plot_performance_percentiles(self, ax):
        """Plot performance percentiles"""
        percentiles = ['min_ms', 'median_ms', 'mean_ms', 'p95_ms', 'p99_ms']
        labels = ['Min', 'Median', 'Mean', 'P95', 'P99']
        
        x = np.arange(len(percentiles))
        width = 0.25
        
        implementations = list(self.df['implementation'].unique())[:3]  # Limit to 3 for space
        
        for i, impl in enumerate(implementations):
            impl_data = self.df[self.df['implementation'] == impl]
            values = [impl_data[p].mean() for p in percentiles]
            
            bars = ax.bar(x + i * width - width, values, width, 
                          label=self.labels[impl],
                          color=self.colors[impl], alpha=0.8,
                          edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Percentile', fontsize=12)
        ax.set_ylabel('Execution Time (ms)', fontsize=12)
        ax.set_title('Performance Percentiles', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
    
    def _plot_efficiency_metrics(self, ax):
        """Plot efficiency metrics"""
        # Calculate throughput (elements per second)
        self.df['throughput'] = self.df['total_elements'] / (self.df['mean_ms'] / 1000)
        
        efficiency_data = self.df.groupby('implementation').agg({
            'throughput': 'mean',
            'peak_memory_mb': 'mean'
        }).reset_index()
        
        x = np.arange(len(efficiency_data))
        
        # Single axis for throughput
        bars = ax.bar(x, efficiency_data['throughput'] / 1e9,
                     color=[self.colors[impl] for impl in efficiency_data['implementation']],
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Implementation', fontsize=12)
        ax.set_ylabel('Throughput (Gelements/s)', fontsize=12)
        ax.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([self.labels[impl] for impl in efficiency_data['implementation']],
                           rotation=30, ha='right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, efficiency_data['throughput'] / 1e9):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    def create_detailed_comparison(self):
        """Create detailed comparison charts"""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
        
        fig.suptitle('Detailed MLIR Optimization Analysis', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        # 1. Speedup by Configuration (spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_speedup_by_config(ax1)
        
        # 2. Memory vs Performance Trade-off
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_memory_performance_tradeoff(ax2)
        
        # 3. Configuration Comparison
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_config_comparison(ax3)
        
        # 4. Optimization Effectiveness (radar chart)
        ax4 = fig.add_subplot(gs[1, 1], projection='polar')
        self._plot_optimization_effectiveness(ax4)
        
        # 5. Best Configuration Table
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_best_configurations(ax5)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        return fig
    
    def _plot_speedup_by_config(self, ax):
        """Plot speedup for each configuration - simplified"""
        # Create configuration labels
        self.df['config'] = self.df.apply(
            lambda x: f"B{int(x['batch_size'])}/S{int(x['seq_len'])}", axis=1
        )
        
        # Get unique configurations (limit to prevent overcrowding)
        configs = sorted(self.df['config'].unique())[:15]
        x = np.arange(len(configs))
        width = 0.35
        
        for i, impl in enumerate(['mlir_v1_fusion', 'mlir_v2_coalescing']):
            impl_data = self.df[self.df['implementation'] == impl]
            speedups = []
            for config in configs:
                config_data = impl_data[impl_data['config'] == config]
                if not config_data.empty:
                    speedups.append(config_data['speedup'].values[0])
                else:
                    speedups.append(0)
            
            ax.bar(x + i * width - width/2, speedups, width, 
                  label=self.labels[impl],
                  color=self.colors[impl], alpha=0.8,
                  edgecolor='black', linewidth=1)
        
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax.set_xlabel('Configuration (Batch/Sequence)', fontsize=12)
        ax.set_ylabel('Speedup Factor', fontsize=12)
        ax.set_title('Speedup by Configuration', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_memory_performance_tradeoff(self, ax):
        """Plot memory vs performance trade-off"""
        for impl in self.df['implementation'].unique():
            impl_data = self.df[self.df['implementation'] == impl]
            
            # Use average values per sequence length for cleaner plot
            avg_data = impl_data.groupby('seq_len').agg({
                'peak_memory_mb': 'mean',
                'mean_ms': 'mean'
            }).reset_index()
            
            ax.scatter(avg_data['peak_memory_mb'], avg_data['mean_ms'],
                      label=self.labels[impl], color=self.colors[impl],
                      alpha=0.7, s=100, edgecolors='black', linewidth=1)
        
        ax.set_xlabel('Peak Memory (MB)', fontsize=12)
        ax.set_ylabel('Execution Time (ms)', fontsize=12)
        ax.set_title('Memory vs Performance Trade-off', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    def _plot_config_comparison(self, ax):
        """Plot configuration impact on speedup"""
        # Compare impact of batch size vs sequence length on speedup
        mlir_v2 = self.df[self.df['implementation'] == 'mlir_v2_coalescing']
        
        if not mlir_v2.empty:
            # Average speedup by batch size
            batch_speedup = mlir_v2.groupby('batch_size')['speedup'].mean()
            # Average speedup by sequence length
            seq_speedup = mlir_v2.groupby('seq_len')['speedup'].mean()
            
            x = np.arange(len(batch_speedup))
            ax.bar(x - 0.2, batch_speedup.values, 0.4, 
                  label='By Batch Size', color='steelblue', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([f'B{int(b)}' for b in batch_speedup.index], fontsize=10)
            ax.set_xlabel('Configuration', fontsize=12)
            ax.set_ylabel('Average Speedup', fontsize=12)
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax.legend(loc='upper left', fontsize=10)
            ax.set_title('MLIR V2 Speedup by Batch Size', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_optimization_effectiveness(self, ax):
        """Plot optimization effectiveness radar chart"""
        metrics = ['Speed', 'Memory', 'Consistency']
        
        implementations = ['standard_triton', 'mlir_v1_fusion', 'mlir_v2_coalescing']
        
        # Calculate metrics
        data = {}
        for impl in implementations:
            impl_data = self.df[self.df['implementation'] == impl]
            
            # Speed metric (inverse of mean time, normalized)
            base_time = self.df[self.df['implementation'] == 'standard_triton']['mean_ms'].mean()
            speed_metric = base_time / impl_data['mean_ms'].mean() if not impl_data.empty else 1.0
            
            # Memory metric (inverse of memory usage, normalized)
            base_memory = self.df[self.df['implementation'] == 'standard_triton']['peak_memory_mb'].mean()
            memory_metric = base_memory / impl_data['peak_memory_mb'].mean() if not impl_data.empty else 1.0
            
            # Consistency metric (inverse of std/mean ratio)
            consistency = 1 / (1 + impl_data['std_ms'].mean() / impl_data['mean_ms'].mean()) if not impl_data.empty else 0.5
            
            data[impl] = [speed_metric, memory_metric, consistency]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for impl in implementations:
            values = data[impl] + data[impl][:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=self.labels[impl], color=self.colors[impl])
            ax.fill(angles, values, alpha=0.25, color=self.colors[impl])
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 2.0)
        ax.set_title('Optimization Effectiveness', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=9)
        ax.grid(True)
    
    def _plot_best_configurations(self, ax):
        """Plot best configurations table"""
        best_configs = []
        
        for impl in self.df['implementation'].unique():
            impl_data = self.df[self.df['implementation'] == impl]
            if not impl_data.empty:
                best_idx = impl_data['speedup'].idxmax()
                best = impl_data.loc[best_idx]
                
                best_configs.append({
                    'Implementation': self.labels[impl][:15],  # Truncate for space
                    'Batch': int(best['batch_size']),
                    'Seq': int(best['seq_len']),
                    'Speedup': f"{best['speedup']:.2f}x",
                    'Time': f"{best['mean_ms']:.1f}ms"
                })
        
        if best_configs:
            best_df = pd.DataFrame(best_configs)
            
            # Create table
            ax.axis('tight')
            ax.axis('off')
            
            table = ax.table(cellText=best_df.values,
                            colLabels=best_df.columns,
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.35, 0.12, 0.12, 0.18, 0.18])
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2.0)
            
            # Style the table
            for (i, j), cell in table.get_celld().items():
                if i == 0:
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                    cell.set_text_props(color='black')
            
            ax.set_title('Best Configurations', fontsize=14, fontweight='bold', pad=20)
    
    def create_summary_report(self):
        """Create a text summary report"""
        report = []
        report.append("=" * 80)
        report.append("MLIR vs TRITON FLASH ATTENTION BENCHMARK VISUALIZATION REPORT")
        report.append("=" * 80)
        report.append(f"\nData Source: {self.json_file}")
        report.append(f"Total Benchmarks: {len(self.df)}")
        
        # Create config column if it doesn't exist
        if 'config' not in self.df.columns:
            self.df['config'] = self.df.apply(
                lambda x: f"B{int(x['batch_size'])}/S{int(x['seq_len'])}", axis=1
            )
        
        report.append(f"Configurations Tested: {len(self.df['config'].unique())}")
        
        # Summary statistics
        report.append("\n" + "=" * 40)
        report.append("PERFORMANCE SUMMARY")
        report.append("=" * 40)
        
        for impl in self.df['implementation'].unique():
            impl_data = self.df[self.df['implementation'] == impl]
            report.append(f"\n{self.labels[impl]}:")
            report.append(f"  Average Time: {impl_data['mean_ms'].mean():.3f} ms")
            report.append(f"  Average Memory: {impl_data['peak_memory_mb'].mean():.1f} MB")
            report.append(f"  Average Speedup: {impl_data['speedup'].mean():.2f}x")
            report.append(f"  Best Speedup: {impl_data['speedup'].max():.2f}x")
            report.append(f"  Worst Speedup: {impl_data['speedup'].min():.2f}x")
        
        # Best configurations
        report.append("\n" + "=" * 40)
        report.append("BEST CONFIGURATIONS")
        report.append("=" * 40)
        
        for impl in self.df['implementation'].unique():
            impl_data = self.df[self.df['implementation'] == impl]
            if not impl_data.empty:
                best_idx = impl_data['speedup'].idxmax()
                best = impl_data.loc[best_idx]
                
                report.append(f"\n{self.labels[impl]}:")
                report.append(f"  Batch Size: {int(best['batch_size'])}")
                report.append(f"  Sequence Length: {int(best['seq_len'])}")
                report.append(f"  Speedup: {best['speedup']:.2f}x")
                report.append(f"  Time: {best['mean_ms']:.3f} ms")
        
        return "\n".join(report)
    
    def save_all_visualizations(self, output_dir: str = "results/visualizations"):
        """Save all visualizations to files"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create and save main performance overview
        print("Creating performance overview...")
        fig1 = self.create_performance_overview()
        fig1.savefig(f"{output_dir}/mlir_performance_overview.png", 
                    dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig1)
        
        # Create and save detailed comparison
        print("Creating detailed comparison...")
        fig2 = self.create_detailed_comparison()
        fig2.savefig(f"{output_dir}/mlir_detailed_comparison.png", 
                    dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig2)
        
        # Save text report
        print("Generating text report...")
        report = self.create_summary_report()
        with open(f"{output_dir}/mlir_visualization_report.txt", 'w') as f:
            f.write(report)
        
        print(f"\nVisualizations saved to {output_dir}/")
        print("  - mlir_performance_overview.png")
        print("  - mlir_detailed_comparison.png")
        print("  - mlir_visualization_report.txt")
        
        return output_dir


def main():
    """Main function to create all visualizations"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize MLIR vs Triton benchmark results')
    parser.add_argument('--json', type=str, 
                       default='results/mlir_vs_triton_benchmark_results.json',
                       help='Path to JSON results file')
    parser.add_argument('--output', type=str,
                       default='results/visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')
    
    args = parser.parse_args()
    
    # Check if JSON file exists
    if not Path(args.json).exists():
        print(f"Error: JSON file '{args.json}' not found!")
        print("Please run the benchmark first to generate results.")
        return
    
    # Create visualizer
    print(f"Loading data from {args.json}...")
    visualizer = MLIRBenchmarkVisualizer(args.json)
    
    # Generate all visualizations
    output_dir = visualizer.save_all_visualizations(args.output)
    
    # Print summary report
    print("\n" + "=" * 60)
    print("VISUALIZATION SUMMARY")
    print("=" * 60)
    print(visualizer.create_summary_report())
    
    # Show plots if requested
    if args.show:
        print("\nShowing plots interactively...")
        fig1 = visualizer.create_performance_overview()
        fig2 = visualizer.create_detailed_comparison()
        plt.show()
    
    print("\nVisualization complete!")
    return output_dir


if __name__ == "__main__":
    main()