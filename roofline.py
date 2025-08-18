#!/usr/bin/env python3
"""
Roofline Model Analysis for Flash Attention Implementations
Analyzes performance bounds and identifies bottlenecks
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import seaborn as sns

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

@dataclass
class GPUSpecs:
    """GPU Hardware Specifications"""
    name: str
    peak_flops_fp16: float  # TFLOPS for FP16
    peak_flops_fp32: float  # TFLOPS for FP32
    memory_bandwidth: float  # GB/s
    l2_cache_size: float    # MB
    sm_count: int
    tensor_core_flops: float  # TFLOPS for tensor cores
    
    @classmethod
    def get_gpu_specs(cls, gpu_name: Optional[str] = None):
        """Get specifications for common GPUs"""
        if gpu_name is None and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
        
        # Common GPU specifications (FP16 TFLOPS, FP32 TFLOPS, Bandwidth GB/s)
        gpu_database = {
            "NVIDIA A100": cls("A100", 312, 156, 2039, 40, 108, 624),
            "NVIDIA A6000": cls("A6000", 155, 77.5, 768, 48, 84, 310),
            "NVIDIA V100": cls("V100", 125, 15.7, 900, 6, 80, 125),
            "NVIDIA RTX 3090": cls("RTX 3090", 142, 35.6, 936, 6, 82, 284),
            "NVIDIA RTX 4090": cls("RTX 4090", 330, 82.6, 1008, 72, 128, 660),
            "NVIDIA H100": cls("H100", 1979, 67, 3350, 50, 132, 3958),
            "NVIDIA RTX 3080": cls("RTX 3080", 119, 29.8, 760, 5, 68, 238),
            "NVIDIA T4": cls("T4", 65, 8.1, 320, 4, 40, 130),
        }
        
        # Try to match GPU name
        for key in gpu_database:
            if gpu_name and key.lower() in gpu_name.lower():
                return gpu_database[key]
        
        # Default to A100 if not found
        print(f"GPU '{gpu_name}' not in database, using A100 specs as default")
        return gpu_database["NVIDIA A100"]


class FlashAttentionRoofline:
    """Roofline Model Analysis for Flash Attention"""
    
    def __init__(self, gpu_specs: Optional[GPUSpecs] = None):
        """Initialize with GPU specifications"""
        self.gpu_specs = gpu_specs or GPUSpecs.get_gpu_specs()
        print(f"Using GPU specs for: {self.gpu_specs.name}")
        print(f"  Peak FP16: {self.gpu_specs.peak_flops_fp16} TFLOPS")
        print(f"  Peak Memory BW: {self.gpu_specs.memory_bandwidth} GB/s")
        print(f"  Tensor Cores: {self.gpu_specs.tensor_core_flops} TFLOPS")
    
    def calculate_arithmetic_intensity(self, 
                                      batch_size: int,
                                      seq_len: int,
                                      num_heads: int,
                                      head_dim: int,
                                      causal: bool = False) -> Dict[str, float]:
        """
        Calculate arithmetic intensity for Flash Attention
        
        Arithmetic Intensity (AI) = FLOPs / Memory Bytes Transferred
        """
        # Calculate FLOPs for attention
        # Forward pass: Q @ K^T, softmax, @ V
        # Q @ K^T: 2 * batch * heads * seq_len * seq_len * head_dim FLOPs
        # Softmax: ~5 * batch * heads * seq_len * seq_len FLOPs  
        # @ V: 2 * batch * heads * seq_len * seq_len * head_dim FLOPs
        
        total_elements = batch_size * num_heads
        
        # Standard attention FLOPs
        matmul1_flops = 2 * total_elements * seq_len * seq_len * head_dim
        softmax_flops = 5 * total_elements * seq_len * seq_len  # Approximate
        matmul2_flops = 2 * total_elements * seq_len * seq_len * head_dim
        
        if causal:
            # Causal masking reduces computation by ~half
            matmul1_flops /= 2
            softmax_flops /= 2
            matmul2_flops /= 2
        
        total_flops = matmul1_flops + softmax_flops + matmul2_flops
        
        # Memory transfers for different implementations
        bytes_per_element = 2  # FP16
        
        # Standard attention (naive): Load Q, K, V, store attention scores, load scores, store output
        standard_memory = batch_size * num_heads * seq_len * head_dim * bytes_per_element * 3  # Q, K, V
        standard_memory += batch_size * num_heads * seq_len * seq_len * bytes_per_element * 2  # Scores
        standard_memory += batch_size * num_heads * seq_len * head_dim * bytes_per_element  # Output
        
        # Flash Attention: Only load Q, K, V once and store output (no intermediate scores)
        flash_memory = batch_size * num_heads * seq_len * head_dim * bytes_per_element * 3  # Q, K, V
        flash_memory += batch_size * num_heads * seq_len * head_dim * bytes_per_element  # Output
        
        # Flash Attention with tiling (considers block-wise recomputation)
        block_size = 64  # Typical block size
        num_blocks = (seq_len + block_size - 1) // block_size
        tiled_memory = flash_memory * (1 + np.log2(num_blocks))  # Account for recomputation
        
        return {
            'flops': total_flops,
            'standard_memory_bytes': standard_memory,
            'flash_memory_bytes': flash_memory,
            'tiled_memory_bytes': tiled_memory,
            'standard_ai': total_flops / standard_memory,
            'flash_ai': total_flops / flash_memory,
            'tiled_ai': total_flops / tiled_memory,
            'theoretical_flops': total_flops / 1e12,  # In TFLOPS
            'theoretical_bandwidth_standard': standard_memory / 1e9,  # In GB
            'theoretical_bandwidth_flash': flash_memory / 1e9,  # In GB
        }
    
    def calculate_achieved_performance(self,
                                      execution_time_ms: float,
                                      arithmetic_intensity_info: Dict[str, float]) -> Dict[str, float]:
        """Calculate achieved performance metrics"""
        execution_time_s = execution_time_ms / 1000
        
        # Achieved TFLOPS
        achieved_tflops = arithmetic_intensity_info['theoretical_flops'] / execution_time_s
        
        # Achieved bandwidth (assuming Flash Attention memory pattern)
        achieved_bandwidth = arithmetic_intensity_info['theoretical_bandwidth_flash'] / execution_time_s
        
        # Efficiency metrics
        compute_efficiency = achieved_tflops / self.gpu_specs.peak_flops_fp16
        bandwidth_efficiency = achieved_bandwidth / self.gpu_specs.memory_bandwidth
        
        # Determine if kernel is compute or memory bound
        ridge_point = self.gpu_specs.peak_flops_fp16 * 1e12 / (self.gpu_specs.memory_bandwidth * 1e9)
        is_compute_bound = arithmetic_intensity_info['flash_ai'] > ridge_point
        
        return {
            'achieved_tflops': achieved_tflops,
            'achieved_bandwidth_gbs': achieved_bandwidth,
            'compute_efficiency': compute_efficiency,
            'bandwidth_efficiency': bandwidth_efficiency,
            'is_compute_bound': is_compute_bound,
            'ridge_point': ridge_point,
            'arithmetic_intensity': arithmetic_intensity_info['flash_ai']
        }
    
    def plot_roofline(self,
                      benchmark_results: List[Dict],
                      save_path: Optional[str] = None):
        """
        Plot roofline model with benchmark results
        
        Args:
            benchmark_results: List of dicts with 'config', 'implementation', 'performance' keys
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Calculate roofline boundaries
        peak_flops = self.gpu_specs.peak_flops_fp16 * 1e12  # Convert to FLOPS
        peak_bandwidth = self.gpu_specs.memory_bandwidth * 1e9  # Convert to B/s
        tensor_core_flops = self.gpu_specs.tensor_core_flops * 1e12
        
        # Arithmetic intensity range (FLOP/byte)
        ai_range = np.logspace(-2, 3, 1000)
        
        # Roofline boundaries
        memory_bound = peak_bandwidth * ai_range / 1e12  # Convert to TFLOPS
        compute_bound = np.ones_like(ai_range) * self.gpu_specs.peak_flops_fp16
        tensor_bound = np.ones_like(ai_range) * self.gpu_specs.tensor_core_flops
        
        # Combined roofline
        roofline = np.minimum(memory_bound, compute_bound)
        tensor_roofline = np.minimum(memory_bound, tensor_bound)
        
        # Plot 1: Traditional Roofline
        ax1.loglog(ai_range, roofline, 'b-', linewidth=2.5, label='Peak FP16')
        ax1.loglog(ai_range, tensor_roofline, 'g--', linewidth=2, label='Tensor Cores', alpha=0.7)
        
        # Ridge point
        ridge_point = self.gpu_specs.peak_flops_fp16 * 1e12 / (self.gpu_specs.memory_bandwidth * 1e9)
        ax1.axvline(x=ridge_point, color='red', linestyle=':', alpha=0.5, label=f'Ridge Point ({ridge_point:.2f})')
        
        # Add bandwidth lines for different cache levels
        l2_bandwidth = self.gpu_specs.memory_bandwidth * 2  # Approximate L2 bandwidth
        l2_bound = l2_bandwidth * ai_range * 1e9 / 1e12
        ax1.loglog(ai_range, np.minimum(l2_bound, compute_bound), 'c:', 
                   linewidth=1.5, alpha=0.5, label='L2 Cache')
        
        # Plot benchmark points
        colors = {'standard_triton': 'orange', 'mlir_v1_fusion': 'red', 'mlir_v2_coalescing': 'green'}
        markers = {'standard_triton': 'o', 'mlir_v1_fusion': 's', 'mlir_v2_coalescing': '^'}
        
        for result in benchmark_results:
            if 'performance' in result:
                ai = result['performance']['arithmetic_intensity']
                tflops = result['performance']['achieved_tflops']
                impl = result['implementation']
                
                ax1.scatter(ai, tflops, 
                           color=colors.get(impl, 'gray'),
                           marker=markers.get(impl, 'o'),
                           s=100, alpha=0.7,
                           label=impl if impl not in [r.get('implementation') for r in benchmark_results[:benchmark_results.index(result)]] else '')
        
        # Annotations
        ax1.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12)
        ax1.set_ylabel('Performance (TFLOPS)', fontsize=12)
        ax1.set_title('Roofline Model Analysis', fontsize=14, fontweight='bold')
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend(loc='lower right', fontsize=10)
        ax1.set_xlim([0.1, 1000])
        ax1.set_ylim([0.1, max(self.gpu_specs.peak_flops_fp16, self.gpu_specs.tensor_core_flops) * 1.5])
        
        # Add shaded regions
        ax1.fill_between([0.1, ridge_point], 0.1, 1000, alpha=0.1, color='blue', label='Memory Bound')
        ax1.fill_between([ridge_point, 1000], 0.1, 1000, alpha=0.1, color='red', label='Compute Bound')
        
        # Plot 2: Efficiency Analysis
        if benchmark_results:
            configs = []
            compute_eff = []
            memory_eff = []
            impl_types = []
            
            for result in benchmark_results:
                if 'performance' in result:
                    config = result['config']
                    configs.append(f"B{config['batch_size']}/S{config['seq_len']}")
                    compute_eff.append(result['performance']['compute_efficiency'] * 100)
                    memory_eff.append(result['performance']['bandwidth_efficiency'] * 100)
                    impl_types.append(result['implementation'])
            
            x = np.arange(len(configs))
            width = 0.35
            
            # Group by implementation
            for i, impl in enumerate(colors.keys()):
                impl_indices = [j for j, imp in enumerate(impl_types) if imp == impl]
                if impl_indices:
                    impl_configs = [configs[j] for j in impl_indices]
                    impl_compute = [compute_eff[j] for j in impl_indices]
                    impl_memory = [memory_eff[j] for j in impl_indices]
                    
                    x_impl = np.arange(len(impl_configs))
                    ax2.bar(x_impl + i * width, impl_compute, width, 
                           label=f'{impl} (Compute)', color=colors[impl], alpha=0.7)
                    ax2.bar(x_impl + i * width, impl_memory, width, 
                           label=f'{impl} (Memory)', color=colors[impl], alpha=0.4, hatch='//')
        
        ax2.set_xlabel('Configuration', fontsize=12)
        ax2.set_ylabel('Hardware Efficiency (%)', fontsize=12)
        ax2.set_title('Hardware Utilization Efficiency', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Roofline plot saved to {save_path}")
        
        return fig
    
    def analyze_scaling_behavior(self,
                                 benchmark_data: Dict,
                                 save_path: Optional[str] = None):
        """Analyze how arithmetic intensity and performance scale with problem size"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data by sequence length
        seq_lengths = []
        ai_values = {'standard': [], 'flash': [], 'tiled': []}
        achieved_tflops = {'standard_triton': [], 'mlir_v1_fusion': [], 'mlir_v2_coalescing': []}
        efficiency = {'standard_triton': [], 'mlir_v1_fusion': [], 'mlir_v2_coalescing': []}
        
        for benchmark in benchmark_data.get('benchmarks', []):
            config = benchmark['configuration']
            seq_len = config['seq_len']
            
            # Calculate AI for this configuration
            ai_info = self.calculate_arithmetic_intensity(
                config['batch_size'], seq_len, 
                config['num_heads'], config['head_dim']
            )
            
            if seq_len not in seq_lengths:
                seq_lengths.append(seq_len)
                ai_values['standard'].append(ai_info['standard_ai'])
                ai_values['flash'].append(ai_info['flash_ai'])
                ai_values['tiled'].append(ai_info['tiled_ai'])
            
            # Get performance for each implementation
            for impl in benchmark['implementations']:
                if 'mean_ms' in benchmark['implementations'][impl]:
                    perf = self.calculate_achieved_performance(
                        benchmark['implementations'][impl]['mean_ms'],
                        ai_info
                    )
                    if impl in achieved_tflops:
                        achieved_tflops[impl].append(perf['achieved_tflops'])
                        efficiency[impl].append(perf['compute_efficiency'] * 100)
        
        # Sort by sequence length
        sorted_indices = np.argsort(seq_lengths)
        seq_lengths = [seq_lengths[i] for i in sorted_indices]
        
        # Plot 1: Arithmetic Intensity Scaling
        ax1 = axes[0, 0]
        ax1.plot(seq_lengths, [ai_values['standard'][i] for i in sorted_indices], 
                'o-', label='Standard Attention', linewidth=2)
        ax1.plot(seq_lengths, [ai_values['flash'][i] for i in sorted_indices], 
                's-', label='Flash Attention', linewidth=2)
        ax1.plot(seq_lengths, [ai_values['tiled'][i] for i in sorted_indices], 
                '^-', label='Tiled Flash', linewidth=2)
        ax1.axhline(y=self.gpu_specs.peak_flops_fp16 * 1e12 / (self.gpu_specs.memory_bandwidth * 1e9),
                   color='red', linestyle='--', alpha=0.5, label='Ridge Point')
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Arithmetic Intensity (FLOP/Byte)')
        ax1.set_title('AI Scaling with Sequence Length')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Achieved Performance Scaling
        ax2 = axes[0, 1]
        for impl in achieved_tflops:
            if achieved_tflops[impl]:
                # Ensure we have matching data lengths
                data_length = min(len(seq_lengths), len(achieved_tflops[impl]))
                if data_length > 0:
                    ax2.plot(seq_lengths[:data_length], 
                            achieved_tflops[impl][:data_length], 
                            'o-', label=impl, linewidth=2)
        ax2.axhline(y=self.gpu_specs.peak_flops_fp16, 
                   color='red', linestyle='--', alpha=0.5, label='Peak FP16')
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Achieved TFLOPS')
        ax2.set_title('Performance Scaling')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Efficiency vs Problem Size
        ax3 = axes[1, 0]
        for impl in efficiency:
            if efficiency[impl]:
                # Ensure we have matching data lengths
                data_length = min(len(seq_lengths), len(efficiency[impl]))
                if data_length > 0:
                    ax3.plot(seq_lengths[:data_length], 
                            efficiency[impl][:data_length], 
                            'o-', label=impl, linewidth=2)
        ax3.set_xlabel('Sequence Length')
        ax3.set_ylabel('Compute Efficiency (%)')
        ax3.set_title('Hardware Efficiency Scaling')
        ax3.set_xscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Memory vs Compute Bound Analysis
        ax4 = axes[1, 1]
        bound_types = []
        for seq_len in seq_lengths:
            ai_info = self.calculate_arithmetic_intensity(
                4, seq_len, 8, 64  # Default config
            )
            ridge = self.gpu_specs.peak_flops_fp16 * 1e12 / (self.gpu_specs.memory_bandwidth * 1e9)
            bound_types.append(1 if ai_info['flash_ai'] > ridge else 0)
        
        ax4.bar(range(len(seq_lengths)), bound_types, color=['blue' if b == 0 else 'red' for b in bound_types])
        ax4.set_xticks(range(len(seq_lengths)))
        ax4.set_xticklabels(seq_lengths)
        ax4.set_xlabel('Sequence Length')
        ax4.set_ylabel('Bound Type')
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['Memory', 'Compute'])
        ax4.set_title('Dominant Performance Bound')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Flash Attention Roofline Scaling Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Scaling analysis saved to {save_path}")
        
        return fig
    
    def generate_roofline_report(self, benchmark_data: Dict) -> str:
        """Generate a detailed roofline analysis report"""
        report = []
        report.append("=" * 80)
        report.append("ROOFLINE MODEL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"\nGPU: {self.gpu_specs.name}")
        report.append(f"Peak FP16 Performance: {self.gpu_specs.peak_flops_fp16} TFLOPS")
        report.append(f"Peak Memory Bandwidth: {self.gpu_specs.memory_bandwidth} GB/s")
        report.append(f"Ridge Point: {self.gpu_specs.peak_flops_fp16 * 1e12 / (self.gpu_specs.memory_bandwidth * 1e9):.2f} FLOP/Byte")
        
        report.append("\n" + "=" * 40)
        report.append("ARITHMETIC INTENSITY ANALYSIS")
        report.append("=" * 40)
        
        # Analyze each configuration
        for benchmark in benchmark_data.get('benchmarks', []):
            config = benchmark['configuration']
            ai_info = self.calculate_arithmetic_intensity(
                config['batch_size'], config['seq_len'],
                config['num_heads'], config['head_dim']
            )
            
            report.append(f"\nConfiguration: B={config['batch_size']}, L={config['seq_len']}")
            report.append(f"  Standard AI: {ai_info['standard_ai']:.2f} FLOP/Byte")
            report.append(f"  Flash AI: {ai_info['flash_ai']:.2f} FLOP/Byte")
            report.append(f"  Theoretical FLOPs: {ai_info['theoretical_flops']:.2f} TFLOPS")
            
            # Performance analysis for each implementation
            for impl_name, impl_data in benchmark['implementations'].items():
                if 'mean_ms' in impl_data:
                    perf = self.calculate_achieved_performance(impl_data['mean_ms'], ai_info)
                    report.append(f"\n  {impl_name}:")
                    report.append(f"    Achieved: {perf['achieved_tflops']:.2f} TFLOPS")
                    report.append(f"    Compute Efficiency: {perf['compute_efficiency']*100:.1f}%")
                    report.append(f"    Bandwidth: {perf['achieved_bandwidth_gbs']:.1f} GB/s")
                    report.append(f"    Bound: {'Compute' if perf['is_compute_bound'] else 'Memory'}")
        
        report.append("\n" + "=" * 40)
        report.append("OPTIMIZATION RECOMMENDATIONS")
        report.append("=" * 40)
        
        # Analyze overall patterns
        compute_bound_configs = 0
        memory_bound_configs = 0
        
        for benchmark in benchmark_data.get('benchmarks', []):
            config = benchmark['configuration']
            ai_info = self.calculate_arithmetic_intensity(
                config['batch_size'], config['seq_len'],
                config['num_heads'], config['head_dim']
            )
            ridge = self.gpu_specs.peak_flops_fp16 * 1e12 / (self.gpu_specs.memory_bandwidth * 1e9)
            if ai_info['flash_ai'] > ridge:
                compute_bound_configs += 1
            else:
                memory_bound_configs += 1
        
        report.append(f"\nCompute-bound configs: {compute_bound_configs}")
        report.append(f"Memory-bound configs: {memory_bound_configs}")
        
        if memory_bound_configs > compute_bound_configs:
            report.append("\nMajority of configurations are MEMORY BOUND:")
            report.append("  - Focus on reducing memory transfers")
            report.append("  - Consider more aggressive tiling")
            report.append("  - Exploit shared memory and registers")
            report.append("  - Use memory coalescing (MLIR V2 approach)")
        else:
            report.append("\nMajority of configurations are COMPUTE BOUND:")
            report.append("  - Focus on computational optimizations")
            report.append("  - Utilize tensor cores when possible")
            report.append("  - Consider mixed precision strategies")
            report.append("  - Optimize GEMM operations")
        
        return "\n".join(report)


def integrate_roofline_with_benchmark(benchmark_json_path: str, output_dir: str = "results/roofline"):
    """
    Integrate roofline analysis with existing benchmark results
    
    Args:
        benchmark_json_path: Path to benchmark JSON results
        output_dir: Directory to save roofline analysis
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load benchmark data
    with open(benchmark_json_path, 'r') as f:
        benchmark_data = json.load(f)
    
    # Initialize roofline analyzer
    analyzer = FlashAttentionRoofline()
    
    # Process benchmark results
    roofline_results = []
    
    for benchmark in benchmark_data['benchmarks']:
        config = benchmark['configuration']
        
        # Calculate arithmetic intensity
        ai_info = analyzer.calculate_arithmetic_intensity(
            config['batch_size'],
            config['seq_len'],
            config['num_heads'],
            config['head_dim']
        )
        
        # Analyze each implementation
        for impl_name, impl_data in benchmark['implementations'].items():
            if 'mean_ms' in impl_data:
                perf = analyzer.calculate_achieved_performance(
                    impl_data['mean_ms'],
                    ai_info
                )
                
                roofline_results.append({
                    'config': config,
                    'implementation': impl_name,
                    'performance': perf,
                    'ai_info': ai_info
                })
    
    # Generate visualizations
    print("Generating roofline plots...")
    
    # Main roofline plot
    fig1 = analyzer.plot_roofline(roofline_results, 
                                  save_path=f"{output_dir}/roofline_model.png")
    
    # Scaling analysis
    fig2 = analyzer.analyze_scaling_behavior(benchmark_data,
                                            save_path=f"{output_dir}/scaling_analysis.png")
    
    # Generate report
    report = analyzer.generate_roofline_report(benchmark_data)
    report_path = f"{output_dir}/roofline_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nRoofline analysis complete!")
    print(f"Results saved to {output_dir}/")
    print(f"  - roofline_model.png")
    print(f"  - scaling_analysis.png")
    print(f"  - roofline_report.txt")
    
    # Save enhanced results with roofline data
    enhanced_results = benchmark_data.copy()
    enhanced_results['roofline_analysis'] = {
        'gpu_specs': analyzer.gpu_specs.__dict__,
        'results': roofline_results
    }
    
    with open(f"{output_dir}/enhanced_benchmark_results.json", 'w') as f:
        json.dump(enhanced_results, f, indent=2)
    
    return enhanced_results


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Roofline Analysis for Flash Attention')
    parser.add_argument('--json', type=str, 
                       default='results/mlir_vs_triton_benchmark_results.json',
                       help='Path to benchmark JSON results')
    parser.add_argument('--output', type=str,
                       default='results/roofline',
                       help='Output directory for roofline analysis')
    parser.add_argument('--gpu', type=str,
                       default=None,
                       help='GPU name for specifications (e.g., "A100", "RTX 4090")')
    
    args = parser.parse_args()
    
    # Run integrated analysis
    if Path(args.json).exists():
        results = integrate_roofline_with_benchmark(args.json, args.output)
        print("\nRoofline analysis successfully integrated with benchmark results!")
    else:
        print(f"Error: Benchmark file {args.json} not found!")
        print("Creating example roofline plot with dummy data...")
        
        # Example with dummy data
        analyzer = FlashAttentionRoofline()
        
        # Generate example configurations
        dummy_results = []
        for seq_len in [128, 256, 512, 1024]:
            ai_info = analyzer.calculate_arithmetic_intensity(4, seq_len, 8, 64)
            for impl in ['standard_triton', 'mlir_v1_fusion', 'mlir_v2_coalescing']:
                # Simulate performance (better for MLIR implementations)
                base_time = seq_len * 0.01
                if impl == 'mlir_v1_fusion':
                    base_time *= 0.8
                elif impl == 'mlir_v2_coalescing':
                    base_time *= 0.7
                
                perf = analyzer.calculate_achieved_performance(base_time, ai_info)
                dummy_results.append({
                    'config': {'batch_size': 4, 'seq_len': seq_len, 'num_heads': 8, 'head_dim': 64},
                    'implementation': impl,
                    'performance': perf
                })
        
        fig = analyzer.plot_roofline(dummy_results)
        plt.show()