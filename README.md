# MLIR vs Triton Flash Attention Benchmark, Roofline and Visualization

## Requirements

The codebase is designed for Python 3.8+ and uses a CUDA‐enabled GPU for best performance. Major dependencies include:

PyTorch (1.13 or newer) – for tensor operations and the baseline attention implementation.

Triton (2.0 or newer) – for custom GPU kernels and the MLIR integration.

Pandas, NumPy, Matplotlib, Seaborn – for data handling and visualisation.

TorchVision and Hugging Face Datasets are optional but convenient for obtaining the WikiText dataset.

Install the dependencies using pip:

```bash
pip install torch triton pandas numpy matplotlib seaborn
```

Preparing the Dataset

The benchmark uses the WikiText‑103 dataset to generate realistic attention inputs. The script expects Parquet files under ./wikitext/wikitext-103-v1/. To prepare the dataset:

Install the Hugging Face datasets package:
```bash
pip install datasets
```

Download and save the test split as Parquet files:
```python
from datasets import load_dataset

ds = load_dataset("wikitext", "wikitext-103-v1", split="test")
ds.to_parquet("./wikitext/wikitext-103-v1/test.parquet")
```

This will create the directory structure expected by the benchmark. If you already have your own text data, you can store it in a similar Parquet file with a text column.

Verify that the directory ./wikitext/wikitext-103-v1/ contains at least one test*.parquet file before running the benchmark.

## Running the Benchmark

Once the dataset is available and dependencies are installed, you can run the benchmark script. A CUDA‑enabled GPU is highly recommended; otherwise the script will fall back to CPU, which is extremely slow.

```bash
python benchmark.py
```

By default this will benchmark a selection of sequence lengths (64, 128, 256, 512) and batch sizes (1, 2, 4, 8) using eight heads and a head dimension of 64. It performs warm‑up iterations, measures runtime and memory usage over 50 runs per configuration, verifies correctness, and writes the results to results/mlir_vs_triton_benchmark_results.json. A detailed text report is saved in results/performance_report.txt.

You can adjust the parameters by editing the main() function or by instantiating the MLIRvsTritonBenchmark class in your own script. Key arguments include seq_lengths, batch_sizes, num_heads, head_dim, and num_iterations.

Running the Roofline Analysis

After generating benchmark results, use roofline.py to analyse arithmetic intensity and hardware utilisation. The script reads the benchmark JSON file and produces plots and reports describing where each kernel sits on the roofline (compute‑ vs memory‑bound) and how performance scales with problem size. 
 
```bash  
python roofline.py --json results/mlir_vs_triton_benchmark_results.json --output results/roofline
```  

This command will write a roofline plot (roofline_model.png), a scaling analysis (scaling_analysis.png), a text report (roofline_report.txt), and an enhanced JSON file with roofline data into results/roofline/.

Generating Visualisations

The visualize.py script reads the benchmark JSON file and generates a variety of charts and a summary report. These plots provide insights into execution time trends, speed‑up distributions, memory usage, throughput, and the effectiveness of different optimisations.

```bash
python visualize.py --json results/mlir_vs_triton_benchmark_results.json --output results/visualizations
```

After running, look in results/visualizations/ for:

mlir_performance_overview.png – a multi‑panel figure summarising execution time, speed‑up heatmaps, memory usage, scaling behaviours, batch impact, performance percentiles and throughput.

mlir_detailed_comparison.png – a figure with speed‑up by configuration, memory‑performance trade‑off, configuration comparisons, a radar chart of optimisation metrics and a table of best configurations.

mlir_visualization_report.txt – a plain‑text summary of benchmark statistics and best configurations.

You can open these images directly or incorporate them into presentations or reports. For interactive browsing, add --show to display the plots on screen.

```bash
Project Structure 
benchmark.py        # Benchmarking harness for flash attention kernels
roofline.py         # Roofline analysis utilities and report generation
visualize.py        # Visualization utilities for benchmark results
wikitext/           # Directory for WikiText dataset (not included)
results/            # Generated JSON files, reports and plots


```

Acknowledgements

The scripts in this repository were inspired by research on optimising transformer attention mechanisms and make use of the open‑source Triton compiler and the PyTorch deep learning framework.