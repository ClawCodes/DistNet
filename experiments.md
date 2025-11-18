## Research Questions

1. **How can distributed systems be utilized to improve the performance of machine learning algorithms?**
   - Does distributing training across multiple nodes reduce training time?
   - What is the relationship between the number of nodes and speedup?

2. **Does processing larger datasets with distributed systems improve model performance?**
   - Given fixed training time, does distributing data across more nodes (enabling larger effective dataset) improve accuracy?

3. **(If time permits) Data Parallelism vs. Model Parallelism**
   - For simple vs. complex tasks, which parallelism strategy performs better?

---

## Experimental Setup

### Hardware & Software
- **Model**: 3-layer fully-connected perceptron (784 → 128 → 64 → 10)
  - Parameters: ~109K (~436KB)
- **Dataset**: QMNIST handwritten digit classification
  - Training: 60,000 samples
  - Testing: 10,000 samples
- **Baseline Performance**: 97% test accuracy (16 epochs, ~90 seconds on single node)
- **Batch Size**: 32
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss

### Model Selection Rationale
We deliberately chose a **small model** (~109K parameters) to:
1. Investigate the break-even point where communication overhead negates distributed benefits
2. Study minimum model size requirements for effective distributed training
3. Provide practical guidance on when distributed training is worthwhile

**Expected challenge**: Communication overhead will be significant relative to computation, allowing us to quantify trade-offs.

### Distributed System Configuration
- **Data Partitioning**: DistributedSampler (interleaved assignment)
- **Gradient Synchronization**: Ring All-Reduce algorithm
- **Communication**: Custom TCP-based tensor communication layer

---

## Experiment 1: Speedup vs. World Size

**Goal**: Measure training time improvement as we increase the number of nodes.

### Setup
- **Fixed**: Dataset size (60K samples), epochs (16), batch size (32)
- **Variable**: World size = {1, 2, 4, 8(?)}
- **Runs**: 3 runs per configuration (average results)

### Metrics to Collect
- **Training time per epoch** (seconds)
- **Total training time** (seconds)
- **Final test accuracy** (%)
- **Speedup**: S(n) = T(1) / T(n)
- **Parallel Efficiency**: E(n) = S(n) / n
- **Communication overhead**: % of time spent in all-reduce

### Expected Results
Given our small model size (~436KB), we anticipate **significant communication overhead**:
- **World Size 2**: Speedup ~1.3-1.5× (efficiency ~65-75%)
- **World Size 4**: Speedup ~2-2.5× (efficiency ~50-60%)
- **World Size 8**: Diminishing or negative returns likely

**Note**: Sub-linear speedup is expected and valuable for characterizing the communication-computation trade-off.

### Analysis
- Plot speedup curve (actual vs. ideal)
- Plot parallel efficiency vs. world size
- Calculate communication-to-computation ratio
- Identify break-even point where adding nodes becomes counterproductive

---

## Experiment 2: Accuracy vs. Data Size (Fixed Time)

**Goal**: Test if processing more data (via distribution) improves model performance when training time is fixed.

### Setup
- **Fixed**: Training time (5 minutes)
- **Variable**:
  - Configuration A: 1 node (processes ~N samples in 5 min)
  - Configuration B: 4 nodes (processes ~4N samples in 5 min)
- **Runs**: 5 runs per configuration

### Metrics to Collect
- **Total samples processed**
- **Final test accuracy** (%)
- **Training loss convergence**
- **Epochs completed** (may differ due to different throughput)

### Expected Results
- More nodes → more data processed in same time
- Hypothesis: Processing more data should improve accuracy, but diminishing returns expected
- Trade-off: More nodes = more synchronization overhead

### Analysis
- Compare final test accuracy between configurations
- Plot training curves (loss vs. time, accuracy vs. time)
- Calculate "samples processed per accuracy point gained"

---

## Experiment 3: Communication Overhead Profiling

**Goal**: Quantify communication overhead in distributed training.

### Setup
- **Fixed**: 16 epochs, batch size 32
- **Variable**: World size = {1, 2, 4}
- **Instrumentation**: Add timers to measure communication in gradient hooks

### Metrics to Collect (per batch)
- **Total batch time** (seconds)
- **Communication time** (all-reduce, measured in hooks)
- **Computation time** (total - communication)
- **Communication percentage**: comm_time / total_time × 100%

### Expected Results
- Communication overhead increases with world size
- For 4 nodes: expect communication ~30-40% of total time
- Larger models would show lower percentage

### Analysis
- Pie chart: Computation vs. Communication time distribution
- Bar chart: Communication overhead % across different world sizes
- Trend analysis: How overhead scales with world size

---

> **Note**: Additional experiments (Scaling Analysis, Network Traffic, Model Parallelism) are documented in [OPTIONAL_EXPERIMENTS.md](OPTIONAL_EXPERIMENTS.md) for potential future work.

---

## Data Collection Strategy

### Automated Logging
- Each node saves metrics to JSON file: `results/exp{N}_ws{W}_rank{R}.json`
- Master node (rank 0) aggregates results

### File Structure
```
results/
├── experiment1_speedup/
│   ├── world_size_1_rank_0.json
│   ├── world_size_2_rank_0.json
│   ├── world_size_2_rank_1.json
│   └── ...
├── experiment2_datasize/
│   └── ...
└── experiment3_profiling/
    └── ...
```

### JSON Format
```json
{
  "experiment": "speedup",
  "world_size": 4,
  "rank": 0,
  "epochs": 16,
  "epoch_times": [5.2, 5.1, 5.0, ...],
  "losses": [0.5, 0.3, 0.2, ...],
  "accuracies": [0.85, 0.92, 0.95, ...],
  "comm_times": [0.5, 0.5, 0.5, ...],
  "compute_times": [4.7, 4.6, 4.5, ...]
}
```

---

## Visualization Plan

### Plots to Generate
1. **Speedup Curve**: Speedup vs. world size (with ideal line)
2. **Efficiency Curve**: Parallel efficiency vs. world size
3. **Training Curves**: Loss/accuracy vs. epoch (overlay different world sizes)
4. **Time Distribution**: Pie chart or stacked bar (compute vs. communication)

### Tools
- `matplotlib` for plotting
- `pandas` for data aggregation
- `seaborn` for enhanced visualizations (optional)

---

## Success Criteria

### Minimum Viable Results
- Demonstrate measurable speedup for 2-4 nodes (even if sub-linear)
- Quantify communication overhead (characterize, not minimize)
- Maintain comparable accuracy (±1% of baseline 97%)
- Identify model size threshold for effective distributed training

### Stretch Goals
- Achieve >50% parallel efficiency with 4 nodes
- Demonstrate communication overhead reduction via gradient accumulation
- Validate findings with larger model experiment
- Implement and compare model parallelism

---

## Notes

- All experiments should be reproducible (fixed random seeds)
- Document any issues or anomalies encountered
- Compare results with PyTorch's `torch.distributed` as reference (if time permits)
