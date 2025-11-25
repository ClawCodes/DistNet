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

## Experiment 1: Speedup and Convergence Analysis

**Goal**: Measure training time improvement and convergence speed as we increase the number of nodes.

### Setup
- **Fixed**: Dataset size (60K samples), epochs (16), batch size (32)
- **Variable**: World size = {1, 2, 3}
- **Runs**: 3 runs per configuration (average results)

### Metrics to Collect
- **Training time per epoch** (seconds)
- **Total training time** (seconds)
- **Time to reach target accuracy** (95%, 96%, 97%)
- **Final test accuracy** (%)
- **Speedup**: S(n) = T(1) / T(n)
- **Parallel Efficiency**: E(n) = S(n) / n
- **Communication overhead**: % of time spent in all-reduce

### Expected Results
Given our small model size (~436KB), we anticipate **significant communication overhead**:
- **World Size 2**: Speedup ~1.3-1.5× (efficiency ~65-75%)
- **World Size 3**: Speedup ~1.8-2.2× (efficiency ~60-70%)
- **QMNIST Caveat**: Task may saturate at ~97% accuracy
  - More nodes should reach target accuracy faster, even if final accuracy plateaus
  - Finding: For small models, distributed training accelerates convergence despite overhead

**Note**: Sub-linear speedup is expected and valuable for characterizing the communication-computation trade-off.

### Analysis
- Plot speedup curve (actual vs. ideal)
- Plot parallel efficiency vs. world size
- **Plot convergence time** for different accuracy targets (95%, 96%, 97%)
- Plot training curves (loss vs. time, accuracy vs. time)
- Calculate communication-to-computation ratio
- Identify break-even point where adding nodes becomes counterproductive

---

## Experiment 2: Communication Overhead Profiling

**Goal**: Quantify communication overhead in distributed training.

### Setup
- **Fixed**: 16 epochs, batch size 32
- **Variable**: World size = {1, 2, 3}
- **Instrumentation**: Add timers to measure communication in gradient hooks

### Metrics to Collect (per batch)
- **Total batch time** (seconds)
- **Communication time** (all-reduce, measured in hooks)
- **Computation time** (total - communication)
- **Communication percentage**: comm_time / total_time × 100%

### Expected Results
- Communication overhead increases with world size
- For 3 nodes: expect communication ~25-35% of total time
- Larger models would show lower percentage

### Analysis
- Pie chart: Computation vs. Communication time distribution
- Bar chart: Communication overhead % across different world sizes
- Trend analysis: How overhead scales with world size

---

## Data Collection Strategy

### Automated Logging
- Each node saves metrics to JSON file: `results/exp{N}_ws{W}_rank{R}.json`
- Master node (rank 0) aggregates results

### File Structure
```
results/
├── speedup_ws1_rank0.json
├── speedup_ws2_rank0.json
├── speedup_ws2_rank1.json
├── speedup_ws3_rank0.json
├── speedup_ws3_rank1.json
├── speedup_ws3_rank2.json
├── profiling_ws1_rank0.json
├── profiling_ws2_rank0.json
└── profiling_ws3_rank0.json
```

### JSON Format
```json
{
  "experiment": "speedup",
  "world_size": 3,
  "rank": 0,
  "metrics": {
    "epoch_times": [5.2, 5.1, 5.0, ...],
    "losses": [0.5, 0.3, 0.2, ...],
    "accuracies": [0.85, 0.92, 0.95, ...],
    "comm_times": [0.5, 0.5, 0.5, ...],
    "compute_times": [4.7, 4.6, 4.5, ...]
  }
}
```

---

## Visualization Plan

### Plots to Generate
1. **Speedup Curve**: Speedup vs. world size (with ideal line)
2. **Efficiency Curve**: Parallel efficiency vs. world size
3. **Convergence Time**: Time to reach accuracy targets (95%, 96%, 97%)
4. **Training Curves**: Loss/accuracy vs. epoch (overlay different world sizes)
5. **Time Distribution**: Pie chart or stacked bar (compute vs. communication)

### Tools
- `matplotlib` for plotting
- `pandas` for data aggregation
- `seaborn` for enhanced visualizations (optional)

---

## Success Criteria

### Minimum Viable Results
- Demonstrate measurable speedup for 2-3 nodes (even if sub-linear)
- Show faster convergence time with distributed training
- Quantify communication overhead (characterize, not minimize)
- Maintain comparable accuracy (±1% of baseline 97%)
- Identify model size threshold for effective distributed training

### Stretch Goals
- Achieve >60% parallel efficiency with 3 nodes
- Demonstrate communication overhead reduction via gradient accumulation
- Validate findings with larger model experiment
- Implement and compare model parallelism

---

## Notes

- Compare results with PyTorch's `torch.distributed` as reference (if time permits)
