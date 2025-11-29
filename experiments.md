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
- **Model**: 4-layer fully-connected perceptron (3072 → 512 → 256 → 128 → 10)
  - Parameters: ~1.74M (~6.63MB)
  - Input: Flattened 32×32×3 RGB images from CIFAR-10
  - Activation: GELU with Dropout (0.2)
- **Dataset**: CIFAR-10 image classification
  - Training: 50,000 samples (10 classes)
  - Testing: 10,000 samples
- **Baseline Performance**: ~54% test accuracy (15 epochs, batch size 128)
- **Batch Size**: 128
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss

### Model Selection Rationale
We chose a **medium-sized fully-connected model** (~1.74M parameters) to:
1. Provide sufficient computation to make distributed training beneficial
2. Quantify the communication-computation trade-off in a realistic setting
3. Study how gradient bucketing and ring all-reduce perform with moderate model sizes

**Key characteristics**: The model is large enough to benefit from distributed training while still having measurable communication overhead (~35% at 4 nodes), allowing us to analyze the efficiency trade-offs.

### Distributed System Configuration
- **Data Partitioning**: DistributedSampler (interleaved assignment)
- **Gradient Synchronization**: Ring All-Reduce algorithm
- **Communication**: Custom TCP-based tensor communication layer

---

## Experiment 1: Speedup and Convergence Analysis

**Goal**: Measure training time improvement and convergence speed as we increase the number of nodes.

### Setup
- **Fixed**: Dataset size (50K samples), epochs (15), batch size (128)
- **Variable**: World size = {1, 2, 3, 4}
- **Bucket Size**: 5MB for gradient bucketing

### Metrics to Collect
- **Training time per epoch** (seconds)
- **Total training time** (seconds)
- **Time to reach target accuracy** (40%, 45%, 50%)
- **Final test accuracy** (%)
- **Speedup**: S(n) = T(1) / T(n)
- **Parallel Efficiency**: E(n) = S(n) / n
- **Communication overhead**: % of time spent in all-reduce

### Expected Results
Given our medium model size (~6.63MB), we anticipate **moderate communication overhead**:
- **World Size 2**: Speedup ~1.4-1.5× (efficiency ~70-75%)
- **World Size 3**: Speedup ~1.9-2.1× (efficiency ~65-70%)
- **World Size 4**: Speedup ~2.2-2.4× (efficiency ~55-60%)
- **CIFAR-10 Performance**: Target accuracy ~50-55% with this simple fully-connected architecture
  - More nodes should reach target accuracy faster due to larger effective batch sizes
  - Communication overhead (~30-35%) should be offset by computational gains

**Note**: Sub-linear speedup is expected due to communication overhead and larger effective batch sizes in distributed training.

### Analysis
- Plot speedup curve (actual vs. ideal)
- Plot parallel efficiency vs. world size
- **Plot convergence time** for different accuracy targets (40%, 45%, 50%)
- Plot training curves (loss vs. time, accuracy vs. time)
- Calculate communication-to-computation ratio
- Identify break-even point where adding nodes becomes counterproductive

---

## Experiment 2: Communication Overhead Profiling

**Goal**: Quantify communication overhead in distributed training.

### Setup
- **Fixed**: 15 epochs, batch size 128
- **Variable**: World size = {1, 2, 3, 4}
- **Instrumentation**: Timers in ring_allreduce to measure communication time

### Metrics to Collect (per batch)
- **Total batch time** (seconds)
- **Communication time** (all-reduce, measured in hooks)
- **Computation time** (total - communication)
- **Communication percentage**: comm_time / total_time × 100%

### Expected Results
- Communication overhead increases with world size
- For 4 nodes: expect communication ~30-35% of total time
- Model size (6.63MB) provides reasonable computation-to-communication ratio

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
- Demonstrate measurable speedup for 2-4 nodes (even if sub-linear)
- Show faster convergence time with distributed training
- Quantify communication overhead (characterize trade-offs)
- Maintain comparable accuracy across different world sizes (±2% variation)
- Document the relationship between model size, communication overhead, and efficiency

### Stretch Goals
- Achieve >60% parallel efficiency with 4 nodes
- Implement learning rate scaling to improve convergence consistency
- Compare with PyTorch's DistributedDataParallel as reference
- Analyze impact of different bucket sizes on communication efficiency

---

## Notes

- Compare results with PyTorch's `torch.distributed` as reference (if time permits)
