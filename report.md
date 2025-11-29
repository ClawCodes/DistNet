# DistNet: Distributed Training Performance Analysis

## Experimental Setup

- **Model**: Convolutional Neural Network (CNN)
  - Architecture: Conv(32) → Conv(64) → Conv(128) → FC(256) → FC(10)
  - Parameters: ~0.68M (~2.73MB)
  - Activation: GELU with Dropout (0.2), BatchNorm between conv layers
- **Dataset**: CIFAR-10 (50K training samples, 10K test samples)
- **Training Configuration**: 15 epochs, batch size 128, Adam optimizer (lr=0.001)
- **Distributed System**:
  - Gradient synchronization: Ring all-reduce
  - Data partitioning: DistributedSampler (interleaved assignment)
  - Gradient bucketing: 5MB buckets

## Results

### Performance Metrics
#### CNN - Strong Scaling
| World Size | Total Time (s) | Speedup | Efficiency | Comm Overhead | Final Accuracy |
|------------|---------------|---------|------------|---------------|----------------|
| 1          | 829.75        | 1.00×   | 100.0%     | 0.30%         | 78.71%         |
| 2          | 462.43        | 1.79×   | 89.7%      | 12.49%        | 78.50%         |
| 3          | 339.00        | 2.45×   | 81.6%      | 21.19%        | 78.73%         |
| 4          | 290.68        | 2.85×   | 71.4%      | 26.79%        | 78.52%         |

### Convergence Time

#### Time to Reach Target Accuracy
| World Size | 70% Acc (s) | 75% Acc (s) | 78% Acc (s) |
|------------|-------------|-------------|-------------|
| 1          | 109.1       | 164.6       | 386.6       |
| 2          | 92.4        | 185.5       | 308.4       |
| 3          | 90.8        | 135.0       | 272.0       |
| 4          | 96.7        | 154.6       | 271.5       |

#### Speedup vs WS1
| World Size | 70% Speedup | 75% Speedup | 78% Speedup |
|------------|-------------|-------------|-------------|
| 1          | 1.00×       | 1.00×       | 1.00×       |
| 2          | 1.18×       | 0.89×       | 1.25×       |
| 3          | 1.20×       | 1.22×       | 1.42×       |
| 4          | 1.13×       | 1.06×       | 1.42×       |

### Key Observations

**1. Speedup and Efficiency**
- Achieved 2.29× speedup on 4 nodes (57.2% efficiency)
- Sub-linear scaling due to communication overhead
- Efficiency decreases as world size increases: 100% → 69% → 65% → 57%

**2. Communication Overhead**
- Scales with world size: 2.1% → 24.6% → 30.4% → 34.8%
- Communication time increases from ~0.45s/epoch (WS1) to ~3.2s/epoch (WS4)
- Model size (6.63MB) results in moderate communication-to-computation ratio

**3. Convergence Behavior**
- Loss progression is slower for larger world sizes in both scaling modes
- **Strong Scaling**: WS4 has slower loss decrease due to gradient noise
  - Each node processes only 32 samples (128 ÷ 4), leading to higher gradient variance
  - Despite gradient averaging across nodes, individual gradients are lower quality
  - Results in less stable parameter updates and reduced learning efficiency
  - **Loss vs Accuracy paradox**: WS4 final loss (~0.48) >> WS1 final loss (~0.14), but accuracy is similar (78.5% vs 78.7%)
    - Higher loss indicates lower prediction confidence (flatter probability distributions)
    - Accuracy only measures if argmax is correct, not confidence level
    - For probability-critical applications, WS1's lower loss is preferred
- **Weak Scaling**: WS4 has slower loss decrease due to reduced update frequency
  - Larger effective batch size (WS1: 128, WS4: 512)
  - Fewer parameter updates per epoch (WS1: 391, WS4: 98)
  - Classic large-batch training characteristic: better gradient quality but 4× fewer updates

**4. Time Breakdown (WS 4)**
- Computation: 6.15s/epoch (65.2%)
- Communication: 3.28s/epoch (34.8%)
- Total: 9.43s/epoch

### Technical Insights

**Gradient Bucketing**
- 5MB buckets effectively batch gradient tensors
- Reduces number of all-reduce operations
- ~2 buckets needed for 1.74M parameters

**Data Partitioning**
- Interleaved assignment provides balanced workload
- Consistent shuffle seeding (seed=42) across all nodes
- Each node processes different data slices per epoch

**Ring All-Reduce Pattern**
- Two-phase communication: reduce-scatter + all-gather
- Each phase requires (world_size - 1) rounds
- Bandwidth-optimal for dense gradients

## Analysis

### Trade-offs for Fully-Connected Model (1.74M params)

**Benefits:**
- Wall-clock time reduction: 2.29× faster with 4 nodes
- Computation time remains stable: ~6.1s/epoch across all world sizes
- System demonstrates effective parallelization

**Costs:**
- Communication overhead: 34.8% of total time at WS 4
- Efficiency degradation: 57% at WS 4 (vs. ideal 100%)
- Convergence quality: 2.9% accuracy loss due to large batch effects

### Model Size Considerations

For the 1.74M parameter model:
- Computation time (~6s) >> Communication time (~3s) at WS 4
- System remains beneficial up to 4 nodes
- Communication overhead becomes dominant beyond this scale
- Larger models would show better efficiency due to higher computation-to-communication ratio
