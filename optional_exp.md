## Experiment 4: Scaling Analysis

**Goal**: Evaluate strong scaling and weak scaling properties.

### 4a. Strong Scaling
- **Fixed**: Total dataset size (60K samples)
- **Variable**: World size = {1, 2, 4, 8}
- Each node processes fewer samples as world size increases
- Measure: Speedup and efficiency

### 4b. Weak Scaling
- **Fixed**: Dataset size per node (15K samples per node)
- **Variable**: World size = {1, 2, 4, 8}
- Total dataset size grows with world size (15K, 30K, 60K, 120K)
- Measure: Time per epoch (should remain constant if perfectly scalable)

### Expected Results
- Strong scaling: Diminishing speedup due to Amdahl's law
- Weak scaling: Time per epoch increases slightly due to communication overhead

---

## Experiment 5: Network Traffic Analysis

**Goal**: Measure actual network usage during training.

### Metrics to Collect
- **Bytes sent/received per node** per epoch
- **Total network traffic**
- **Peak bandwidth utilization**
- **Number of messages exchanged**

### Analysis
- Calculate theoretical minimum traffic (parameter size × world_size)
- Compare actual vs. theoretical
- Identify inefficiencies in communication protocol

---

## Experiment 6: Model Parallelism Comparison

**Goal**: Compare data parallelism vs. model parallelism for different task complexities.

### Setup
- **Simple task**: QMNIST (current model)
- **Complex task**: CIFAR-10 with deeper network (if time permits)
- **Strategies**:
  - Data parallelism (current implementation)
  - Pipeline parallelism (split layers across nodes)
  - Tensor parallelism (split parameters across nodes)

### Metrics to Collect
- Training time
- Communication overhead
- Final accuracy
- Memory usage per node

### Expected Results
- Simple tasks: Data parallelism likely better (model is small)
- Complex tasks: Model parallelism may be necessary (model doesn't fit in one node)

---

## Notes

These experiments would strengthen the project but are not essential for answering the core research questions. Prioritize Experiments 1-3 first.
