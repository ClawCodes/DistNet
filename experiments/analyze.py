"""Analysis and visualization tools for distributed training experiments."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List


class ExperimentLogger:
    """
    Logger for distributed training experiments.

    Communication time measurement (for training script):
        def ring_allreduce_hook(name, grad):
            start_time = time.perf_counter()
            reduced_grad = ring_allreduce(grad)
            end_time = time.perf_counter()
            comm_time = end_time - start_time

            global epoch_comm_time
            epoch_comm_time += comm_time

            return reduced_grad

        # Then pass epoch_comm_time to log_epoch(comm_time=...)

        compute_time = epoch_time - comm_time
    """

    def __init__(self, rank: int, world_size: int, experiment_name: str):
        self.rank = rank
        self.world_size = world_size
        self.experiment_name = experiment_name
        self.metrics = {
            'epoch_times': [],
            'losses': [],
            'accuracies': [],
            'comm_times': [],
            'compute_times': []
        }

    def log_epoch(self, epoch: int, loss: float, accuracy: float = 0.0,
                  epoch_time: float = 0.0, comm_time: float = 0.0):
        self.metrics['epoch_times'].append(epoch_time)
        self.metrics['losses'].append(loss)
        self.metrics['accuracies'].append(accuracy)
        self.metrics['comm_times'].append(comm_time)
        self.metrics['compute_times'].append(epoch_time - comm_time)

    def save(self, output_dir: str = './results'):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filename = f'{output_dir}/{self.experiment_name}_ws{self.world_size}_rank{self.rank}.json'

        data = {
            'experiment': self.experiment_name,
            'world_size': self.world_size,
            'rank': self.rank,
            'metrics': self.metrics
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[Rank {self.rank}] Saved results to {filename}")

    @staticmethod
    def load(filepath: str) -> Dict:
        with open(filepath, 'r') as f:
            return json.load(f)


def analyze_speedup(results_dir: str = './results'):
    """Analyze speedup from experiments with different world sizes."""
    print("Analyzing speedup experiment...")

    world_sizes = [1, 2, 3, 4]
    training_times = []

    for ws in world_sizes:
        filepath = f'{results_dir}/speedup_ws{ws}_rank0.json'
        try:
            data = ExperimentLogger.load(filepath)
            total_time = sum(data['metrics']['epoch_times'])
            training_times.append(total_time)
            print(f"World size {ws}: {total_time:.2f}s")
        except FileNotFoundError:
            print(f"Warning: {filepath} not found, skipping...")
            return

    baseline_time = training_times[0]

    # eg. 1.0x, 1.5x, 1.8x
    speedup = [baseline_time / t for t in training_times]           
    
    # 100% (1.0x/1), 75% (1.5x/2), 60% (1.8x/3)
    # Ideal: 100% (linear)
    efficiency = [s / ws for s, ws in zip(speedup, world_sizes)]

    print("\nSpeedup Analysis:")
    for ws, s, e in zip(world_sizes, speedup, efficiency):
        print(f"  World size {ws}: Speedup={s:.2f}x, Efficiency={e*100:.1f}%")

    plot_speedup_curve(world_sizes, speedup, efficiency, 
                       output_file=f'{results_dir}/speedup_analysis.png')


def analyze_convergence_time(results_dir: str = './results', experiment: str = 'speedup'):
    """Analyze time to reach target accuracy thresholds."""
    print("\nAnalyzing convergence time...")

    world_sizes = [1, 2, 3, 4]
    target_accuracies = [0.95, 0.96, 0.97]

    """
    results = {
       0.95: [],  # Store the time to reach 95% for each world size
       0.96: [],  # Store the time to reach 96% for each world size
       0.97: []   # Store the time to reach 97% for each world size
    }
    """
    results = {target: [] for target in target_accuracies}

    for ws in world_sizes:
        filepath = f'{results_dir}/{experiment}_ws{ws}_rank0.json'
        try:
            data = ExperimentLogger.load(filepath)
            accuracies = data['metrics']['accuracies']
            epoch_times = data['metrics']['epoch_times']

            for target in target_accuracies:
                time_to_target = None
                cumulative_time = 0
                for acc, etime in zip(accuracies, epoch_times):
                    cumulative_time += etime
                    if acc >= target:
                        time_to_target = cumulative_time
                        break
                results[target].append(time_to_target)

                status = f"{time_to_target:.2f}s" if time_to_target else "not reached"
                print(f"World size {ws} → {target*100:.0f}% accuracy: {status}")
                """
                ex:
                World size 1 → 95% accuracy: 25.0s
                World size 1 → 96% accuracy: 30.5s...

                World size 2 → 95% accuracy: 18.5s...
                
                World size 3 → 96% accuracy: 17.1s
                World size 3 → 97% accuracy: not reached (shouldn't happen)
                """

        except FileNotFoundError:
            print(f"Warning: {filepath} not found, skipping...")
            for target in target_accuracies:
                results[target].append(None)

    plot_convergence_time(world_sizes, results, 
                          output_file=f'{results_dir}/convergence_time.png')


def analyze_communication_overhead(results_dir: str = './results'):
    """Analyze communication overhead from profiling experiment."""
    print("\nAnalyzing communication overhead...")

    world_sizes = [1, 2, 3, 4]
    comm_percentages = []

    for ws in world_sizes:
        filepath = f'{results_dir}/profiling_ws{ws}_rank0.json'
        try:
            data = ExperimentLogger.load(filepath)
            total_time = sum(data['metrics']['epoch_times'])
            comm_time = sum(data['metrics']['comm_times'])
            comm_pct = (comm_time / total_time * 100) if total_time > 0 else 0
            comm_percentages.append(comm_pct)
            print(f"World size {ws}: Communication overhead = {comm_pct:.1f}%")
        except FileNotFoundError:
            print(f"Warning: {filepath} not found, skipping...")
            return

    plot_communication_overhead(world_sizes, comm_percentages,
                               output_file=f'{results_dir}/comm_overhead.png')


def plot_speedup_curve(world_sizes: List[int], speedup: List[float],
                      efficiency: List[float], output_file: str = 'speedup.png'):
    """Plot speedup and efficiency curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(world_sizes, speedup, 'o-', linewidth=2, markersize=8, label='Actual Speedup')
    ax1.plot(world_sizes, world_sizes, '--', linewidth=2, color='gray', label='Ideal Speedup')
    ax1.set_xlabel('World Size (Number of Nodes)', fontsize=12)
    ax1.set_ylabel('Speedup', fontsize=12)
    ax1.set_title('Speedup vs. World Size', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(world_sizes)

    ax2.plot(world_sizes, [e * 100 for e in efficiency], 'o-',
            linewidth=2, markersize=8, color='red')
    ax2.axhline(y=100, linestyle='--', color='gray', linewidth=2, label='100% Efficiency')
    ax2.set_xlabel('World Size (Number of Nodes)', fontsize=12)
    ax2.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax2.set_title('Parallel Efficiency vs. World Size', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(world_sizes)
    ax2.set_ylim([0, 110])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved speedup analysis to {output_file}")
    plt.close()


def plot_communication_overhead(world_sizes: List[int], comm_percentages: List[float],
                                output_file: str = 'comm_overhead.png'):
    """Plot communication overhead percentage."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['green', 'orange', 'red']
    ax1.bar(world_sizes, comm_percentages, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('World Size (Number of Nodes)', fontsize=12)
    ax1.set_ylabel('Communication Overhead (%)', fontsize=12)
    ax1.set_title('Communication Overhead vs. World Size', fontsize=14, fontweight='bold')
    ax1.set_xticks(world_sizes)
    ax1.grid(True, alpha=0.3, axis='y')

    largest_ws_idx = -1
    compute_pct = 100 - comm_percentages[largest_ws_idx]
    comm_pct = comm_percentages[largest_ws_idx]

    ax2.pie([compute_pct, comm_pct], labels=['Computation', 'Communication'],
           autopct='%1.1f%%', startangle=90, colors=['lightblue', 'coral'])
    ax2.set_title(f'Time Distribution (World Size {world_sizes[largest_ws_idx]})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved communication overhead analysis to {output_file}")
    plt.close()


def plot_convergence_time(world_sizes: List[int], results: Dict[float, List],
                          output_file: str = 'convergence_time.png'):
    """Plot time to reach target accuracy thresholds."""
    fig, ax = plt.subplots(figsize=(10, 6))

    target_accuracies = sorted(results.keys())
    x = np.arange(len(world_sizes))
    width = 0.25

    for i, target in enumerate(target_accuracies):
        times = results[target]
        times_filtered = [t if t is not None else 0 for t in times]
        offset = width * (i - 1)
        ax.bar(x + offset, times_filtered, width,
               label=f'{target*100:.0f}% accuracy', alpha=0.8)

    ax.set_xlabel('World Size (Number of Nodes)', fontsize=12)
    ax.set_ylabel('Time to Reach Target (seconds)', fontsize=12)
    ax.set_title('Convergence Time vs. World Size', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(world_sizes)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved convergence time analysis to {output_file}")
    plt.close()


def plot_training_curves(results_dir: str = './results', experiment: str = 'speedup'):
    """Plot training loss and accuracy curves."""
    print(f"\nPlotting training curves for {experiment}...")

    world_sizes = [1, 2, 3, 4]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ws in world_sizes:
        filepath = f'{results_dir}/{experiment}_ws{ws}_rank0.json'
        try:
            data = ExperimentLogger.load(filepath)
            epochs = range(1, len(data['metrics']['losses']) + 1)

            ax1.plot(epochs, data['metrics']['losses'],
                    label=f'World Size {ws}', linewidth=2)

            if any(data['metrics']['accuracies']):
                ax2.plot(epochs, [a * 100 for a in data['metrics']['accuracies']],
                        label=f'World Size {ws}', linewidth=2)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found, skipping...")
            continue

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss vs. Epoch', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Test Accuracy vs. Epoch', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = f'{results_dir}/training_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {output_file}")
    plt.close()


if __name__ == '__main__':
    print("="*60)
    print("Distributed Training Experiment Analysis")
    print("="*60)

    results_dir = './results'

    if not Path(results_dir).exists():
        print(f"\nError: Results directory '{results_dir}' not found.")
        print("Please run experiments first to generate results.")
        exit(1)

    analyze_speedup(results_dir)
    analyze_convergence_time(results_dir, experiment='speedup')
    analyze_communication_overhead(results_dir)
    plot_training_curves(results_dir, experiment='speedup')

    print("\n" + "="*60)
    print("Analysis complete! Check the results directory for plots.")
    print("="*60)
