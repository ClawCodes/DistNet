"""Analysis and visualization tools for distributed training experiments."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_result(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        return json.load(f)


def find_result_file(base_dir: str, world_size: int) -> Optional[str]:
    """Find result file for given world size."""
    base_path = Path(base_dir)

    patterns = [
        f'speedup_ws{world_size}/node0_*.json',
        f'*_ws{world_size}/node0_*.json',
    ]

    for pattern in patterns:
        matches = list(base_path.glob(pattern))
        if matches:
            return str(matches[0])

    for subdir in base_path.iterdir():
        if not subdir.is_dir():
            continue
        node_files = list(subdir.glob('node0_*.json'))
        if node_files:
            try:
                data = load_result(str(node_files[0]))
                if data.get('world_size') == world_size:
                    return str(node_files[0])
            except:
                continue

    return None


def analyze_speedup(results_dir: str, world_sizes: List[int]):
    """Analyze speedup and efficiency."""
    print("Analyzing speedup...")

    times = []
    for ws in world_sizes:
        filepath = find_result_file(results_dir, ws)
        if not filepath:
            print(f"Warning: No result for world size {ws}")
            return None, None

        data = load_result(filepath)
        total_time = sum(data['metrics']['epoch_times'])
        times.append(total_time)
        print(f"  WS {ws}: {total_time:.2f}s")

    speedup = [times[0] / t for t in times]
    efficiency = [s / ws * 100 for s, ws in zip(speedup, world_sizes)]

    print("\nSpeedup Analysis:")
    for ws, s, e in zip(world_sizes, speedup, efficiency):
        print(f"  WS {ws}: {s:.2f}x speedup, {e:.1f}% efficiency")

    return speedup, efficiency


def analyze_convergence(results_dir: str, world_sizes: List[int], targets: List[float]):
    """Analyze time to reach target accuracies."""
    print("\nAnalyzing convergence...")

    results = {t: [] for t in targets}

    for ws in world_sizes:
        filepath = find_result_file(results_dir, ws)
        if not filepath:
            for t in targets:
                results[t].append(None)
            continue

        data = load_result(filepath)
        accuracies = data['metrics']['accuracies']
        epoch_times = data['metrics']['epoch_times']

        for target in targets:
            cumulative = 0
            reached = None
            for acc, etime in zip(accuracies, epoch_times):
                cumulative += etime
                if acc >= target:
                    reached = cumulative
                    break
            results[target].append(reached)
            status = f"{reached:.1f}s" if reached else "not reached"
            print(f"  WS {ws} → {target*100:.0f}%: {status}")

    return results


def plot_speedup(world_sizes: List[int], speedup: List[float],
                efficiency: List[float], output: str):
    """Plot speedup and efficiency."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(world_sizes, speedup, 'o-', linewidth=2, markersize=8, label='Actual')
    ax1.plot(world_sizes, world_sizes, '--', linewidth=2, color='gray', label='Ideal')
    ax1.set_xlabel('World Size')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Speedup vs. World Size', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xticks(world_sizes)

    ax2.plot(world_sizes, efficiency, 'o-', linewidth=2, markersize=8, color='red')
    ax2.axhline(100, linestyle='--', color='gray', linewidth=2)
    ax2.set_xlabel('World Size')
    ax2.set_ylabel('Efficiency (%)')
    ax2.set_title('Parallel Efficiency', fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_xticks(world_sizes)
    ax2.set_ylim([0, 110])

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def plot_convergence(world_sizes: List[int], results: Dict[float, List], output: str):
    """Plot convergence time."""
    fig, ax = plt.subplots(figsize=(10, 6))

    targets = sorted(results.keys())
    x = np.arange(len(world_sizes))
    width = 0.25

    for i, target in enumerate(targets):
        times = [t if t else 0 for t in results[target]]
        ax.bar(x + width * (i - 1), times, width,
               label=f'{target*100:.0f}% accuracy', alpha=0.8)

    ax.set_xlabel('World Size')
    ax.set_ylabel('Time to Reach Target (s)')
    ax.set_title('Convergence Time vs. World Size', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(world_sizes)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def plot_training_curves(results_dir: str, world_sizes: List[int], output: str):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ws in world_sizes:
        filepath = find_result_file(results_dir, ws)
        if not filepath:
            continue

        data = load_result(filepath)
        epochs = range(1, len(data['metrics']['losses']) + 1)

        ax1.plot(epochs, data['metrics']['losses'],
                label=f'WS {ws}', linewidth=2)

        if any(data['metrics']['accuracies']):
            ax2.plot(epochs, [a * 100 for a in data['metrics']['accuracies']],
                    label=f'WS {ws}', linewidth=2)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Test Accuracy', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def plot_comm_overhead(world_sizes: List[int], results_dir: str, output: str):
    """Plot communication overhead."""
    comm_pcts = []

    for ws in world_sizes:
        filepath = find_result_file(results_dir, ws)
        if not filepath:
            return

        data = load_result(filepath)
        total = sum(data['metrics']['epoch_times'])
        comm = sum(data['metrics']['comm_times'])
        pct = (comm / total * 100) if total > 0 else 0
        comm_pcts.append(pct)
        print(f"  WS {ws}: {pct:.1f}% communication")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['green', 'orange', 'red', 'darkred']
    ax1.bar(world_sizes, comm_pcts, color=colors[:len(world_sizes)],
            alpha=0.7, edgecolor='black')
    ax1.set_xlabel('World Size')
    ax1.set_ylabel('Communication Overhead (%)')
    ax1.set_title('Communication Overhead', fontweight='bold')
    ax1.set_xticks(world_sizes)
    ax1.grid(alpha=0.3, axis='y')

    compute = 100 - comm_pcts[-1]
    comm = comm_pcts[-1]
    ax2.pie([compute, comm], labels=['Computation', 'Communication'],
           autopct='%1.1f%%', startangle=90, colors=['lightblue', 'coral'])
    ax2.set_title(f'Time Distribution (WS {world_sizes[-1]})', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def main():
    print("=" * 60)
    print("Distributed Training Experiment Analysis")
    print("=" * 60)

    results_dir = sys.argv[1] if len(sys.argv) > 1 else './results'

    if not Path(results_dir).exists():
        print(f"\nError: '{results_dir}' not found")
        print("\nUsage: python analyze.py [results_dir]")
        print("Example: python analyze.py runs")
        return

    print(f"\nAnalyzing: {results_dir}\n")

    world_sizes = [1, 2, 3, 4]
    targets = [0.95, 0.96, 0.97]

    speedup, efficiency = analyze_speedup(results_dir, world_sizes)
    if speedup:
        plot_speedup(world_sizes, speedup, efficiency,
                    f'{results_dir}/speedup_analysis.png')

    conv_results = analyze_convergence(results_dir, world_sizes, targets)
    plot_convergence(world_sizes, conv_results,
                    f'{results_dir}/convergence_time.png')

    print("\nCommunication overhead:")
    plot_comm_overhead(world_sizes, results_dir,
                      f'{results_dir}/comm_overhead.png')

    print("\nTraining curves:")
    plot_training_curves(results_dir, world_sizes,
                        f'{results_dir}/training_curves.png')

    print("\n" + "=" * 60)
    print(f"Complete! Check {results_dir} for plots")
    print("=" * 60)


if __name__ == '__main__':
    main()
