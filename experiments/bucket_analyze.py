"""Analyze bucket size impact on communication overhead for ResNet experiments."""

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_result(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_bucket_experiments(results_dir: str, bucket_sizes: List[int]):
    """Analyze ResNet bucket size experiments."""

    data = {
        'bucket_sizes': [],
        'avg_comm_time': [],
        'comm_overhead_pct': []
    }

    for bucket_size in bucket_sizes:
        bucket_dir = Path(results_dir) / f'resnet_bucket{bucket_size}_ws4'

        if not bucket_dir.exists():
            print(f"Warning: {bucket_dir} not found, skipping bucket_size={bucket_size}")
            continue

        # Load node0 result
        result_files = list(bucket_dir.glob('node0_*.json'))
        if not result_files:
            print(f"Warning: No result file in {bucket_dir}")
            continue

        result = load_result(str(result_files[0]))

        # Calculate metrics
        comm_times = result['metrics']['comm_times']
        epoch_times = result['metrics']['epoch_times']

        avg_comm = np.mean(comm_times)
        avg_total = np.mean(epoch_times)
        comm_pct = (avg_comm / avg_total * 100) if avg_total > 0 else 0

        data['bucket_sizes'].append(bucket_size)
        data['avg_comm_time'].append(avg_comm)
        data['comm_overhead_pct'].append(comm_pct)

        print(f"Bucket {bucket_size}MB:")
        print(f"  Avg comm time:  {avg_comm:.2f}s")
        print(f"  Avg total time: {avg_total:.2f}s")
        print(f"  Comm overhead:  {comm_pct:.2f}%")
        print()

    return data


def plot_bucket_analysis(data: Dict, output: str):
    """Plot bucket size vs communication metrics: bar chart + line chart with dual y-axis."""

    fig, ax1 = plt.subplots(figsize=(10, 6))

    bucket_sizes = data['bucket_sizes']
    x_pos = np.arange(len(bucket_sizes))

    # Left y-axis: Communication Time
    color1 = '#2171b5'
    bars = ax1.bar(x_pos, data['avg_comm_time'],
                   label='Communication Time', color=color1, alpha=0.8, width=0.6)
    ax1.set_xlabel('Bucket Size (MB)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Communication Time per Epoch (s)', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{bs}' for bs in bucket_sizes])

    # Set y-axis limits for left axis (communication time)
    ax1.set_ylim([0, 100])  # 0 to max + 15% padding

    # Annotate communication time values on bars
    for bar, val in zip(bars, data['avg_comm_time']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}s',
                ha='center', va='bottom', fontsize=10, color=color1, fontweight='bold')

    # Right y-axis: Communication Overhead Percentage (line chart)
    ax2 = ax1.twinx()
    color2 = "#661a13"
    line = ax2.plot(x_pos, data['comm_overhead_pct'], 'o-',
                    linewidth=2.5, markersize=10, color=color2,
                    label='Communication Overhead %')
    ax2.set_ylabel('Communication Overhead (%)', fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Set y-axis limits for right axis (percentage)
    ax2.set_ylim([16, 21])  # 0 to max + 15% padding

    # Annotate overhead percentage values on line
    for x, y in zip(x_pos, data['comm_overhead_pct']):
        ax2.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=10,
                    color=color2, fontweight='bold')

    # Title and legend
    ax1.set_title('Bucket Size Impact on Communication', fontweight='bold', fontsize=14, pad=20)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)

    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output}")


def main():
    print("=" * 60)
    print("ResNet Bucket Size Analysis")
    print("=" * 60)
    print()

    results_dir = sys.argv[1] if len(sys.argv) > 1 else './runs'

    if not Path(results_dir).exists():
        print(f"Error: '{results_dir}' not found")
        return

    bucket_sizes = [3, 5, 10, 15]

    data = analyze_bucket_experiments(results_dir, bucket_sizes)

    if not data['bucket_sizes']:
        print("Error: No valid data found")
        return

    output_file = f'{results_dir}/bucket_size_analysis.png'
    plot_bucket_analysis(data, output_file)

    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
