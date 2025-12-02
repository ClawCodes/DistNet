import json
import glob
import statistics
import matplotlib.pyplot as plt

def main():
    # Find all experiment result files
    files = glob.glob("results/scale_layers/first_run/node0_batch_128_epochs_10_buckets_15_layers_*_width512.json")
    files = sorted(files, key=lambda f: int(f.split("layers_")[-1].split("_")[0]))

    layer_labels = []
    compute_avgs = []
    comm_avgs = []

    for f in files:
        data = json.load(open(f))

        # Extract layer count from filename
        layers = int(f.split("layers_")[-1].split("_")[0])
        layer_labels.append(str(layers))

        # Compute averages from metrics
        compute_avg = statistics.mean(data["metrics"]["compute_times"])
        comm_avg = statistics.mean(data["metrics"]["comm_times"])

        compute_avgs.append(compute_avg)
        comm_avgs.append(comm_avg)

    x = range(len(layer_labels))

    plt.bar(x, compute_avgs, label="compute", color="b")
    plt.bar(x, comm_avgs, bottom=compute_avgs, label="comm", color="c")

    for i, (cmp, comm) in enumerate(zip(compute_avgs, comm_avgs)):
        total = cmp + comm
        pct_cmp = cmp / total * 100
        pct_comm = comm / total * 100
        plt.text(i, total - 10, f"C:{pct_cmp:.1f}%\nM:{pct_comm:.1f}%", ha='center')

    plt.xticks(x, layer_labels)
    plt.ylabel("Time per epoch (seconds)")
    plt.xlabel("Layer count (model depth)")
    plt.title("Compute vs Communication Time Scaling With Layer Count")

    # plt.show()
    plt.savefig("results/scale_layers/first_run/layer_scaling.png")

if __name__ == '__main__':
    main()