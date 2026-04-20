# plot_layerwise_ffn.py
import re
from collections import defaultdict
import matplotlib.pyplot as plt

LOG_PATH = "/Users/yoonji/Projects/lectures/csci-ga2565-pico-llm-mys/analyze_ffn.txt"
OUT_PATH = "ffn_layerwise_purity.png"

layer_neuron_re = re.compile(r"=== Layer (\d+), Neuron (\d+), Top-(\d+) triggers ===")
t0_header_re = re.compile(r"-- Top tokens at t\+0 --")
tok_count_re = re.compile(r"^\s*'(.+?)':\s*(\d+)/(\d+)\s*$")


def parse_purity(log_text: str):
    """
    Returns:
      purity_by_layer: dict layer -> list[purity]
      seen_layers: set of layers that appear in the log
    purity = max_count_at_t0 / k
    """
    purity_by_layer = defaultdict(list)
    seen_layers = set()

    lines = log_text.splitlines()
    i = 0
    cur_layer = None
    cur_k = None

    while i < len(lines):
        m = layer_neuron_re.search(lines[i])
        if m:
            cur_layer = int(m.group(1))
            cur_k = int(m.group(3))
            seen_layers.add(cur_layer)
            i += 1
            continue

        # find the local token freq block's t+0 section
        if cur_layer is not None and t0_header_re.search(lines[i]):
            i += 1
            max_count = 0
            denom = cur_k if cur_k is not None else 25

            # read token lines until blank or next header
            while i < len(lines):
                line = lines[i]
                if line.strip() == "" or line.strip().startswith("-- Top tokens at"):
                    break
                tm = tok_count_re.match(line)
                if tm:
                    count = int(tm.group(2))
                    denom = int(tm.group(3))  # trust log denominator
                    if count > max_count:
                        max_count = count
                i += 1

            if denom > 0:
                purity_by_layer[cur_layer].append(max_count / denom)

            # reset until next neuron block
            cur_layer = None
            cur_k = None
            continue

        i += 1

    return purity_by_layer, seen_layers


def main():
    with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    purity_by_layer, seen_layers = parse_purity(txt)

    if not purity_by_layer:
        print("No purity stats found. Check that analyze_ffn.txt contains 'Top tokens at t+0' blocks.")
        return

    # Infer N = number of layers from the max layer index we saw.
    # If layers are 0..(N-1), then N = max+1.
    max_layer = max(seen_layers)
    N = max_layer + 1

    # Match your original design: [0, N//2, N-1]
    layers_to_compare = [0, N // 2, N - 1]
    layers_to_compare = sorted(set(layers_to_compare))

    # Filter to only layers present in the log
    available = [L for L in layers_to_compare if L in purity_by_layer and len(purity_by_layer[L]) > 0]
    missing = [L for L in layers_to_compare if L not in available]

    if not available:
        print(f"Requested layers {layers_to_compare}, but none were found with purity values in the log.")
        print("Available layers in log:", sorted(purity_by_layer.keys()))
        return

    # Prepare data for plot
    data = [purity_by_layer[L] for L in available]

    plt.figure()
    plt.boxplot(data, labels=[str(L) for L in available], showfliers=True)

    means = [sum(v) / len(v) for v in data]
    plt.plot(range(1, len(available) + 1), means, marker="o", linestyle="")

    plt.xlabel("Layer")
    plt.ylabel("Token purity at t (max token freq / k)")
    plt.title("Layer-wise FFN Neuron Token Specificity (from logged top-k triggers)")

    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")
    print("Counts per layer:", {L: len(purity_by_layer[L]) for L in available})

    if missing:
        print(f"Note: requested layers {missing} were not found (or had no entries) in the log.")
        print("Layers found in log:", sorted(purity_by_layer.keys()))


if __name__ == "__main__":
    main()
