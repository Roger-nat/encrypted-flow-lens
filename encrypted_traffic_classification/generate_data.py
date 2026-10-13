"""Generate a clearly labeled synthetic network-flow dataset for a demo."""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
OUT = Path(__file__).parent / "data" / "synthetic_network_flows.csv"

def main():
    rng = np.random.default_rng(SEED)
    classes = ["benign", "suspicious"]
    n_each = 1200
    rows = []
    for label in classes:
        n = n_each
        if label == "benign":
            duration = rng.lognormal(mean=2.2, sigma=0.65, size=n)
            fwd_pkts = rng.poisson(18, size=n) + 1
            bwd_pkts = rng.poisson(15, size=n) + 1
            fwd_bytes = fwd_pkts * rng.lognormal(5.0, 0.55, size=n)
            bwd_bytes = bwd_pkts * rng.lognormal(4.8, 0.60, size=n)
            iat = rng.lognormal(3.0, 0.65, size=n)
            dst_ports = rng.choice([443, 80, 53, 123, 993, 5228], size=n)
            syn_ratio = rng.beta(2, 8, size=n)
        else:
            duration = rng.lognormal(mean=1.6, sigma=0.9, size=n)
            fwd_pkts = rng.poisson(45, size=n) + 1
            bwd_pkts = rng.poisson(5, size=n) + 1
            fwd_bytes = fwd_pkts * rng.lognormal(4.2, 0.9, size=n)
            bwd_bytes = bwd_pkts * rng.lognormal(3.7, 0.9, size=n)
            iat = rng.lognormal(1.9, 0.9, size=n)
            dst_ports = rng.choice([443, 80, 4444, 8080, 9001, 53], size=n, p=[.20, .15, .2, .2, .15, .1])
            syn_ratio = rng.beta(5, 3, size=n)
        for i in range(n):
            total_pkts = int(fwd_pkts[i] + bwd_pkts[i])
            rows.append({
                "flow_duration_ms": round(float(duration[i] * 100), 3),
                "fwd_packets": int(fwd_pkts[i]),
                "bwd_packets": int(bwd_pkts[i]),
                "fwd_bytes": round(float(max(1, fwd_bytes[i])), 2),
                "bwd_bytes": round(float(max(1, bwd_bytes[i])), 2),
                "packet_rate": round(float(total_pkts / max(duration[i], 0.01)), 4),
                "mean_interarrival_ms": round(float(iat[i]), 4),
                "destination_port": int(dst_ports[i]),
                "syn_flag_ratio": round(float(syn_ratio[i]), 4),
                "bytes_per_packet": round(float((fwd_bytes[i] + bwd_bytes[i]) / total_pkts), 4),
                "label": label
            })
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} synthetic flows to {OUT}")

if __name__ == "__main__":
    main()
