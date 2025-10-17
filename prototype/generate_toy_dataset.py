# prototype/generate_toy_dataset.py
"""
Toy dataset generator for the prototype pipeline.

Generates:
- item_emb.npy : item embeddings (n_items x emb_dim)
- price_mean.npy : per-item scalar price mean (n_items,)
- price_cov.npy : per-item scalar price variance (n_items,)
- edges.npy : adjacency list (list of edge tuples)
- items.txt : human-readable item ids (one per line)

Usage:
    python prototype/generate_toy_dataset.py --n_items 200 --emb_dim 64
"""
import argparse
import numpy as np
import os
import random

def generate_toy(n_items=200, emb_dim=64, edge_prob=0.02, out_dir="toy_data"):
    os.makedirs(out_dir, exist_ok=True)
    # item embeddings (simulate MMSBR.item_emb_final)
    item_emb = np.random.normal(scale=0.5, size=(n_items, emb_dim)).astype(np.float32)
    # price mean and cov (simulate MMSBR price mean/cov)
    price_mean = np.random.uniform(10, 500, size=(n_items,)).astype(np.float32)
    price_cov = np.random.uniform(1, 20, size=(n_items,)).astype(np.float32)
    # simple graph: random edges with a slight locality structure
    edges = []
    for i in range(n_items):
        for j in range(i+1, n_items):
            if random.random() < edge_prob:
                edges.append((i, j))
                edges.append((j, i))
    # save items
    items = [f"item_{i}" for i in range(n_items)]
    np.save(os.path.join(out_dir, "item_emb.npy"), item_emb)
    np.save(os.path.join(out_dir, "price_mean.npy"), price_mean)
    np.save(os.path.join(out_dir, "price_cov.npy"), price_cov)
    # edges as array shape (E,2)
    if len(edges) > 0:
        np.save(os.path.join(out_dir, "edges.npy"), np.array(edges, dtype=np.int32))
    else:
        np.save(os.path.join(out_dir, "edges.npy"), np.zeros((0,2), dtype=np.int32))
    with open(os.path.join(out_dir, "items.txt"), "w") as f:
        for it in items:
            f.write(it + "\n")
    print(f"Generated toy data in {out_dir}: n_items={n_items}, emb_dim={emb_dim}, edges={len(edges)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_items", type=int, default=200)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--out_dir", type=str, default="toy_data")
    args = parser.parse_args()
    generate_toy(n_items=args.n_items, emb_dim=args.emb_dim, out_dir=args.out_dir)
