# https://github.com/your/repo/blob/main/prototype/prototype_pipeline.py
"""
Prototype pipeline:
- Loads MMSBR-like item embeddings and price stats (toy_data/)
- Runs a simple graph refinement (one mean-aggregation GNN step)
- Encodes session textual context (sentence-transformers if available)
- Produces a query vector from session item history + text context
- Retrieves top-N items using brute-force dot product
- Reranks using a simple price-modulation score (inspired by MMSBR's price modulation)
- Produces a short templated explanation (can be replaced by an LLM call)

Usage example:
python prototype/prototype_pipeline.py --topk 5 --session "0,1,2" --text "Budget phone under $300"
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn

# optional: sentence-transformers for text embeddings
try:
    from sentence_transformers import SentenceTransformer
    SB_AVAILABLE = True
except Exception:
    SB_AVAILABLE = False

def load_toy_data(data_dir="toy_data"):
    item_emb = np.load(os.path.join(data_dir, "item_emb.npy"))
    price_mean = np.load(os.path.join(data_dir, "price_mean.npy"))
    price_cov = np.load(os.path.join(data_dir, "price_cov.npy"))
    edges = np.load(os.path.join(data_dir, "edges.npy"))
    with open(os.path.join(data_dir, "items.txt")) as f:
        items = [l.strip() for l in f.readlines()]
    return item_emb, price_mean, price_cov, edges, items

class SimpleGNN(nn.Module):
    """
    Simple 1-layer mean aggregator:
        h_i' = W_self h_i + W_neigh mean_{j in N(i)} h_j
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.w_self = nn.Linear(input_dim, hidden_dim, bias=True)
        self.w_neigh = nn.Linear(input_dim, hidden_dim, bias=True)
        self.act = nn.ReLU()

    def forward(self, item_emb, edges):
        # item_emb: (n_items, dim)
        n_items = item_emb.shape[0]
        device = item_emb.device
        # compute neighbor means
        neigh_sum = torch.zeros_like(item_emb)
        neigh_count = torch.zeros((n_items, 1), device=device)
        if edges.shape[0] > 0:
            src = torch.LongTensor(edges[:,0]).to(device)
            dst = torch.LongTensor(edges[:,1]).to(device)
            # accumulate
            neigh_sum.index_add_(0, src, item_emb[dst])
            ones = torch.ones((edges.shape[0],1), device=device)
            neigh_count.index_add_(0, src, ones)
        # avoid divide-by-zero
        mask = neigh_count.squeeze(1) > 0
        neigh_mean = torch.zeros_like(item_emb)
        neigh_mean[mask] = neigh_sum[mask] / neigh_count[mask]
        out = self.w_self(item_emb) + self.w_neigh(neigh_mean)
        out = self.act(out)
        # L2 normalize
        out = out / (out.norm(p=2, dim=1, keepdim=True) + 1e-9)
        return out

def text_encode(text, sb_model=None, emb_dim=64):
    if sb_model is not None:
        vec = sb_model.encode([text], convert_to_numpy=True)[0]
        # project / normalize if needed
        if vec.ndim == 1:
            vec = vec.astype(np.float32)
        return vec
    else:
        # fallback: deterministic pseudo-random vector based on text hash
        rng = np.random.RandomState(abs(hash(text)) % (2**32))
        return rng.normal(size=(emb_dim,)).astype(np.float32)

def compute_session_vector(session_item_ids, item_emb_gnn, text_vec=None, use_text=True):
    # session_item_ids: list of ints (indices)
    if len(session_item_ids) == 0:
        sess_vec = item_emb_gnn.mean(axis=0)
    else:
        # mean of last-k (here we use all provided)
        sess_vec = item_emb_gnn[session_item_ids].mean(axis=0)
    if use_text and text_vec is not None:
        # simple concat and linear projection
        combined = np.concatenate([sess_vec, text_vec], axis=0)
        # project down to same dim
        # We'll create a simple random linear map for prototype
        D = item_emb_gnn.shape[1]
        W = np.ones((combined.shape[0], D), dtype=np.float32) * (1.0 / combined.shape[0])
        q = combined.dot(W)
        q = q / (np.linalg.norm(q) + 1e-9)
        return q
    else:
        q = sess_vec / (np.linalg.norm(sess_vec) + 1e-9)
        return q

def price_modulation(sess_price_mean, item_price_mean):
    # Simple scalar similarity: higher when closer in price
    # Inspired by MMSBR price influence (not an exact Wasserstein implementation)
    # Scale into [0,1]
    diff = np.abs(sess_price_mean - item_price_mean)
    # e.g., similarity = exp(-diff / scale)
    scale = 50.0
    sim = np.exp(-diff / scale)
    return sim

def rerank_and_explain(query_vec, item_emb_gnn, price_mean, session_price_mean, items, topN=10):
    # brute-force dot product retrieval
    scores = item_emb_gnn.dot(query_vec)
    # basic top-N
    top_idx = np.argsort(-scores)[:topN]
    results = []
    for idx in top_idx:
        interest = float(scores[idx])
        pscore = price_modulation(session_price_mean, float(price_mean[idx]))
        final_score = interest + interest * pscore
        results.append({
            "idx": int(idx),
            "item_id": items[idx],
            "interest": float(interest),
            "price_score": float(pscore),
            "final_score": float(final_score),
            "price": float(price_mean[idx])
        })
    # sort by final_score
    results = sorted(results, key=lambda x: -x["final_score"])
    # generate templated explanations
    explanations = []
    for r in results:
        reason = f"Selected because content similarity score {r['interest']:.3f} and price match ({r['price']:.0f}$) score {r['price_score']:.3f).}"
        # Slightly more readable template:
        reason = (f"Item {r['item_id']} scored well: content similarity {r['interest']:.3f}, "
                  f"price ~${r['price']:.0f} matches session preference (mod {r['price_score']:.3f}).")
        explanations.append(reason)
    return results, explanations

def main(args):
    item_emb, price_mean, price_cov, edges, items = load_toy_data(args.data_dir)
    print(f"Loaded {item_emb.shape[0]} items, embedding dim {item_emb.shape[1]}")
    device = torch.device("cpu")
    item_emb_t = torch.from_numpy(item_emb).to(device)
    edges_np = edges.astype(np.int32)
    # compute GNN-refined embeddings
    gnn = SimpleGNN(input_dim=item_emb.shape[1], hidden_dim=item_emb.shape[1])
    with torch.no_grad():
        item_emb_gnn_t = gnn(item_emb_t, edges_np)
    item_emb_gnn = item_emb_gnn_t.cpu().numpy()

    # text encoder
    sb_model = None
    if args.use_sentence_transformers and SB_AVAILABLE:
        sb_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Loaded sentence-transformers model for text encoding.")
    elif args.use_sentence_transformers and not SB_AVAILABLE:
        print("sentence-transformers not installed. Falling back to random deterministic text vectors.")

    # parse session
    if args.session:
        session_item_ids = [int(x.strip()) for x in args.session.split(",") if x.strip() != ""]
        session_item_ids = [min(max(0, i), len(items)-1) for i in session_item_ids]
    else:
        session_item_ids = [0,1,2]  # default
    # get session price mean (toy: mean price of items in session)
    session_price_mean = float(price_mean[session_item_ids].mean()) if len(session_item_ids)>0 else float(price_mean.mean())

    # text context
    text_vec = None
    if args.text:
        text_vec = text_encode(args.text, sb_model=sb_model if sb_model else None, emb_dim=item_emb.shape[1]//2)
        # if sb returned bigger dim, reduce via simple truncation or pad
        if text_vec.shape[0] != item_emb.shape[1]//2:
            # adjust to length item_emb/2
            rng = np.random.RandomState(0)
            text_vec = rng.normal(size=(item_emb.shape[1]//2,)).astype(np.float32)

    # build query
    q = compute_session_vector(session_item_ids, item_emb_gnn, text_vec=text_vec, use_text=(text_vec is not None))
    # retrieve and rerank
    results, explanations = rerank_and_explain(q, item_emb_gnn, price_mean, session_price_mean, items, topN=args.topk)
    print("\nTop results:")
    for i, r in enumerate(results):
        print(f"{i+1:2d}. {r['item_id']}\tfinal_score={r['final_score']:.4f}\tprice=${r['price']:.1f}")
    print("\nExplanations (templated):")
    for i, ex in enumerate(explanations[:args.topk]):
        print(f"{i+1}. {ex}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="toy_data")
    parser.add_argument("--session", type=str, default="0,1,2",
                        help="comma separated item indices for the current session")
    parser.add_argument("--text", type=str, default="", help="textual context/intent")
    parser.add_argument("--use_sentence_transformers", action="store_true", help="use SBERT if available")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    main(args)
