import pandas as pd #2.2.3
import networkx as nx #3.5
import numpy as np #1.26.4
import faiss #1.12.0
from sentence_transformers import SentenceTransformer #4.1.0
from tqdm import tqdm #4.67.1

def build_graph_from_csv(path):
    df = pd.read_csv(path)
    G = nx.MultiDiGraph()  
    for _, row in df.iterrows():
        s, p, o = str(row['subject']).strip(), str(row['predicate']).strip(), str(row['object']).strip()
        # Add nodes with optional labels/metadata
        if not G.has_node(s):
            G.add_node(s, label=s)
        if not G.has_node(o):
            G.add_node(o, label=o)
        # Add edge with predicate as key/attr
        G.add_edge(s, o, key=p, predicate=p)
    return G,df

def triples_to_texts(df):
    texts = []
    for _, r in df.iterrows():
        texts.append(f"{r['subject']} ||| {r['predicate']} ||| {r['object']}")
    return texts

def build_faiss_index(embs,texts):
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product (cosine if normalized)
    # normalize for cosine-sim
    faiss.normalize_L2(embs)
    index.add(embs)
    index_to_triple = texts  # list aligned with embeddings order

    return index,index_to_triple

def retrieve_top_k(query,embedder,index,index_to_triple,k=10):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    idxs = I[0].tolist()
    hits = [ index_to_triple[i] for i in idxs ]
    return hits, idxs

def build_subgraph_from_hits(G, df, hits):
    # Map hit triple texts back to df rows and expand neighborhood
    rows = []
    for t in hits:
        subj, pred, obj = [x.strip() for x in t.split("|||")]
        rows.append((subj, pred, obj))
    # Collect nodes to include: subjects, objects, and their 1-hop neighbors
    nodes = set()
    for s,p,o in rows:
        nodes.add(s); nodes.add(o)
    # add 1-hop
    for n in list(nodes):
        if n in G:
            nodes.update([nbr for nbr in G.predecessors(n)])
            nodes.update([nbr for nbr in G.successors(n)])
    SG = G.subgraph(nodes).copy()
    return SG, rows



def triples_to_context(triples):
    lines = []
    for s,p,o in triples:
        lines.append(f"- ({s}) —[{p}]→ ({o})")
    return "\n".join(lines)