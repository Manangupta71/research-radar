"# ResearchRadar" 
## Motivation

Keyword search on arXiv tells you what exists, it doesn't tell you what's missing. As someone working across research communities, I noticed researchers often struggle to identify *where the white spaces are* in a field before starting new work. ResearchRadar is an attempt to make that gap visible, automatically using semantic embeddings rather than keyword frequency, so proximity in meaning is what drives the analysis, not surface-level term matching.

---

## How It Works
```
User query
    ↓
Fetch 40 recent arXiv abstracts (arXiv Python SDK)
    ↓
Encode abstracts → 384-dim dense vectors
using sentence-transformers (all-MiniLM-L6-v2)
    ↓
Cluster vectors with KMeans (k = min(6, n//4))
initialized with k-means++ for stable centroids
    ↓
Label each cluster via TF-IDF (bigrams, top-5 terms)
to extract human-readable theme keywords
    ↓
Gap scoring: clusters with size < 0.75× mean cluster size
flagged as under-explored, ranked by gap score = 1 - (size/mean)
    ↓
Flask REST API returns ranked gaps + full cluster breakdown
consumed by a vanilla JS frontend with no build step
```

The core insight: if sentence embeddings place papers in the same region of vector space, they share semantic intent — not just keywords. Sparse regions in that space represent ideas the community hasn't fully converged on yet.

---
