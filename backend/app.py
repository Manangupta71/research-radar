from flask import Flask, request, jsonify
from flask_cors import CORS
import arxiv
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

model = SentenceTransformer('all-MiniLM-L6-v2')

def fetch_papers(topic, max_results=40):
    client = arxiv.Client()
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "abstract": result.summary.replace('\n', ' '),
            "url": result.entry_id,
            "authors": [a.name for a in result.authors[:3]],
            "published": str(result.published.date())
        })
    return papers

def cluster_papers(papers, n_clusters=5):
    abstracts = [p["abstract"] for p in papers]
    embeddings = model.encode(abstracts, show_progress_bar=False)

    n_clusters = min(n_clusters, len(papers))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    return labels, embeddings

def extract_cluster_labels(papers, labels, n_clusters):
    clusters = {}
    for i in range(n_clusters):
        cluster_papers = [papers[j] for j in range(len(papers)) if labels[j] == i]
        abstracts = [p["abstract"] for p in cluster_papers]

        tfidf = TfidfVectorizer(stop_words='english', max_features=5, ngram_range=(1,2))
        try:
            tfidf.fit(abstracts)
            keywords = tfidf.get_feature_names_out().tolist()
        except:
            keywords = ["general topic"]

        clusters[i] = {
            "keywords": keywords,
            "papers": cluster_papers,
            "size": len(cluster_papers)
        }
    return clusters

def detect_gaps(clusters):
    sizes = [clusters[i]["size"] for i in clusters]
    mean_size = np.mean(sizes)
    gaps = []

    for i, (cid, data) in enumerate(clusters.items()):
        if data["size"] < mean_size * 0.75:
            gaps.append({
                "cluster_id": cid,
                "keywords": data["keywords"],
                "paper_count": data["size"],
                "gap_score": round(1 - (data["size"] / mean_size), 2),
                "sample_papers": data["papers"][:2]
            })

    gaps.sort(key=lambda x: x["gap_score"], reverse=True)
    return gaps

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    topic = data.get('topic', '').strip()
    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    try:
        papers = fetch_papers(topic, max_results=40)
        if len(papers) < 5:
            return jsonify({"error": "Not enough papers found. Try a broader topic."}), 400

        n_clusters = min(6, len(papers) // 4)
        labels, embeddings = cluster_papers(papers, n_clusters=n_clusters)
        clusters = extract_cluster_labels(papers, labels, n_clusters)
        gaps = detect_gaps(clusters)

        cluster_summary = [
            {
                "id": cid,
                "keywords": data["keywords"],
                "size": data["size"],
                "papers": data["papers"]
            }
            for cid, data in clusters.items()
        ]

        return jsonify({
            "topic": topic,
            "total_papers": len(papers),
            "clusters": cluster_summary,
            "gaps": gaps
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)