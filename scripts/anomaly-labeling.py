# anomaly-labeling.py
# Enhanced hybrid semantic + statistical anomaly labeling workflow
# Keeps original CLI workflow and adds: embeddings, TF-IDF, fuzzy + WordNet expansion,
# contextual scoring, feedback learning, ML classifier, calibrated confidence, and transfer learning.

import os
import re
import json
import warnings
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Visualization (optional / used in analysis helpers)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from rapidfuzz import fuzz, process
import nltk
from nltk.corpus import wordnet

warnings.filterwarnings('ignore')

# =========================
# Configuration
# =========================
PROJECT_ROOT = Path(r"C:\Computer Science\AIMLDL\log-anomaly-detection")
DATA_PATH = PROJECT_ROOT / "dataset" / "structured_data"
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "labeled_data"
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

# If you run on Linux/WSL, you can set via env instead:
# PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
# DATA_PATH = PROJECT_ROOT / "dataset" / "structured_data"
# OUTPUT_PATH = PROJECT_ROOT / "dataset" / "labeled_data"
# OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

LOG_SOURCES = [
    'Android_2k', 'Apache_2k', 'BGL_2k', 'Hadoop_2k', 'HDFS_2k',
    'HealthApp_2k', 'HPC_2k', 'Linux_2k', 'Mac_2k', 'OpenSSH_2k',
    'OpenStack_2k', 'Proxifier_2k', 'Spark_2k', 'Thunderbird_2k',
    'Windows_2k', 'Zookeeper_2k'
]

LABELS = {
    0: "normal",
    1: "security_anomaly",
    2: "system_failure",
    3: "performance_issue",
    4: "network_anomaly",
    5: "config_error",
    6: "hardware_issue",
    7: "unknown_anomaly"
}

# --- Raw seed patterns (will be expanded via WordNet & normalized) ---
RAW_PATTERNS = {
    'security': ['authentication failure', 'invalid user', 'break-in attempt',
                 'failed password', 'unauthorized', 'access denied', 'login failed',
                 'permission denied', 'security violation', 'intrusion'],
    'system': ['error', 'critical', 'fatal', 'exception', 'crash', 'abort',
               'segmentation fault', 'core dump', 'kernel panic', 'died'],
    'performance': ['timeout', 'slow', 'overload', 'resource exhausted',
                    'quota exceeded', 'memory pressure', 'cpu spike', 'bottleneck',
                    'high latency', 'response time'],
    'network': ['connection refused', 'host unreachable', 'network unreachable',
                'connection timeout', 'socket error', 'dns error', 'connection lost',
                'network down', 'packet loss'],
    'config': ['configuration error', 'config invalid', 'parameter error',
               'setting invalid', 'option unknown', 'syntax error', 'parse error',
               'invalid configuration', 'config mismatch'],
    'hardware': ['hardware error', 'disk error', 'i/o error', 'device error',
                 'sensor error', 'temperature', 'voltage', 'power failure',
                 'component failure', 'device timeout']
}

# Map category -> label id
CATEGORY_TO_LABEL = {
    'security': 1,
    'system': 2,
    'performance': 3,
    'network': 4,
    'config': 5,
    'hardware': 6
}

# =========================
# Text utilities
# =========================
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # replace numbers and IPs to reduce sparsity
    text = re.sub(r'\d+', '<num>', text)
    text = re.sub(r'\b(ip|addr|address)\b.*?\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<ip>', text)
    # normalize punctuation/underscores
    text = re.sub(r'[\W_]+', ' ', text)
    return text.strip()

def expand_keywords(keywords):
    expanded = set()
    for kw in keywords:
        expanded.add(kw)
        try:
            for syn in wordnet.synsets(kw.replace(' ', '_')):
                for lemma in syn.lemmas():
                    expanded.add(lemma.name().replace('_', ' '))
        except Exception:
            pass
    return list(expanded)

def fuzzy_match(text, keywords, threshold=80):
    # keywords expected to be normalized already
    for kw in keywords:
        if fuzz.partial_ratio(text, kw) >= threshold:
            return True
    return False

def build_expanded_patterns(raw_patterns):
    """Expand, normalize, and dedupe patterns using WordNet."""
    expanded = {}
    for cat, kws in raw_patterns.items():
        base = set()
        for k in kws:
            base.add(normalize_text(k))
        # expand
        big = set()
        for k in base:
            big.update(normalize_text(x) for x in expand_keywords([k]))
        # keep originals too
        big.update(base)
        expanded[cat] = sorted(big)
    return expanded

# Build once on import
PATTERNS = build_expanded_patterns(RAW_PATTERNS)

# =========================
# Data I/O
# =========================
def load_all_datasets():
    datasets = {}
    failed = []

    print("Loading datasets...")
    for source in LOG_SOURCES:
        try:
            file_path = DATA_PATH / f"{source}.log_structured.csv"
            df = pd.read_csv(file_path)
            datasets[source] = df
            print(f"✓ {source}: {len(df):,} logs, {df.shape[1]} columns")
        except Exception as e:
            print(f"✗ {source}: {e}")
            failed.append(source)

    total = sum(len(df) for df in datasets.values())
    print(f"\nLoaded: {len(datasets)}/{len(LOG_SOURCES)} sources")
    print(f"Total logs: {total:,}")
    if failed:
        print(f"Failed to load: {failed}")

    return datasets, failed

# =========================
# Semantic / ML backends
# =========================
# SentenceTransformer is heavy; import lazily to avoid cost when not needed
_EMB_MODEL = None

def get_embedding_model(model_name='all-MiniLM-L6-v2'):
    global _EMB_MODEL
    if _EMB_MODEL is None:
        from sentence_transformers import SentenceTransformer
        try:
            _EMB_MODEL = SentenceTransformer(model_name, device='cuda')
        except Exception:
            # fallback to CPU
            _EMB_MODEL = SentenceTransformer(model_name)
    return _EMB_MODEL

def cosine_sim_max(query_emb, corpus_embs):
    # corpus_embs: list of torch tensors
    if not corpus_embs:
        return 0.0
    from sentence_transformers import util
    sims = [util.cos_sim(query_emb, emb).item() for emb in corpus_embs]
    return float(np.max(sims)) if len(sims) else 0.0

class MLLabeler:
    """Lightweight TF-IDF + LogisticRegression classifier."""
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.is_trained = False

    def train(self, texts, labels):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=2, max_features=100000)
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(max_iter=200, n_jobs=None, class_weight='balanced')
        self.model.fit(X, labels)
        self.is_trained = True

    def predict_proba(self, texts):
        if not self.is_trained:
            return None
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def predict(self, texts):
        if not self.is_trained:
            return None
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

# =========================
# Analysis utilities
# =========================
def analyze_datasets(datasets):
    stats = {}

    print("Analyzing datasets with expanded patterns & fuzzy matching...")
    for source, df in datasets.items():
        if 'EventTemplate' not in df.columns:
            print(f"Warning: {source} missing EventTemplate column")
            continue

        templates = df['EventTemplate'].value_counts()

        anomaly_count = 0
        if 'Content' in df.columns:
            content_lower = df['Content'].astype(str).apply(normalize_text)
            # fuzzy match across categories
            for category, keywords in PATTERNS.items():
                # vectorized approx: quick substring filter then fuzzy refine (lightweight)
                # For speed, we use simple contains for prefilter
                pre = content_lower.str.contains('|'.join(map(re.escape, keywords)), na=False)
                # add sum; fuzzy pass can be expensive, so we keep it simple here
                anomaly_count += pre.sum()

        stats[source] = {
            'logs': len(df),
            'templates': len(templates),
            'efficiency': len(df) / max(len(templates), 1),
            'anomaly_rate': (anomaly_count / max(len(df),1)) * 100,
            'top_templates': templates.head(3).to_dict()
        }

    return stats

def rank_sources_by_priority(stats, completed_sources=None):
    if completed_sources is None:
        completed_sources = []

    rankings = []
    for source, data in stats.items():
        if source in completed_sources:
            continue
        anomaly_score = data['anomaly_rate']
        template_score = max(0, 100 - (data['templates'] / 20))
        efficiency_score = min(100, data['efficiency'] / 10)
        priority = (anomaly_score * 0.4 + template_score * 0.3 + efficiency_score * 0.3)
        rankings.append({
            'source': source,
            'priority': priority,
            'anomaly_rate': data['anomaly_rate'],
            'templates': data['templates'],
            'efficiency': data['efficiency']
        })

    return sorted(rankings, key=lambda x: x['priority'], reverse=True)

# =========================
# SmartPatternLibrary 2.0
# =========================
class SmartPatternLibrary:
    """
    Hybrid label suggester:
    - Keyword/phrase word-scores + TF-IDF idf weighting (1-3 grams)
    - Feedback reweighting for corrections
    - Semantic embeddings similarity per-label corpus
    - Optional ML classifier trained on accumulated labeled exports
    """
    def __init__(self, save_path=None):
        self.save_path = save_path or (OUTPUT_PATH / "smart_patterns.json")
        self.label_patterns = defaultdict(lambda: {
            'keywords': defaultdict(int),
            'templates': [],
            'sources': set(),
            'total_logs': 0
        })
        self.word_scores = defaultdict(lambda: defaultdict(float))
        self.positive_counts = defaultdict(lambda: defaultdict(int))
        self.negative_counts = defaultdict(lambda: defaultdict(int))

        # TF-IDF
        self.idf_scores = {}
        self.tfidf_vectorizer = None

        # Embeddings store
        self.template_embeddings = defaultdict(list)  # label -> [tensor, ...]
        self._emb_model_name = 'all-MiniLM-L6-v2'

        # ML Classifier
        self.ml = MLLabeler()

        # Confidence calibration cache
        self.calibration_stats = {'high': [], 'medium': [], 'low': []}

    # ---------- Internal helpers ----------
    def _update_tfidf(self):
        all_templates = []
        for label, data in self.label_patterns.items():
            all_templates += data['templates']
        if len(all_templates) < 3:
            return
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=1)
        self.tfidf_vectorizer.fit(all_templates)
        # idf dictionary for quick lookup
        self.idf_scores = dict(zip(self.tfidf_vectorizer.get_feature_names_out(),
                                   self.tfidf_vectorizer.idf_))

    def _template_to_embedding(self, text):
        model = get_embedding_model(self._emb_model_name)
        from torch import Tensor
        emb = model.encode(normalize_text(text), convert_to_tensor=True)
        return emb

    # ---------- Public API ----------
    def add_source_data(self, labeling_data, source_name, alpha=1.0, beta=0.2):
        """Learn from labeled templates and build embeddings/TF-IDF."""
        print(f"Learning patterns from {source_name}...")
        count_added = 0

        for item in labeling_data:
            if item.get('label') is None:
                continue

            label = int(item['label'])
            template = normalize_text(item['template'])
            samples = [normalize_text(s) for s in item.get('samples', [])]
            full_text = ' '.join([template] + samples)
            log_count = int(item.get('count', 1))

            self.label_patterns[label]['sources'].add(source_name)
            self.label_patterns[label]['templates'].append(template)
            self.label_patterns[label]['total_logs'] += log_count

            # update word scores with feedback weighting
            words = set(re.findall(r'\b[a-zA-Z]{3,}\b', full_text))
            common_words = {'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'was', 'not'}
            words = words - common_words
            for w in words:
                self.label_patterns[label]['keywords'][w] += log_count
                self.word_scores[w][label] += alpha * log_count
                for other_label in list(self.word_scores[w].keys()):
                    if other_label != label:
                        self.word_scores[w][other_label] *= (1 - beta)

            # add embedding
            try:
                emb = self._template_to_embedding(template)
                self.template_embeddings[label].append(emb)
            except Exception as e:
                pass

            count_added += 1

        self._update_tfidf()
        self.save_library()
        print(f"Updated pattern library with {count_added} labeled templates from {source_name}")

    def suggest_label(self, template, samples):
        """Ensemble scoring: keywords/TF-IDF + contextual + embeddings + ML classifier."""
        template_n = normalize_text(template)
        samples_n = [normalize_text(s) for s in samples]
        full_text = ' '.join([template_n] + samples_n)

        # 1) Keyword/word_scores with IDF weighting
        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', full_text))
        label_scores = defaultdict(float)
        for w in words:
            if w in self.word_scores:
                total_w = sum(self.word_scores[w].values())
                if total_w <= 0:
                    continue
                idf = self.idf_scores.get(w, 1.0)
                for label, score in self.word_scores[w].items():
                    label_scores[label] += (score / total_w) * idf

        # 2) Contextual boost: matches in both template & content
        for w in words:
            if w in template_n:
                for label in list(label_scores.keys()):
                    label_scores[label] *= 1.1  # +10% boost

        # 3) Embedding similarity (fallback or additively mixed)
        # If weak keyword evidence, rely on embeddings; else blend
        use_embeddings = (not label_scores) or (max(label_scores.values()) < 1.0)
        if use_embeddings:
            try:
                q_emb = self._template_to_embedding(template_n)
                emb_sims = {}
                for label, embs in self.template_embeddings.items():
                    if embs:
                        sim = cosine_sim_max(q_emb, embs)
                        if sim > 0:
                            emb_sims[label] = sim
                # blend or fallback
                if emb_sims:
                    if not label_scores:
                        label_scores = defaultdict(float, emb_sims)
                    else:
                        # blend with small weight
                        for k, v in emb_sims.items():
                            label_scores[k] += 0.75 * v
            except Exception:
                pass

        # 4) ML classifier probability (if trained)
        if self.ml.is_trained:
            try:
                proba = self.ml.predict_proba([full_text])
                if proba is not None:
                    # ensure mapping to labels 0..7
                    # Handle missing classes in training by aligning columns
                    classes = list(self.ml.model.classes_)
                    # fill label_scores with calibrated probs
                    for idx, c in enumerate(classes):
                        label_scores[int(c)] += float(proba[0, idx]) * 1.25  # weighted blend
            except Exception:
                pass

        if not label_scores:
            # No evidence: default to normal
            return 0, 'low'

        # pick best label
        best_label = max(label_scores, key=label_scores.get)
        best_score = label_scores[best_label]

        # Confidence calibration using relative margin
        scores_sorted = sorted(label_scores.values(), reverse=True)
        margin = (scores_sorted[0] - scores_sorted[1]) if len(scores_sorted) > 1 else scores_sorted[0]
        # empirical thresholds:
        if best_score >= 2.0 and margin >= 0.5:
            conf = 'high'
        elif best_score >= 1.0 and margin >= 0.2:
            conf = 'medium'
        else:
            conf = 'low'

        return int(best_label), conf

    def register_feedback(self, template, samples, chosen_label, alpha=1.0, beta=0.2):
        """When user corrects a label, reweight word_scores."""
        template_n = normalize_text(template)
        samples_n = [normalize_text(s) for s in samples]
        full_text = ' '.join([template_n] + samples_n)
        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', full_text))
        for w in words:
            self.positive_counts[w][chosen_label] += 1
            self.word_scores[w][chosen_label] += alpha
            for other in self.word_scores[w]:
                if other != chosen_label:
                    self.negative_counts[w][other] += 1
                    self.word_scores[w][other] *= (1 - beta)

    def save_library(self):
        data = {
            'label_patterns': {},
            'word_scores': {},
            'positive_counts': {},
            'negative_counts': {},
            'emb_model': self._emb_model_name
        }
        for label, patterns in self.label_patterns.items():
            data['label_patterns'][str(label)] = {
                'keywords': dict(patterns['keywords']),
                'templates': patterns['templates'],
                'sources': list(patterns['sources']),
                'total_logs': patterns['total_logs']
            }
        for word, scores in self.word_scores.items():
            data['word_scores'][word] = {str(int(k)): float(v) for k, v in scores.items()}

        # counts
        data['positive_counts'] = {w: {str(int(k)): int(v) for k, v in d.items()} for w, d in self.positive_counts.items()}
        data['negative_counts'] = {w: {str(int(k)): int(v) for k, v in d.items()} for w, d in self.negative_counts.items()}

        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_library(self):
        if not self.save_path.exists():
            return False
        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)

            self._emb_model_name = data.get('emb_model', 'all-MiniLM-L6-v2')

            for label_str, patterns in data.get('label_patterns', {}).items():
                label = int(label_str)
                self.label_patterns[label]['keywords'] = defaultdict(int, patterns['keywords'])
                self.label_patterns[label]['templates'] = patterns['templates']
                self.label_patterns[label]['sources'] = set(patterns.get('sources', []))
                self.label_patterns[label]['total_logs'] = patterns.get('total_logs', 0)

            # Restore word scores
            self.word_scores.clear()
            for word, scores in data.get('word_scores', {}).items():
                self.word_scores[word] = defaultdict(float, {int(k): float(v) for k, v in scores.items()})

            # Restore counts
            self.positive_counts.clear()
            for w, d in data.get('positive_counts', {}).items():
                self.positive_counts[w] = defaultdict(int, {int(k): int(v) for k, v in d.items()})

            self.negative_counts.clear()
            for w, d in data.get('negative_counts', {}).items():
                self.negative_counts[w] = defaultdict(int, {int(k): int(v) for k, v in d.items()})

            # Rebuild embeddings from templates
            for label, pat in self.label_patterns.items():
                self.template_embeddings[label] = []
                for t in pat['templates']:
                    try:
                        emb = self._template_to_embedding(t)
                        self.template_embeddings[label].append(emb)
                    except Exception:
                        pass

            self._update_tfidf()
            print("Smart pattern library loaded.")
            return True
        except Exception as e:
            print(f"Error loading pattern library: {e}")
            return False

    def get_statistics(self):
        stats = {}
        for label in LABELS.keys():
            if self.label_patterns[label]['templates']:
                sources = self.label_patterns[label]['sources']
                stats[label] = {
                    'label_name': LABELS[label],
                    'templates': len(self.label_patterns[label]['templates']),
                    'keywords': len(self.label_patterns[label]['keywords']),
                    'sources': len(sources) if isinstance(sources, (list, set)) else 0,
                    'total_logs': self.label_patterns[label]['total_logs']
                }
        return stats

    # ---- ML training on combined labeled CSVs ----
    def train_ml_from_exports(self):
        """Train lightweight classifier from all *_labeled.csv in OUTPUT_PATH."""
        csvs = list(OUTPUT_PATH.glob("*_labeled.csv"))
        if not csvs:
            print("No labeled exports found for ML training.")
            return False

        texts, labels = [], []
        for fp in csvs:
            try:
                df = pd.read_csv(fp)
                df = df[df['AnomalyLabel'] >= 0].copy()
                if df.empty:
                    continue
                # combine template + content
                t = df['EventTemplate'].astype(str).apply(normalize_text)
                c = df['Content'].astype(str).apply(normalize_text)
                combo = (t + " " + c).tolist()
                texts.extend(combo)
                labels.extend(df['AnomalyLabel'].astype(int).tolist())
            except Exception as e:
                print(f"Skipping {fp.name}: {e}")

        if len(texts) < 100:
            print("Not enough data to train ML classifier yet.")
            return False

        print(f"Training ML classifier on {len(texts):,} labeled logs...")
        self.ml.train(texts, labels)
        print("ML classifier trained.")
        return True

# =========================
# Labeling pipeline steps
# =========================
def prepare_templates_for_labeling(df, source_name, pattern_library: SmartPatternLibrary):
    if 'EventTemplate' not in df.columns or 'Content' not in df.columns:
        print(f"Error: Missing required columns in {source_name}")
        print(f"Available columns: {list(df.columns)}")
        return []

    templates = df['EventTemplate'].value_counts()
    labeling_data = []

    print(f"Processing {len(templates)} unique templates with semantic suggestions...")

    for template, count in templates.items():
        matching_rows = df[df['EventTemplate'] == template]
        samples = matching_rows['Content'].astype(str).head(3).tolist()
        samples = [s if s else 'Empty content' for s in samples]

        try:
            suggested_label, confidence = pattern_library.suggest_label(template, samples)
        except Exception as e:
            print(f"Warning: Pattern suggestion failed for {template[:50]}: {e}")
            suggested_label, confidence = 0, "low"

        labeling_data.append({
            'template': template,
            'count': int(count),
            'percentage': (count / len(df)) * 100,
            'samples': samples,
            'suggested': int(suggested_label),
            'confidence': confidence,
            'label': None,
            'notes': ''
        })

    return sorted(labeling_data, key=lambda x: x['count'], reverse=True)

def auto_label_high_confidence(labeling_data):
    auto_count = 0
    for item in labeling_data:
        if item['confidence'] == 'high' and item['label'] is None:
            item['label'] = item['suggested']
            item['notes'] = 'Auto-labeled (high confidence)'
            auto_count += 1
    print(f"Auto-labeled {auto_count} templates")
    return auto_count

def interactive_labeling_session(data, source_name, start=0, count=10):
    print(f"\n{'='*60}")
    print(f"LABELING SESSION: {source_name}")
    print("Labels:", ", ".join(f"{k}:{v}" for k, v in LABELS.items()))
    print("Commands: 0-7 (label), 'skip', 'quit', 'save', 'info'")
    print(f"{'='*60}")

    end = min(start + count, len(data))
    labeled = 0

    for i in range(start, end):
        item = data[i]
        print(f"\n[{i+1}/{len(data)}] Template")
        print(f"Frequency: {item['count']:,} logs ({item['percentage']:.1f}%)")
        print(f"Template: {item['template']}")
        print("Sample logs:")
        for j, sample in enumerate(item['samples'][:3], 1):
            display = sample[:120] + "..." if len(sample) > 120 else sample
            print(f"  {j}. {display}")
        print(f"Suggested: {item['suggested']} ({LABELS[item['suggested']]}) - {item['confidence']}")

        while True:
            response = input(f"\nLabel (suggested {item['suggested']}): ").strip()
            if response.lower() == 'quit':
                return i, labeled
            elif response.lower() == 'skip':
                break
            elif response.lower() == 'save':
                save_labeling_progress(data, source_name)
                continue
            elif response.lower() == 'info':
                print(f"\nAdditional info:")
                print(f"Template pattern: {item['template']}")
                df = datasets.get(source_name)
                if df is not None:
                    more_samples = df[df['EventTemplate'] == item['template']]['Content'].astype(str).head(5).tolist()
                    for k, sample in enumerate(more_samples, 1):
                        print(f"  Extra {k}: {str(sample)[:100]}...")
                continue
            elif response.isdigit() and 0 <= int(response) < len(LABELS):
                item['label'] = int(response)
                notes = input("Optional notes: ").strip()
                if notes:
                    item['notes'] = notes
                labeled += 1
                # feedback into library
                pattern_library.register_feedback(item['template'], item['samples'], item['label'])
                break
            else:
                print(f"Enter a number 0-{len(LABELS)-1}, 'skip', 'save', 'info', or 'quit'")

    print(f"\nLabeled {labeled} templates in this session")
    return end, labeled

def bulk_label_by_suggestion(labeling_data):
    groups = defaultdict(list)
    for item in labeling_data:
        if item['label'] is None:
            groups[item['suggested']].append(item)

    total_labeled = 0
    print("Bulk labeling by pattern suggestions:")
    for suggested_label, items in groups.items():
        if len(items) == 0:
            continue
        print(f"\n{LABELS[suggested_label]}: {len(items)} templates")
        for i, item in enumerate(items[:3], 1):
            print(f"  {i}. [{item['count']:4,}] {item['template'][:60]}...")
        if len(items) > 3:
            print(f"  ... and {len(items)-3} more")
        choice = input(f"Label all as {LABELS[suggested_label]}? (y/n/s=skip): ").strip().lower()
        if choice == 'y':
            for item in items:
                item['label'] = suggested_label
                item['notes'] = 'Bulk labeled by suggestion'
            total_labeled += len(items)
            print(f"Labeled {len(items)} templates")
    print(f"\nBulk labeled {total_labeled} templates total")
    return total_labeled

def save_labeling_progress(data, source_name):
    save_data = []
    for item in data:
        item_copy = dict(item)
        if 'samples' in item_copy and not isinstance(item_copy['samples'], str):
            item_copy['samples'] = json.dumps(item_copy['samples'])
        save_data.append(item_copy)
    df = pd.DataFrame(save_data)
    progress_file = OUTPUT_PATH / f"{source_name}_progress.csv"
    df.to_csv(progress_file, index=False)
    labeled = sum(1 for item in data if item['label'] is not None)
    print(f"Progress saved: {labeled}/{len(data)} templates for {source_name}")

def load_labeling_progress(source_name):
    progress_file = OUTPUT_PATH / f"{source_name}_progress.csv"
    if not progress_file.exists():
        return None
    try:
        df = pd.read_csv(progress_file).where(pd.notna, None)
        data = df.to_dict('records')
        for item in data:
            if 'samples' in item and isinstance(item['samples'], str):
                try:
                    item['samples'] = json.loads(item['samples'])
                except:
                    item['samples'] = ['Error loading samples']
        labeled = sum(1 for item in data if item.get('label') is not None)
        print(f"Loaded progress: {labeled}/{len(data)} templates for {source_name}")
        return data
    except Exception as e:
        print(f"Error loading progress for {source_name}: {e}")
        return None

def show_labeling_progress(data, source_name=None):
    total = len(data)
    labeled = sum(1 for item in data if item['label'] is not None)
    if source_name:
        print(f"\nProgress for {source_name}:")
    else:
        print(f"\nLabeling Progress:")
    print(f"Templates: {labeled}/{total} ({labeled/total*100:.1f}%)")

    if labeled > 0:
        total_logs = sum(item['count'] for item in data)
        labeled_logs = sum(item['count'] for item in data if item['label'] is not None)
        print(f"Log coverage: {labeled_logs:,}/{total_logs:,} ({labeled_logs/total_logs*100:.1f}%)")
        dist = defaultdict(int)
        for item in data:
            if item['label'] is not None:
                dist[int(item['label'])] += item['count']
        print("Label distribution:")
        for label in sorted(dist.keys()):
            count = dist[label]
            print(f"  {label} ({LABELS[label]}): {count:,} logs")
    return labeled, total

def export_labeled_dataset(df, labeling_data, source_name):
    template_labels = {item['template']: item['label']
                       for item in labeling_data if item['label'] is not None}
    result_df = df.copy()
    result_df['AnomalyLabel'] = result_df['EventTemplate'].map(template_labels).fillna(-1).astype(int)
    result_df['AnomalyLabelName'] = result_df['AnomalyLabel'].map(lambda x: LABELS.get(x, 'unlabeled'))
    result_df['Source'] = source_name

    output_file = OUTPUT_PATH / f"{source_name}_labeled.csv"
    result_df.to_csv(output_file, index=False)

    total = len(result_df)
    labeled_count = (result_df['AnomalyLabel'] >= 0).sum()
    anomaly_count = (result_df['AnomalyLabel'] > 0).sum()

    print(f"\nExported labeled dataset: {output_file}")
    print(f"Total logs: {total:,}")
    print(f"Labeled logs: {labeled_count:,} ({labeled_count/total*100:.1f}%)")
    if labeled_count > 0:
        print(f"Anomaly logs: {anomaly_count:,} ({anomaly_count/labeled_count*100:.1f}% of labeled)")

    return result_df

# =========================
# QA and Transfer Learning
# =========================
def validate_labeling_quality(labeling_data):
    issues = []
    for i, item in enumerate(labeling_data):
        if item.get('label') is None:
            continue
        content = ' '.join(str(s) for s in item['samples']).lower()
        label = int(item['label'])
        if label == 1:
            security_words = ['auth', 'login', 'password', 'user', 'invalid', 'fail', 'denied', 'unauthorized']
            if not any(w in content for w in security_words):
                issues.append(f"Template {i}: Security label without security keywords")
        elif label == 0:
            error_words = ['error', 'fail', 'critical', 'exception', 'crash', 'fatal']
            if any(w in content for w in error_words):
                issues.append(f"Template {i}: Normal label with error keywords")
        # heuristic: high-frequency anomalies deserve review
        if label > 0 and item['percentage'] > 15:
            issues.append(f"Template {i}: High-frequency anomaly ({item['percentage']:.1f}%)")
    if issues:
        print(f"\nFound {len(issues)} potential issues:")
        for issue in issues[:10]:
            print(f"  {issue}")
    else:
        print("\nNo validation issues found - labeling quality looks good!")
    return issues

def analyze_cross_source_patterns(completed_sources_data):
    if len(completed_sources_data) < 2:
        print("Need at least 2 completed sources for cross-analysis")
        return
    print("\nCROSS-SOURCE PATTERN ANALYSIS")
    print("="*50)

    source_distributions = {}
    for source_name, data in completed_sources_data.items():
        dist = defaultdict(int)
        total_logs = 0
        for item in data:
            if item['label'] is not None:
                dist[int(item['label'])] += int(item['count'])
                total_logs += int(item['count'])
        if total_logs > 0:
            source_distributions[source_name] = {label: (count/total_logs)*100 for label, count in dist.items()}

    print(f"{'Source':<15}", end="")
    for label_id in sorted(LABELS.keys()):
        print(f"{LABELS[label_id][:8]:<10}", end="")
    print()
    print("-" * (15 + 10 * len(LABELS)))
    for source, dist in source_distributions.items():
        print(f"{source:<15}", end="")
        for label_id in sorted(LABELS.keys()):
            pct = dist.get(label_id, 0)
            print(f"{pct:>8.1f}% ", end="")
        print()

def transfer_labels_across_sources(source_mgr, threshold=90):
    """Propagate labels between similar templates via fuzzy matching."""
    print("\nTemplate transfer learning across sources...")
    # Build map: template -> label from completed
    known = {}
    for src in source_mgr.completed:
        progress = load_labeling_progress(src)
        if progress:
            for item in progress:
                if item.get('label') is not None:
                    known[item['template']] = int(item['label'])

    # For current source, fill suggestions if fuzzy match > threshold
    if source_mgr.current_data:
        templates = [it['template'] for it in source_mgr.current_data if it['label'] is None]
        for t in templates:
            match, score, _ = process.extractOne(t, list(known.keys()))
            if score >= threshold:
                inferred = known.get(match)
                for it in source_mgr.current_data:
                    if it['template'] == t and it['label'] is None:
                        it['suggested'] = inferred
                        if it['confidence'] == 'low':
                            it['confidence'] = 'medium'
        print("Transfer suggestions updated where high-similarity templates were found.")

# =========================
# Dataset synthesis & ML export
# =========================
def create_combined_dataset(completed_sources):
    if not completed_sources:
        print("No completed sources found")
        return None

    print(f"Creating combined dataset from {len(completed_sources)} sources...")
    combined_dfs = []
    for source in completed_sources:
        labeled_file = OUTPUT_PATH / f"{source}_labeled.csv"
        if labeled_file.exists():
            df = pd.read_csv(labeled_file)
            combined_dfs.append(df)
            print(f"Added {source}: {len(df):,} logs")
    if not combined_dfs:
        print("No labeled datasets found")
        return None

    combined = pd.concat(combined_dfs, ignore_index=True)
    combined_file = OUTPUT_PATH / "combined_labeled_dataset.csv"
    combined.to_csv(combined_file, index=False)

    total = len(combined)
    labeled = (combined['AnomalyLabel'] >= 0).sum()
    anomalies = (combined['AnomalyLabel'] > 0).sum()

    print(f"\nCombined dataset saved: {combined_file}")
    print(f"Total logs: {total:,}")
    print(f"Labeled logs: {labeled:,} ({labeled/total*100:.1f}%)")
    if labeled > 0:
        print(f"Anomaly logs: {anomalies:,} ({anomalies/labeled*100:.1f}% of labeled)")

    print(f"\nPer-source breakdown:")
    for source in combined['Source'].unique():
        source_data = combined[combined['Source'] == source]
        s_total = len(source_data)
        s_labeled = (source_data['AnomalyLabel'] >= 0).sum()
        s_anomalies = (source_data['AnomalyLabel'] > 0).sum()
        print(f"  {source}: {s_total:,} logs, {s_labeled:,} labeled, {s_anomalies:,} anomalies")
    return combined

def export_ml_ready_data(dataset, output_name="combined"):
    labeled_data = dataset[dataset['AnomalyLabel'] >= 0].copy()
    if len(labeled_data) == 0:
        print("No labeled data to export for ML")
        return
    print(f"Preparing ML data from {len(labeled_data):,} labeled logs...")

    labeled_data['Content'] = labeled_data['Content'].astype(str)
    labeled_data['EventTemplate'] = labeled_data['EventTemplate'].astype(str)

    labeled_data['ContentLength'] = labeled_data['Content'].str.len()
    labeled_data['TemplateLength'] = labeled_data['EventTemplate'].str.len()
    labeled_data['HasError'] = labeled_data['Content'].str.lower().str.contains('error|fail|critical|exception')
    labeled_data['HasAuth'] = labeled_data['Content'].str.lower().str.contains('auth|login|user|password')
    labeled_data['HasNetwork'] = labeled_data['Content'].str.lower().str.contains('connection|network|timeout')
    labeled_data['HasSystem'] = labeled_data['Content'].str.lower().str.contains('system|kernel|process')
    labeled_data['HasNumbers'] = labeled_data['Content'].str.contains(r'\d+')
    labeled_data['HasIPAddress'] = labeled_data['Content'].str.contains(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')

    feature_cols = ['Content', 'EventTemplate', 'ContentLength', 'TemplateLength',
                    'HasError', 'HasAuth', 'HasNetwork', 'HasSystem', 'HasNumbers',
                    'HasIPAddress', 'Source']

    X = labeled_data[feature_cols]
    y = labeled_data['AnomalyLabel']

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    train_file = OUTPUT_PATH / f"{output_name}_train.csv"
    test_file = OUTPUT_PATH / f"{output_name}_test.csv"

    pd.concat([X_train, y_train], axis=1).to_csv(train_file, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_file, index=False)

    print(f"\nML data exported:")
    print(f"Training set: {len(X_train):,} samples -> {train_file}")
    print(f"Test set:    {len(X_test):,} samples -> {test_file}")

    print(f"\nTraining set label distribution:")
    for label, count in y_train.value_counts().sort_index().items():
        pct = count/len(y_train)*100
        print(f"  {label} ({LABELS[label]}): {count:,} ({pct:.1f}%)")

    return train_file, test_file

# =========================
# Source Manager (unchanged but enhanced calls)
# =========================
class SourceManager:
    def __init__(self, datasets, stats, pattern_library: SmartPatternLibrary):
        self.datasets = datasets
        self.stats = stats
        self.pattern_library = pattern_library
        self.completed = []
        self.current_data = None
        self.current_source = None
        self.load_completed_sources()

    def load_completed_sources(self):
        completed_file = OUTPUT_PATH / "completed_sources.json"
        if completed_file.exists():
            try:
                with open(completed_file, 'r') as f:
                    data = json.load(f)
                    self.completed = data.get('completed', [])
                    print(f"Loaded {len(self.completed)} completed sources from disk")
            except:
                pass

    def save_completed_sources(self):
        completed_file = OUTPUT_PATH / "completed_sources.json"
        with open(completed_file, 'w') as f:
            json.dump({
                'completed': self.completed,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)

    def get_next_recommended_source(self):
        rankings = rank_sources_by_priority(self.stats, self.completed)
        if not rankings:
            return None
        return rankings[0]['source']

    def start_new_source(self, source):
        print(f"\nStarting source: {source}")
        existing = load_labeling_progress(source)
        if existing:
            self.current_data = existing
            print("Resumed from saved progress")
        else:
            self.current_data = prepare_templates_for_labeling(
                self.datasets[source], source, self.pattern_library)
            auto_label_high_confidence(self.current_data)

        self.current_source = source
        show_labeling_progress(self.current_data, source)

        # transfer learning: propagate labels from completed sources if similar
        transfer_labels_across_sources(self, threshold=90)
        return self.current_data

    def complete_current_source(self):
        if not self.current_source or not self.current_data:
            print("No active source to complete")
            return None

        final_df = export_labeled_dataset(
            self.datasets[self.current_source],
            self.current_data,
            self.current_source
        )

        # Feed back to pattern library
        self.pattern_library.add_source_data(self.current_data, self.current_source)

        if self.current_source not in self.completed:
            self.completed.append(self.current_source)
            self.save_completed_sources()

        print(f"✓ Completed {self.current_source}")
        print(f"Total completed sources: {len(self.completed)}/{len(LOG_SOURCES)}")

        completed_source = self.current_source
        self.current_source = None
        self.current_data = None

        # Try (re)training ML with new data
        self.pattern_library.train_ml_from_exports()

        return final_df, completed_source

    def get_overall_status(self):
        print(f"\nOVERALL STATUS")
        print(f"{'='*40}")
        print(f"Total sources: {len(LOG_SOURCES)}")
        print(f"Completed sources: {len(self.completed)}")
        print(f"Remaining sources: {len(LOG_SOURCES) - len(self.completed)}")
        print(f"Progress: {len(self.completed)/len(LOG_SOURCES)*100:.1f}%")

        if self.completed:
            print(f"\nCompleted: {', '.join(self.completed)}")

        try:
            lib_stats = self.pattern_library.get_statistics()
            if lib_stats:
                total_templates = sum(s.get('templates', 0) for s in lib_stats.values())
                all_sources = set()
                for label_data in lib_stats.values():
                    # label_data['sources'] is int count in get_statistics, not a set
                    pass
                print(f"\nPattern library: {total_templates} templates (embeddings + TF-IDF ready)")
            else:
                print(f"\nPattern library: Empty (no patterns learned yet)")
        except Exception as e:
            print(f"\nPattern library: Error reading stats - {e}")

        if self.current_source:
            print(f"\nCurrent source: {self.current_source}")
            if self.current_data:
                show_labeling_progress(self.current_data, self.current_source)
        else:
            next_source = self.get_next_recommended_source()
            if next_source:
                print(f"\nNext recommended: {next_source}")

# =========================
# Workflow helpers
# =========================
def streamlined_workflow():
    print("STREAMLINED MULTI-SOURCE LOG ANOMALY LABELING WORKFLOW")
    print("="*70)

    source_mgr.get_overall_status()

    if source_mgr.current_source and source_mgr.current_data:
        print(f"\nActive source: {source_mgr.current_source}")
        labeled, total = show_labeling_progress(source_mgr.current_data, source_mgr.current_source)

        if labeled < total:
            print(f"\nOptions:")
            print(f"1. Continue labeling current source")
            print(f"2. Bulk label remaining templates")
            print(f"3. Complete source with current progress")
            print(f"4. Switch to different source")

            choice = input("Choose option (1-4): ").strip()

            if choice == '1':
                return continue_labeling_current_source()
            elif choice == '2':
                return bulk_label_remaining_templates()
            elif choice == '3':
                return complete_current_source()
            elif choice == '4':
                pass
            else:
                print("Invalid choice")
                return
        else:
            print("\nCurrent source fully labeled!")
            choice = input("Complete this source? (y/n): ").strip().lower()
            if choice == 'y':
                return complete_current_source()

    next_source = source_mgr.get_next_recommended_source()
    if next_source:
        print(f"\nNext recommended source: {next_source}")
        source_stats = source_mgr.stats[next_source]
        print(f"  Templates: {source_stats['templates']}")
        print(f"  Estimated anomaly rate: {source_stats['anomaly_rate']:.1f}%")
        print(f"  Efficiency: {source_stats['efficiency']:.1f} logs/template")

        choice = input("\nStart this source? (y/n): ").strip().lower()
        if choice == 'y':
            source_mgr.start_new_source(next_source)
            return 'source_started'

    if len(source_mgr.completed) >= len(LOG_SOURCES):
        print("\n🎉 All sources completed!")
        print("Run create_combined_dataset() and export_ml_ready_data() for final export")
        return 'all_completed'
    else:
        print("\nNo more recommended sources or user declined.")
        print("Use manual commands if needed.")
        return 'manual_needed'

def continue_labeling_current_source(batch_size=10):
    if not source_mgr.current_source or not source_mgr.current_data:
        print("No active source. Run streamlined_workflow() first.")
        return

    unlabeled_indices = [i for i, item in enumerate(source_mgr.current_data)
                         if item['label'] is None]
    if not unlabeled_indices:
        print("All templates labeled! Run complete_current_source() to finish.")
        return

    print(f"\nContinuing labeling: {source_mgr.current_source}")
    print(f"Remaining templates: {len(unlabeled_indices)}")
    start_idx = unlabeled_indices[0]

    pos, labeled = interactive_labeling_session(
        source_mgr.current_data, source_mgr.current_source, start_idx, batch_size
    )

    save_labeling_progress(source_mgr.current_data, source_mgr.current_source)

    print(f"\nSession complete. Progress automatically saved.")
    show_labeling_progress(source_mgr.current_data, source_mgr.current_source)

    return pos, labeled

def bulk_label_remaining_templates():
    if not source_mgr.current_source or not source_mgr.current_data:
        print("No active source")
        return

    unlabeled_count = sum(1 for item in source_mgr.current_data if item['label'] is None)
    if unlabeled_count == 0:
        print("All templates already labeled")
        return

    print(f"\nBulk labeling {unlabeled_count} remaining templates...")

    high_conf_count = 0
    for item in source_mgr.current_data:
        if item['label'] is None and item['confidence'] == 'high':
            item['label'] = item['suggested']
            item['notes'] = 'Bulk: Auto-accepted high confidence'
            high_conf_count += 1
    print(f"Auto-accepted {high_conf_count} high confidence suggestions")

    remaining_unlabeled = [item for item in source_mgr.current_data if item['label'] is None]
    if remaining_unlabeled:
        total_bulk_labeled = bulk_label_by_suggestion(remaining_unlabeled)
        print(f"Bulk labeled {total_bulk_labeled} additional templates")

    still_unlabeled = [item for item in source_mgr.current_data if item['label'] is None]
    if still_unlabeled:
        print(f"\nRemaining {len(still_unlabeled)} templates - options:")
        print("1. Label all as 'normal' (conservative)")
        print("2. Label all as 'unknown_anomaly' (liberal)")
        print("3. Skip (leave unlabeled)")

        choice = input("Choose (1-3): ").strip()
        if choice == '1':
            for item in still_unlabeled:
                item['label'] = 0
                item['notes'] = 'Bulk: Default normal'
            print(f"Labeled {len(still_unlabeled)} templates as normal")
        elif choice == '2':
            for item in still_unlabeled:
                item['label'] = 7
                item['notes'] = 'Bulk: Default unknown anomaly'
            print(f"Labeled {len(still_unlabeled)} templates as unknown anomaly")

    save_labeling_progress(source_mgr.current_data, source_mgr.current_source)
    show_labeling_progress(source_mgr.current_data, source_mgr.current_source)

    return True

def complete_current_source():
    if not source_mgr.current_source or not source_mgr.current_data:
        print("No active source to complete")
        return None

    labeled_count = sum(1 for item in source_mgr.current_data if item['label'] is not None)
    total_count = len(source_mgr.current_data)
    completion_rate = labeled_count / total_count if total_count > 0 else 0

    print(f"\nCompleting source: {source_mgr.current_source}")
    print(f"Template completion: {labeled_count}/{total_count} ({completion_rate*100:.1f}%)")

    total_logs = sum(item['count'] for item in source_mgr.current_data)
    labeled_logs = sum(item['count'] for item in source_mgr.current_data if item['label'] is not None)
    log_coverage = labeled_logs / total_logs if total_logs > 0 else 0
    print(f"Log coverage: {labeled_logs:,}/{total_logs:,} ({log_coverage*100:.1f}%)")

    if completion_rate < 0.7:
        print("Warning: Less than 70% of templates labeled")
        choice = input("Continue with completion anyway? (y/n): ").strip().lower()
        if choice != 'y':
            print("Completion cancelled. Continue labeling or use bulk labeling.")
            return None

    print("\nValidating labeling quality...")
    issues = validate_labeling_quality(source_mgr.current_data)
    if len(issues) > 5:
        print(f"Found {len(issues)} potential issues. Review recommended.")
        choice = input("Continue with completion anyway? (y/n): ").strip().lower()
        if choice != 'y':
            print("Completion cancelled. Review labels first.")
            return None

    final_dataset, completed_source = source_mgr.complete_current_source()

    print(f"\n✅ Successfully completed {completed_source}!")
    print(f"Pattern library updated with new knowledge")

    next_source = source_mgr.get_next_recommended_source()
    if next_source:
        print(f"\nNext recommended source: {next_source}")
        print("Run streamlined_workflow() to continue")
    else:
        print("\n🎉 All prioritized sources completed!")
        print("Consider creating combined dataset and ML exports")

    return final_dataset

def quick_completion_statistics():
    print("\nQUICK COMPLETION STATISTICS")
    print("="*45)

    completed_count = len(source_mgr.completed)
    remaining_count = len(LOG_SOURCES) - completed_count
    print(f"Progress: {completed_count}/{len(LOG_SOURCES)} sources ({completed_count/len(LOG_SOURCES)*100:.1f}%)")

    if source_mgr.completed:
        print(f"\nCompleted sources:")
        total_logs_processed = 0
        total_anomalies_found = 0

        for source in source_mgr.completed:
            labeled_file = OUTPUT_PATH / f"{source}_labeled.csv"
            if labeled_file.exists():
                df = pd.read_csv(labeled_file)
                logs = len(df)
                anomalies = (df['AnomalyLabel'] > 0).sum()
                total_logs_processed += logs
                total_anomalies_found += anomalies
                print(f"  {source}: {logs:,} logs, {anomalies:,} anomalies ({anomalies/logs*100:.1f}%)")

        print(f"\nTotals: {total_logs_processed:,} logs, {total_anomalies_found:,} anomalies")
        if total_logs_processed > 0:
            print(f"Overall anomaly rate: {total_anomalies_found/total_logs_processed*100:.1f}%")

    if remaining_count > 0:
        print(f"\nRemaining sources: {remaining_count}")
        next_source = source_mgr.get_next_recommended_source()
        if next_source:
            print(f"Next recommended: {next_source}")

# =========================
# Initialization & Commands
# =========================
def initialize_system():
    global datasets, stats, pattern_library, source_mgr

    print("INITIALIZING LOG ANOMALY DETECTION SYSTEM (Enhanced)")
    print("="*60)

    # Ensure NLTK resources (WordNet) are available for synonyms
    try:
        _ = wordnet.synsets("test")
    except LookupError:
        print("Downloading NLTK WordNet data...")
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    print("1. Loading datasets...")
    datasets, failed_sources = load_all_datasets()
    if not datasets:
        print("ERROR: No datasets loaded successfully")
        return False

    print("2. Analyzing datasets...")
    stats = analyze_datasets(datasets)

    print("3. Initializing pattern library...")
    pattern_library = SmartPatternLibrary()
    pattern_library.load_library()

    print("4. Setting up source manager...")
    source_mgr = SourceManager(datasets, stats, pattern_library)

    print("5. Showing initial rankings...")
    rankings = rank_sources_by_priority(stats, source_mgr.completed)
    print(f"\nTop 5 recommended sources:")
    for i, rank in enumerate(rankings[:5], 1):
        print(f"{i}. {rank['source']}: priority={rank['priority']:.1f}, "
              f"anomalies={rank['anomaly_rate']:.1f}%, templates={rank['templates']}")

    print("\n✅ System initialization complete!")
    return True

def show_available_commands():
    print("\n" + "="*70)
    print("AVAILABLE COMMANDS")
    print("="*70)

    print("\n🚀 MAIN WORKFLOW:")
    print("  streamlined_workflow()                - Main entry point (start here)")
    print("  continue_labeling_current_source()    - Continue current labeling session")
    print("  complete_current_source()             - Finish and export current source")

    print("\n⚡ BULK OPERATIONS:")
    print("  bulk_label_remaining_templates()      - Auto-label remaining templates")

    print("\n📊 STATUS & ANALYSIS:")
    print("  source_mgr.get_overall_status()       - Show complete progress")
    print("  quick_completion_statistics()         - Quick stats overview")

    print("\n🧪 MODEL & TRANSFER:")
    print("  pattern_library.train_ml_from_exports() - Train TF-IDF+LR classifier from labeled CSVs")
    print("  transfer_labels_across_sources(source_mgr, threshold=90) - Cross-source label propagation")

    print("\n💾 DATA MANAGEMENT & EXPORT:")
    print("  create_combined_dataset(source_mgr.completed) - Combine completed sources")
    print("  export_ml_ready_data(combined_df)     - Export for ML training (use output of previous command)")

    print(f"\n{'='*70}")
    print("🎯 TO GET STARTED:")
    print("1. Run the initialization cell below.")
    print("2. Run: streamlined_workflow()")
    print("3. Follow the guided prompts!")
    print(f"{'='*70}")

# =========================
# Bootstrap
# =========================
if __name__ == "__main__":
    success = initialize_system()
    if success:
        show_available_commands()
        # Optional auto-start:
        # streamlined_workflow()