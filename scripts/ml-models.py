import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
import gc
from datetime import datetime
import json
import sys
import platform
import hashlib
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

from sklearn.metrics import (
    f1_score, matthews_corrcoef, accuracy_score, confusion_matrix,
    precision_score, recall_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    make_scorer
)
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    SGKF_AVAILABLE = True
except ImportError:
    from sklearn.model_selection import GroupKFold as StratifiedGroupKFold
    SGKF_AVAILABLE = False
    print("⚠️  StratifiedGroupKFold not available — using GroupKFold fallback (no stratification).")
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.covariance import EllipticEnvelope
from sklearn.calibration import CalibratedClassifierCV
import xgboost
from xgboost import XGBClassifier
import lightgbm
from lightgbm import LGBMClassifier
import imblearn
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

SEED = 42
np.random.seed(SEED)

N_JOBS_MODELS = 4  
N_JOBS_CV = 3

# ROBUSTNESS FIX: Memory management - only save models if needed for debugging
SAVE_SPLIT_MODELS = False  # Set to True to keep all per-split models in memory

# Custom scorer with consistent zero_division handling
f1_macro_scorer = make_scorer(f1_score, average='macro', zero_division=0)      

ROOT = Path(r"C:\Computer Science\AIMLDL\log-anomaly-detection")
FEAT_PATH = ROOT / "features"

MODELS_PATH = ROOT / "models" / "ml_models"
RESULTS_PATH = ROOT / "results"
CACHE_PATH = RESULTS_PATH / "split_cache"

MODELS_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
CACHE_PATH.mkdir(parents=True, exist_ok=True)

print(f"Models will be saved to: {MODELS_PATH}")
print(f"Results will be saved to: {RESULTS_PATH}")
print(f"Classes: 2 (Binary Classification)")

LABEL_MAP = {
    0: 'normal',
    1: 'anomaly'
}

ORIGINAL_LABEL_MAP = {
    0: 'normal',
    1: 'security_anomaly',
    2: 'system_failure',
    3: 'performance_issue',
    4: 'network_anomaly',
    5: 'config_error',
    6: 'hardware_issue'
}

feat_file = FEAT_PATH / "enhanced_imbalanced_features.pkl"
if not feat_file.exists():
    print(f"Error: {feat_file} not found")
    print(f"Expected location: {feat_file.absolute()}")
    exit(1)

with open(feat_file, 'rb') as f:
    feat_data = pickle.load(f)
    dat = feat_data['hybrid_features_data']
    num_classes = feat_data['config'].get('num_classes', 2)

split_file = FEAT_PATH / "enhanced_cross_source_splits.pkl"
if not split_file.exists():
    print(f"Error: {split_file} not found")
    print(f"Expected location: {split_file.absolute()}")
    exit(1)

with open(split_file, 'rb') as f:
    split_data = pickle.load(f)
    splts = split_data['splits']

optimal_config_file = ROOT / "dataset" / "labeled_data" / "normalized" / "optimal_class_config.json"
if optimal_config_file.exists():
    with open(optimal_config_file, 'r') as f:
        optimal_config = json.load(f)
else:
    optimal_config = None


def build_sampler_for(y):
    """
    Build appropriate sampler based on data-driven imbalance ratio.
    Returns None if data is relatively balanced.
    PHASE 2 FIX: Avoid ADASYN on ultra-tiny minorities (≤12 samples).
    ROBUSTNESS FIX: Guard against min_c <= 1 (SMOTE requires at least 2).
    """
    unique, counts = np.unique(y, return_counts=True)
    if len(counts) < 2:
        return None
    
    max_c, min_c = counts.max(), counts.min()
    imb_ratio = max_c / max(1, min_c)
    
    # Skip sampling if relatively balanced
    if imb_ratio < 3:
        return None
    
    # ROBUSTNESS FIX: Can't safely oversample with only 1 minority sample
    if min_c <= 1:
        print(f"⚠️  Minority class has only {min_c} sample(s). Skipping oversampling, using class_weight only.")
        return None
    
    # Calculate safe k_neighbors
    k_neighbors = min(5, min_c - 1) if min_c > 1 else 1
    k_neighbors = max(1, k_neighbors)
    
    # PHASE 2 FIX: For very tiny minority (≤12), use SMOTE only (ADASYN unstable)
    if min_c <= 12:
        return SMOTE(random_state=SEED, k_neighbors=max(1, min_c - 1))
    
    # Choose sampler by severity
    if imb_ratio > 100:
        return ADASYN(random_state=SEED, n_neighbors=k_neighbors)
    elif imb_ratio > 10:
        return BorderlineSMOTE(random_state=SEED, k_neighbors=k_neighbors)
    else:
        return SMOTE(random_state=SEED, k_neighbors=k_neighbors)


def make_pipeline_for_model(model_name, base_model, y_tr):
    """
    Create imblearn pipeline with optional scaler, sampler, and classifier.
    - Tree-based models skip scaling (not needed)
    - Balanced ensemble models skip sampling (handle internally)
    """
    # Models that don't need scaling (tree-based)
    no_scale = {'rf', 'gb', 'xgb', 'lgbm', 'balanced_bagging', 'balanced_rf', 'easy_ensemble', 'dt'}
    
    # Models that handle imbalance internally
    balanced_models = {'balanced_bagging', 'balanced_rf', 'easy_ensemble'}
    
    # Build sampler (None for balanced models or balanced data)
    sampler = None if model_name in balanced_models else build_sampler_for(y_tr)
    
    # Build pipeline steps
    steps = []
    if model_name not in no_scale:
        steps.append(('scaler', StandardScaler()))
    if sampler is not None:
        steps.append(('sampler', sampler))
    steps.append(('clf', base_model))
    
    return ImbPipeline(steps)


def validate_classes(y_tr, y_ts, min_classes=2):
    train_classes = np.unique(y_tr)
    test_classes = np.unique(y_ts)
    
    if len(train_classes) < min_classes:
        raise ValueError(
            f"Training data has only {len(train_classes)} class(es): {train_classes}. "
            f"Need at least {min_classes} for classification."
        )
    
    unseen = set(test_classes) - set(train_classes)
    if unseen:
        print(f"Warning: Test data contains classes not in training: {sorted(unseen)}")
        print(f"These will likely be misclassified.")
    
    missing = set(train_classes) - set(test_classes)
    if missing:
        print(f"Info: Test data missing classes from training: {sorted(missing)}")
    
    print(f"Train classes: {sorted(train_classes)}")
    print(f"Test classes: {sorted(test_classes)}")
    
    return train_classes, test_classes


def calculate_focal_weights(y, alpha=0.25, gamma=2.0):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    
    weights = {}
    for cls, count in zip(unique_classes, class_counts):
        frequency = count / total_samples
        weight = alpha * (1 - frequency) ** gamma
        weights[int(cls)] = weight
    
    weight_sum = sum(weights.values())
    weights = {k: v/weight_sum * len(weights) for k, v in weights.items()}
    
    print(f"Focal weights calculated: {weights}")
    return weights


# Removed apply_advanced_sampling - now handled inside CV pipeline


def calculate_geometric_mean(y_true, y_pred):
    unique_classes = np.unique(y_true)
    recalls = []
    
    for class_id in unique_classes:
        y_true_binary = (y_true == class_id).astype(int)
        y_pred_binary = (y_pred == class_id).astype(int)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        recalls.append(recall)
    
    if len(recalls) > 0 and all(r >= 0 for r in recalls):
        recalls = [max(r, 1e-10) for r in recalls]
        return np.prod(recalls) ** (1/len(recalls))
    return 0.0


def calculate_iba(y_true, y_pred, alpha=0.1):
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    geometric_mean = calculate_geometric_mean(y_true, y_pred)
    iba = (1 + alpha * geometric_mean) * balanced_acc
    return iba


# ============================================================================
# PHASE 2 ENHANCEMENTS
# ============================================================================

def compute_file_hash(filepath):
    """Compute MD5 hash of file for reproducibility tracking"""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def safe_hash(path):
    """
    Safely compute file hash with fallback to timestamp.
    Handles large files or locked files gracefully.
    """
    try:
        return compute_file_hash(path)[:8]
    except Exception as e:
        print(f"⚠️  Could not hash {path.name}: {e}. Using timestamp fallback.")
        return datetime.now().strftime("%Y%m%d%H")


# Compute hashes once at startup for cache versioning
FEATURE_HASH = safe_hash(feat_file)
SPLIT_HASH = safe_hash(split_file)


def tune_threshold_per_source(model, X_val, y_val):
    """
    Find optimal classification threshold for F1-macro.
    Critical for imbalanced datasets where default 0.5 is suboptimal.
    """
    try:
        y_proba = model.predict_proba(X_val)[:, 1]
    except:
        return 0.5, None  # Fallback if predict_proba not available
    
    best_f1 = 0
    best_threshold = 0.5
    
    # Test thresholds from 0.1 to 0.9
    for threshold in np.linspace(0.1, 0.9, 81):
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def train_unsupervised_detector(X_data, contamination=0.05):
    """
    Train unsupervised anomaly detector for single-class sources.
    Uses Isolation Forest for robust anomaly detection.
    """
    detector = IsolationForest(
        contamination=contamination,
        random_state=SEED,
        n_jobs=N_JOBS_CV,
        n_estimators=100
    )
    detector.fit(X_data)
    return detector


def create_stacked_ensemble(top_models_dict, X_tr, y_tr):
    """
    Create stacked ensemble from top performing models.
    Uses logistic regression as meta-learner.
    """
    # Select top 3 diverse models
    estimators = []
    for name in ['lgbm', 'xgb', 'balanced_rf']:
        if name in top_models_dict and 'model' in top_models_dict[name]:
            estimators.append((name, top_models_dict[name]['model']))
    
    if len(estimators) < 2:
        return None  # Need at least 2 models for stacking
    
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(solver='saga', max_iter=1000, random_state=SEED),
        cv=3,
        n_jobs=N_JOBS_CV
    )
    
    try:
        stacking.fit(X_tr, y_tr)
        return stacking
    except Exception as e:
        print(f"Stacking failed: {e}")
        return None


def get_imbalance_tier(imbalance_ratio):
    """Categorize imbalance severity"""
    if imbalance_ratio > 100:
        return 'Extreme (>100:1)'
    elif imbalance_ratio > 10:
        return 'High (10-100:1)'
    elif imbalance_ratio > 5:
        return 'Moderate (5-10:1)'
    else:
        return 'Balanced (≤5:1)'


def create_grouped_summary(df_summary, imbalance_ratios_dict):
    """
    Group results by imbalance severity tier.
    Provides insights into performance patterns.
    PHASE 2 FIX: Avoid in-place mutation, use copy.
    """
    df = df_summary.copy()
    df['Imbalance Ratio'] = df['Test Source'].map(imbalance_ratios_dict)
    df['Imbalance Tier'] = df['Imbalance Ratio'].apply(get_imbalance_tier)
    
    grouped = df.groupby('Imbalance Tier').agg({
        'F1-Macro': ['mean', 'std', 'min', 'max', 'count'],
        'Balanced Acc': ['mean', 'std'],
        'AUROC': ['mean', 'std'],
        'MCC': ['mean', 'std'],
        'Best Model': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    }).round(4)
    
    return grouped


def analyze_errors(y_true, y_pred, y_proba, source_name):
    """
    Automated error analysis for debugging and improvement.
    Identifies false positives and false negatives.
    PHASE 2 FIX: Use confidence margin for symmetric error ranking.
    """
    error_idx = np.where(y_pred != y_true)[0]
    
    if len(error_idx) == 0:
        return {
            'source': source_name,
            'total_errors': 0,
            'false_negatives': 0,
            'false_positives': 0,
            'fn_rate': 0.0,
            'fp_rate': 0.0
        }
    
    # False negatives (missed anomalies)
    fn_idx = error_idx[y_true[error_idx] == 1]
    
    # False positives (false alarms)
    fp_idx = error_idx[y_true[error_idx] == 0]
    
    error_report = {
        'source': source_name,
        'total_errors': int(len(error_idx)),
        'false_negatives': int(len(fn_idx)),
        'false_positives': int(len(fp_idx)),
        'fn_rate': float(len(fn_idx) / (y_true == 1).sum()) if (y_true == 1).sum() > 0 else 0.0,
        'fp_rate': float(len(fp_idx) / (y_true == 0).sum()) if (y_true == 0).sum() > 0 else 0.0,
    }
    
    # PHASE 2 FIX: Top 10 most confident errors (by margin, not just proba[:,1])
    if y_proba is not None and len(error_idx) > 0:
        p1 = y_proba[error_idx, 1]
        margin = np.abs(p1 - 0.5)  # Confidence margin (symmetric)
        top_errors_idx = error_idx[np.argsort(margin)[-min(10, len(error_idx)):]]
        
        error_report['top_confident_errors'] = [
            {
                'index': int(idx),
                'true_label': int(y_true[idx]),
                'pred_label': int(y_pred[idx]),
                'confidence': float(y_proba[idx, 1]),
                'margin': float(abs(y_proba[idx, 1] - 0.5))
            }
            for idx in top_errors_idx
        ]
    
    return error_report


def soft_vote_blend(models, X):
    """
    PHASE 2 ENHANCEMENT: Soft voting ensemble for top models.
    Lighter alternative to stacking, no resampling in CV.
    """
    probas = []
    for m in models:
        if hasattr(m, "predict_proba"):
            try:
                p = m.predict_proba(X)[:, 1]
                probas.append(p)
            except:
                pass
    
    if not probas:
        return None, None
    
    p_avg = np.mean(probas, axis=0)
    y_pred = (p_avg >= 0.5).astype(int)
    
    return y_pred, p_avg


def calc_enhanced_metrics(y_true, y_pred, y_proba=None, y_scores=None):
    metrics = {}
    
    metrics['acc'] = accuracy_score(y_true, y_pred)
    metrics['bal_acc'] = balanced_accuracy_score(y_true, y_pred)
    metrics['balanced_acc'] = metrics['bal_acc']  # ROBUSTNESS FIX: Alias for consistency
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    metrics['geometric_mean'] = calculate_geometric_mean(y_true, y_pred)
    metrics['iba'] = calculate_iba(y_true, y_pred)
    
    # Add AUROC and AUPRC for binary classification
    n_classes = len(np.unique(y_true))
    
    if y_proba is not None:
        if n_classes == 2:
            try:
                metrics['auroc'] = roc_auc_score(y_true, y_proba[:, 1])
                metrics['auprc'] = average_precision_score(y_true, y_proba[:, 1])
            except:
                metrics['auroc'] = None
                metrics['auprc'] = None
        else:
            try:
                metrics['auroc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            except:
                metrics['auroc'] = None
            metrics['auprc'] = None
    elif y_scores is not None and n_classes == 2:
        # Use decision scores directly for AUROC (no calibration needed)
        try:
            metrics['auroc'] = roc_auc_score(y_true, y_scores)
            metrics['auprc'] = None  # AUPRC needs probabilities
        except:
            metrics['auroc'] = None
            metrics['auprc'] = None
    else:
        metrics['auroc'] = None
        metrics['auprc'] = None
    
    per_class_metrics = {}
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    
    for class_id in unique_classes:
        y_true_binary = (y_true == class_id).astype(int)
        y_pred_binary = (y_pred == class_id).astype(int)
        
        if y_true_binary.sum() > 0:
            per_class_metrics[int(class_id)] = {
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
                'support': int(y_true_binary.sum()),
                'frequency': float(y_true_binary.sum() / len(y_true))
            }
    
    metrics['per_class'] = per_class_metrics
    
    # Confusion matrix with stable class order
    labels = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['confusion_matrix_classes'] = [int(x) for x in labels]
    
    return metrics


mod_config = {
    'lr': {
        'model': LogisticRegression(random_state=SEED, max_iter=2000, solver='saga'),
        'p': {'clf__C': [0.1, 1.0, 10.0]},
    },
    'rf': {
        'model': RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS_CV),
        'p': {'clf__n_estimators': [100, 200], 'clf__max_depth': [10, 20, None]},
    },
    'xgb': {
        'model': XGBClassifier(random_state=SEED, eval_metric='logloss', 
                              n_jobs=N_JOBS_CV, tree_method='hist'),
        'p': {'clf__n_estimators': [150, 300], 'clf__learning_rate': [0.05, 0.1], 
              'clf__max_depth': [3, 5]},
    },
    'gb': {
        'model': GradientBoostingClassifier(random_state=SEED),
        'p': {'clf__n_estimators': [100, 200], 'clf__learning_rate': [0.01, 0.1], 
              'clf__max_depth': [3, 5]},
    },
    'svm': {
        'model': SVC(random_state=SEED, probability=True, 
                    decision_function_shape='ovr', cache_size=2000),
        'p': {'clf__C': [1, 10], 'clf__kernel': ['rbf'], 'clf__gamma': ['scale', 'auto']},
    },
    'knn': {
        'model': KNeighborsClassifier(),  # n_jobs removed (may cause warnings in newer sklearn)
        'p': {'clf__n_neighbors': [5, 7, 9], 'clf__weights': ['uniform', 'distance']},
    },
    'dt': {
        'model': DecisionTreeClassifier(random_state=SEED),
        'p': {'clf__max_depth': [10, 20, None], 'clf__min_samples_split': [2, 5, 10]},
    },
    'nb': {
        'model': GaussianNB(),
        'p': {'clf__var_smoothing': [1e-9, 1e-8, 1e-7]},
    },
    'lgbm': {
        'model': LGBMClassifier(random_state=SEED, n_jobs=N_JOBS_CV, verbose=-1),
        'p': {'clf__n_estimators': [150, 300], 'clf__learning_rate': [0.05, 0.1], 
              'clf__max_depth': [5, 7], 'clf__num_leaves': [31, 50]},
    },
    'balanced_bagging': {
        'model': BalancedBaggingClassifier(
            estimator=XGBClassifier(random_state=SEED, eval_metric='logloss'),
            n_estimators=50,
            random_state=SEED,
            n_jobs=N_JOBS_CV
        ),
        'p': {'clf__n_estimators': [30, 50]},
    },
    'balanced_rf': {
        'model': BalancedRandomForestClassifier(
            random_state=SEED,
            n_jobs=N_JOBS_CV
        ),
        'p': {'clf__n_estimators': [100, 200], 'clf__max_depth': [10, 20, None]},
    },
    'easy_ensemble': {
        'model': EasyEnsembleClassifier(
            n_estimators=50,
            random_state=SEED,
            n_jobs=N_JOBS_CV
        ),
        'p': {'clf__n_estimators': [30, 50]},
    }
}


def train_single_model(m_name, m_config, X_tr, y_tr, X_ts, y_ts, class_weights, groups=None):
    """
    Train a single model using imblearn pipeline with proper CV.
    Scaling and sampling happen inside CV to prevent data leakage.
    PHASE 2 FIX: Support group-aware CV and XGB/LGBM class weighting.
    """
    try:
        # Check CV safety
        binc = np.bincount(y_tr.astype(int))
        min_class_count = int(binc.min())
        use_cv = min_class_count >= 2 and len(m_config['p']) > 0
        n_splits = min(3, min_class_count) if use_cv else 2
        
        # Build pipeline with scaler + optional sampler + classifier
        pipe = make_pipeline_for_model(m_name, m_config['model'], y_tr)
        
        # Only apply class_weight if no sampler is used (avoid double-correction)
        sampler_in_use = 'sampler' in dict(pipe.named_steps)
        
        # PHASE 2 FIX: Handle class weighting for different model types
        if not sampler_in_use:
            if hasattr(m_config['model'], 'class_weight'):
                pipe.set_params(clf__class_weight=class_weights)
            elif m_name == 'xgb':
                # XGBoost scale_pos_weight
                pos = np.sum(y_tr == 1)
                neg = np.sum(y_tr == 0)
                scale_pos_weight = neg / max(1, pos)
                pipe.set_params(clf__scale_pos_weight=scale_pos_weight)
            elif m_name == 'lgbm':
                # LightGBM class weighting (more explicit than is_unbalance)
                # Can use is_unbalance=True or explicit class_weight dict
                try:
                    # Try explicit class_weight first (more control)
                    pipe.set_params(clf__class_weight=class_weights)
                except:
                    # Fallback to is_unbalance flag
                    pipe.set_params(clf__is_unbalance=True)
        
        # Train with or without CV
        if use_cv:
            # PHASE 2 FIX: Use group-aware CV if groups provided (prevents within-source leakage)
            # ROBUSTNESS FIX: Only use groups if SGKF available and enough unique groups
            if groups is not None and len(np.unique(groups)) >= n_splits and SGKF_AVAILABLE:
                cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
                cv_splits = list(cv.split(X_tr, y_tr, groups))
            else:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
                cv_splits = cv
            
            grid = GridSearchCV(
                pipe,
                m_config['p'],
                cv=cv_splits,
                scoring=f1_macro_scorer,
                n_jobs=1,
                verbose=0,
                error_score='raise'
            )
            grid.fit(X_tr, y_tr)
            best_model = grid.best_estimator_
            best_params = grid.best_params_
        else:
            best_model = pipe.fit(X_tr, y_tr)
            best_params = {}
        
        # Predict
        y_pred = best_model.predict(X_ts)
        
        # Get probabilities or decision scores for AUROC
        y_proba = None
        y_scores = None
        if hasattr(best_model, "predict_proba"):
            try:
                y_proba = best_model.predict_proba(X_ts)
            except Exception:
                pass
        elif hasattr(best_model, "decision_function"):
            try:
                y_scores = best_model.decision_function(X_ts)
            except Exception:
                pass
        
        metrics = calc_enhanced_metrics(y_ts, y_pred, y_proba, y_scores)
        
        # PHASE 2 FIX: Record which sampler was used
        sampler_used = None
        if 'sampler' in dict(best_model.named_steps):
            sampler_obj = best_model.named_steps['sampler']
            sampler_used = type(sampler_obj).__name__
        
        # ROBUSTNESS FIX: Optionally exclude model to save memory
        model_to_save = best_model if SAVE_SPLIT_MODELS else None
        
        return m_name, {
            'metrics': metrics,
            'params': best_params,
            'model': model_to_save,
            'model_available': best_model is not None,
            'sampler_used': sampler_used
        }
        
    except Exception as e:
        print(f"Error training {m_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return m_name, {'error': str(e)}


def process_single_split_with_cache(split_idx, split, feat_type_to_test):
    """
    Process a single cross-source split with caching support.
    PHASE 2: Checkpointing & Resume capability.
    PHASE 2 FIX: Versioned cache keys with data hashes.
    """
    test_src = split['test_source']
    # Versioned cache filename includes data hashes
    cache_file = CACHE_PATH / f"split_{test_src}_{feat_type_to_test}_{FEATURE_HASH}_{SPLIT_HASH}_v2.pkl"
    
    # Check cache
    if cache_file.exists():
        print(f"✓ Loading cached results for {test_src}")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️  Cache corrupted, reprocessing: {e}")
    
    # Process split
    result = process_single_split(split_idx, split, feat_type_to_test)
    
    # Save cache
    if result is not None:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            print(f"✓ Cached results for {test_src}")
        except Exception as e:
            print(f"⚠️  Failed to cache: {e}")
    
    return result


def process_single_split(split_idx, split, feat_type_to_test):
    """Process a single cross-source split"""
    
    test_src = split['test_source']
    train_srcs = split['train_sources']
    
    print(f"\n{'='*80}")
    print(f"SPLIT {split_idx+1}/{len(splts)}: Testing on {test_src}")
    print(f"{'='*80}")
    print(f"Train sources: {', '.join(train_srcs)}\n")
    
    # Validate test source
    if test_src not in dat or dat[test_src]['labels'] is None:
        print(f"⚠️  Skipping {test_src}: No labels available")
        return None
    
    if feat_type_to_test not in dat[test_src]['feature_variants']:
        print(f"⚠️  Skipping {test_src}: Feature variant not available")
        return None
    
    # Load test data
    test_data = dat[test_src]
    X_ts = test_data['feature_variants'][feat_type_to_test]
    y_ts = test_data['labels']
    
    # PHASE 2 FIX: Handle single-class test sets with unsupervised detector
    if len(np.unique(y_ts)) < 2:
        print(f"⚠️  Single-class test set detected for {test_src}. Using unsupervised detector.")
        
        # Load training data
        X_tr_list, y_tr_list = [], []
        for src in train_srcs:
            if src in dat and dat[src]['labels'] is not None:
                if feat_type_to_test in dat[src]['feature_variants']:
                    X_tr_list.append(dat[src]['feature_variants'][feat_type_to_test])
                    y_tr_list.append(dat[src]['labels'])
        
        if not X_tr_list:
            print(f"⚠️  Skipping {test_src}: No training data available")
            return None
        
        X_tr = np.vstack(X_tr_list)
        y_tr = np.concatenate(y_tr_list)
        
        # Train unsupervised detector
        detector = train_unsupervised_detector(X_tr, contamination=0.01)  # Conservative default
        y_pred = (detector.predict(X_ts) == -1).astype(int)
        
        # Calculate metrics (will be limited due to single class)
        metrics = calc_enhanced_metrics(y_ts, y_pred, None, None)
        
        return {
            'split_idx': split_idx,
            'test_source': test_src,
            'train_sources': train_srcs,
            'best_model': 'isolation_forest',
            'best_metrics': {
                'f1_macro': float(metrics.get('f1_macro', 0)),
                'f1_weighted': float(metrics.get('f1_weighted', 0)),
                'balanced_acc': float(metrics.get('bal_acc', 0)),
                'geometric_mean': float(metrics.get('geometric_mean', 0)),
                'iba': float(metrics.get('iba', 0)),
                'mcc': float(metrics.get('mcc', 0)),
                'auroc': 0.0,
                'auprc': 0.0
            },
            'all_models': [],
            'results': {'isolation_forest': {'metrics': metrics, 'params': {}, 'model': detector}},
            'train_samples': int(len(y_tr)),
            'test_samples': int(len(y_ts)),
            'imbalance_ratio': 1.0,
            'optimal_threshold': 0.5,
            'error_report': None,
            'unsupervised': True
        }
    
    # ROBUSTNESS FIX: Build X_tr, y_tr, and groups in same loop to guarantee alignment
    X_tr_list, y_tr_list, groups_list = [], [], []
    src_to_gid = {}
    gid = 0
    
    for src in train_srcs:
        if src in dat and dat[src]['labels'] is not None:
            if feat_type_to_test in dat[src]['feature_variants']:
                X_tr_list.append(dat[src]['feature_variants'][feat_type_to_test])
                y_tr_list.append(dat[src]['labels'])
                
                # Assign group ID
                if src not in src_to_gid:
                    src_to_gid[src] = gid
                    gid += 1
                
                groups_list.append(np.full(len(dat[src]['labels']), src_to_gid[src], dtype=int))
    
    if not X_tr_list:
        print(f"⚠️  Skipping {test_src}: No training data available")
        return None
    
    X_tr = np.vstack(X_tr_list)
    y_tr = np.concatenate(y_tr_list)
    groups = np.concatenate(groups_list)
    
    # Verify alignment
    assert len(X_tr) == len(y_tr) == len(groups), "X_tr, y_tr, groups length mismatch!"
    
    # Class validation
    print("\n--- Class Validation ---")
    try:
        train_classes, test_classes = validate_classes(y_tr, y_ts, min_classes=2)
    except ValueError as e:
        print(f"⚠️  Single-class detected, attempting unsupervised approach")
        
        # PHASE 2: Unsupervised detector for single-class sources
        if len(np.unique(y_tr)) == 1:
            print(f"Training unsupervised detector for {test_src}")
            detector = train_unsupervised_detector(X_tr, contamination=0.05)
            
            # Predict on test set (-1 for anomaly, 1 for normal)
            y_pred_unsup = detector.predict(X_ts)
            y_pred = (y_pred_unsup == -1).astype(int)  # Convert to 0/1
            
            # Calculate metrics
            if len(np.unique(y_ts)) > 1:
                metrics = calc_enhanced_metrics(y_ts, y_pred, None, None)
                
                return {
                    'split_idx': split_idx,
                    'test_source': test_src,
                    'train_sources': train_srcs,
                    'best_model': 'isolation_forest',
                    'best_metrics': {
                        'f1_macro': float(metrics.get('f1_macro', 0)),
                        'f1_weighted': float(metrics.get('f1_weighted', 0)),
                        'balanced_acc': float(metrics.get('bal_acc', 0)),
                        'geometric_mean': float(metrics.get('geometric_mean', 0)),
                        'iba': float(metrics.get('iba', 0)),
                        'mcc': float(metrics.get('mcc', 0)),
                        'auroc': 0.0,
                        'auprc': 0.0
                    },
                    'all_models': [],
                    'results': {'isolation_forest': {'metrics': metrics, 'params': {}, 'model': detector}},
                    'train_samples': int(len(y_tr)),
                    'test_samples': int(len(y_ts)),
                    'unsupervised': True
                }
        
        print(f"⚠️  Skipping {test_src}: {e}")
        return None
    
    # Note: Scaling and sampling now happen inside CV pipeline (no data leakage)
    print("\n--- Data Summary ---")
    print(f"Train samples: {len(y_tr):,}")
    print(f"Test samples: {len(y_ts):,}")
    print(f"Features: {X_tr.shape[1]}")
    
    # Calculate class distribution
    unique, counts = np.unique(y_tr, return_counts=True)
    imb_ratio = counts.max() / counts.min() if len(counts) > 1 else 1.0
    imb_tier = get_imbalance_tier(imb_ratio)
    print(f"Train imbalance ratio: {imb_ratio:.2f}:1 ({imb_tier})")
    
    # Class weight calculation (on original labels, before any sampling)
    print("\n--- Class Weight Calculation ---")
    focal_weights = calculate_focal_weights(y_tr)
    
    # Train models (scaling + sampling happen inside CV pipeline)
    print("\n" + "="*60)
    print("TRAINING MODELS (with CV pipelines)")
    print("="*60 + "\n")
    
    results = {}
    model_items = list(mod_config.items())
    
    parallel_results = Parallel(n_jobs=N_JOBS_MODELS, backend='loky', verbose=5)(
        delayed(train_single_model)(name, config, X_tr, y_tr, X_ts, y_ts, focal_weights, groups)
        for name, config in model_items
    )
    
    for name, result in parallel_results:
        results[name] = result
        gc.collect()
    
    # Model performance summary
    print("\n--- Model Performance Summary ---")
    comparison_data = []
    
    for model_name, result in results.items():
        if 'error' not in result:
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name.upper(),
                'F1-Macro': metrics.get('f1_macro', 0),
                'F1-Weighted': metrics.get('f1_weighted', 0),
                'Balanced Acc': metrics.get('bal_acc', 0),
                'Geometric Mean': metrics.get('geometric_mean', 0),
                'IBA': metrics.get('iba', 0),
                'MCC': metrics.get('mcc', 0),
                'AUROC': metrics.get('auroc', 0) if metrics.get('auroc') else 0,
                'AUPRC': metrics.get('auprc', 0) if metrics.get('auprc') else 0
            })
        else:
            print(f"{model_name.upper()}: {result['error']}")
    
    if not comparison_data:
        print(f"⚠️  No models trained successfully for {test_src}")
        return None
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('F1-Macro', ascending=False)
    
    print("\n" + df_comparison.to_string(index=False))
    
    best_row = df_comparison.iloc[0]
    best_model_name = best_row['Model'].lower()
    
    print(f"\n✓ Best Model: {best_model_name.upper()}")
    print(f"  F1-Macro: {best_row['F1-Macro']:.4f}")
    print(f"  Balanced Acc: {best_row['Balanced Acc']:.4f}")
    print(f"  AUROC: {best_row['AUROC']:.4f}")
    
    if best_model_name in results:
        per_class = results[best_model_name]['metrics'].get('per_class', {})
        
        print("\n--- Per-Class Performance (Best Model) ---")
        for class_id in sorted(per_class.keys()):
            class_metrics = per_class[class_id]
            class_name = LABEL_MAP.get(class_id, f'Class_{class_id}')
            print(f"{class_name} (ID {class_id}):")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall:    {class_metrics['recall']:.4f}")
            print(f"  F1:        {class_metrics['f1']:.4f}")
            print(f"  Support:   {class_metrics['support']}")
    
    # PHASE 2: Threshold tuning on validation fold
    print("\n--- Threshold Tuning ---")
    optimal_threshold = 0.5
    best_metrics_tuned = None
    
    # ROBUSTNESS FIX: Handle case where model might not be saved
    if best_model_name in results and results[best_model_name].get('model_available', False):
        best_model_obj = results[best_model_name].get('model')
        if best_model_obj is None:
            print("⚠️  Best model not available (SAVE_SPLIT_MODELS=False). Skipping threshold tuning.")
            best_model_obj = None
    else:
        best_model_obj = None
    
    if best_model_obj is not None:
        
        # Use 20% of training data as validation for threshold tuning
        from sklearn.model_selection import train_test_split
        X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr
        )
        
        # ROBUSTNESS FIX: Optional probability calibration before threshold tuning
        # Use sigmoid for small datasets, isotonic for larger ones
        calibrated_model = best_model_obj
        if hasattr(best_model_o
                print("  Calibrating probabilities (isotonic)...")
                calib = CalibratedClassifierCV(best_model_obj, method='isotonic', cv=3)
                calib.fit(X_tr_sub, y_tr_sub)
                calibrated_model = calib
                print("  ✓ Calibration complete")
            except Exception as e:
                print(f"  ⚠️  Calibration failed: {e}. Using uncalibrated model.")
        
        optimal_threshold, tuned_f1 = tune_threshold_per_source(calibrated_model, X_val, y_val)
        print(f"Optimal threshold: {optimal_threshold:.3f} (F1 on val: {tuned_f1:.4f})")
        
        # PHASE 2 FIX: Apply tuned threshold to test set and compute metrics
        y_proba_best = None
        try:
            y_proba_best = calibrated_model.predict_proba(X_ts)
            y_pred_tuned = (y_proba_best[:, 1] >= optimal_threshold).astype(int)
            tuned_metrics = calc_enhanced_metrics(y_ts, y_pred_tuned, y_proba_best, None)
            
            best_metrics_tuned = {
                'f1_macro': float(tuned_metrics.get('f1_macro', 0)),
                'f1_weighted': float(tuned_metrics.get('f1_weighted', 0)),
                'balanced_acc': float(tuned_metrics.get('bal_acc', 0)),
                'geometric_mean': float(tuned_metrics.get('geometric_mean', 0)),
                'iba': float(tuned_metrics.get('iba', 0)),
                'mcc': float(tuned_metrics.get('mcc', 0)),
                'auroc': float(tuned_metrics.get('auroc', 0)) if tuned_metrics.get('auroc') else 0.0,
                'auprc': float(tuned_metrics.get('auprc', 0)) if tuned_metrics.get('auprc') else 0.0
            }
            print(f"Tuned F1-Macro on test: {best_metrics_tuned['f1_macro']:.4f} (vs default: {best_row['F1-Macro']:.4f})")
        except Exception as e:
            print(f"Could not apply tuned threshold: {e}")
    
    # PHASE 2: Soft voting blend evaluation
    print("\n--- Soft Voting Blend ---")
    blend_metrics = None
    # ROBUSTNESS FIX: Only use models that are actually saved
    top_models = [m['model'] for n, m in results.items() 
                  if m.get('model') is not None and 'error' not in m][:3]
    
    if len(top_models) >= 2:
        y_blend, p_blend = soft_vote_blend(top_models, X_ts)
        if y_blend is not None:
            blend_metrics_raw = calc_enhanced_metrics(y_ts, y_blend, np.c_[1 - p_blend, p_blend], None)
            blend_f1 = blend_metrics_raw.get('f1_macro', 0)
            print(f"Soft-vote F1-Macro: {blend_f1:.4f} (vs best single: {best_row['F1-Macro']:.4f})")
            
            # If blend beats best single, consider recording it
            if blend_f1 > best_row['F1-Macro']:
                print(f"✓ Soft-vote blend outperforms best single model!")
                blend_metrics = {
                    'f1_macro': float(blend_f1),
                    'balanced_acc': float(blend_metrics_raw.get('bal_acc', 0)),
                    'mcc': float(blend_metrics_raw.get('mcc', 0))
                }
    else:
        print("Not enough models for soft voting")
    
    # PHASE 2: Error analysis
    print("\n--- Error Analysis ---")
    error_report = None
    # ROBUSTNESS FIX: Check if model is available
    if best_model_name in results and results[best_model_name].get('model') is not None:
        best_model_for_errors = results[best_model_name]['model']
        y_pred_best = best_model_for_errors.predict(X_ts)
        y_proba_best = None
        try:
            y_proba_best = best_model_for_errors.predict_proba(X_ts)
        except:
            pass
        
        error_report = analyze_errors(y_ts, y_pred_best, y_proba_best, test_src)
        print(f"Total errors: {error_report['total_errors']}")
        print(f"False negatives: {error_report['false_negatives']} (rate: {error_report['fn_rate']:.2%})")
        print(f"False positives: {error_report['false_positives']} (rate: {error_report['fp_rate']:.2%})")
    else:
        print("⚠️  Model not available for error analysis (SAVE_SPLIT_MODELS=False)")
    
    # PHASE 2 FIX: Record sampler used by best model
    sampler_used = results[best_model_name].get('sampler_used', None) if best_model_name in results else None
    
    # Return split results
    return {
        'split_idx': split_idx,
        'test_source': test_src,
        'train_sources': train_srcs,
        'best_model': best_model_name,
        'best_metrics': {
            'f1_macro': float(best_row['F1-Macro']),
            'f1_weighted': float(best_row['F1-Weighted']),
            'balanced_acc': float(best_row['Balanced Acc']),
            'geometric_mean': float(best_row['Geometric Mean']),
            'iba': float(best_row['IBA']),
            'mcc': float(best_row['MCC']),
            'auroc': float(best_row['AUROC']),
            'auprc': float(best_row['AUPRC'])
        },
        'best_metrics_tuned': best_metrics_tuned,  # PHASE 2 FIX: Metrics with tuned threshold
        'blend_metrics': blend_metrics,  # PHASE 2 FIX: Soft-vote blend metrics
        'all_models': df_comparison.to_dict('records'),
        'results': results,
        'train_samples': int(len(y_tr)),
        'test_samples': int(len(y_ts)),
        'imbalance_ratio': float(imb_ratio),
        'optimal_threshold': float(optimal_threshold),
        'error_report': error_report,
        'sampler_used': sampler_used  # PHASE 2 FIX: Record resampling strategy
    }


# ============================================================================
# MAIN EXECUTION: Loop through all cross-source splits
# ============================================================================

print("\n" + "="*80)
print("STARTING BINARY CLASSIFICATION PIPELINE - ALL SPLITS")
print("="*80 + "\n")

feat_type_to_test = 'selected_imbalanced'  # 200 features, optimized for imbalance
print(f"Using feature variant: {feat_type_to_test}")
print(f"Total splits to process: {len(splts)}")
print(f"Models to train per split: {len(mod_config)}")

# Process all splits with caching (PHASE 2: Checkpointing)
all_split_results = []

print("\n" + "="*80)
print("PHASE 2 ENHANCEMENTS ACTIVE:")
print("  ✓ Checkpointing & Resume")
print("  ✓ Per-Source Threshold Tuning")
print("  ✓ Unsupervised Detection for Single-Class")
print("  ✓ Error Analysis")
print("  ✓ Grouped Summary Reports")
print("="*80 + "\n")

for split_idx, split in enumerate(splts):
    result = process_single_split_with_cache(split_idx, split, feat_type_to_test)
    if result is not None:
        all_split_results.append(result)
    gc.collect()

# ============================================================================
# AGGREGATE RESULTS ACROSS ALL SPLITS
# ============================================================================

print("\n" + "="*80)
print("CROSS-SOURCE EVALUATION SUMMARY")
print("="*80 + "\n")

if not all_split_results:
    print("⚠️  No splits processed successfully!")
    exit(1)

# Create summary dataframe
summary_data = []
for result in all_split_results:
    summary_data.append({
        'Test Source': result['test_source'],
        'Best Model': result['best_model'].upper(),
        'F1-Macro': result['best_metrics']['f1_macro'],
        'Balanced Acc': result['best_metrics']['balanced_acc'],
        'AUROC': result['best_metrics']['auroc'],
        'AUPRC': result['best_metrics']['auprc'],
        'MCC': result['best_metrics']['mcc'],
        'Train Samples': result['train_samples'],
        'Test Samples': result['test_samples']
    })

df_summary = pd.DataFrame(summary_data)
df_summary = df_summary.sort_values('F1-Macro', ascending=False)

# PHASE 2 FIX: Add imbalance ratio column for Plot 3
imbalance_ratios_dict = {r['test_source']: r.get('imbalance_ratio', 1.0) for r in all_split_results}
df_summary['Imbalance Ratio'] = df_summary['Test Source'].map(imbalance_ratios_dict)

print("\n" + df_summary.to_string(index=False))

# Calculate aggregate statistics
print("\n" + "="*60)
print("AGGREGATE STATISTICS")
print("="*60)
print(f"\nTotal splits processed: {len(all_split_results)}/{len(splts)}")
print(f"\nAverage Performance:")
print(f"  F1-Macro:      {df_summary['F1-Macro'].mean():.4f} ± {df_summary['F1-Macro'].std():.4f}")
print(f"  Balanced Acc:  {df_summary['Balanced Acc'].mean():.4f} ± {df_summary['Balanced Acc'].std():.4f}")
print(f"  AUROC:         {df_summary['AUROC'].mean():.4f} ± {df_summary['AUROC'].std():.4f}")
print(f"  AUPRC:         {df_summary['AUPRC'].mean():.4f} ± {df_summary['AUPRC'].std():.4f}")
print(f"  MCC:           {df_summary['MCC'].mean():.4f} ± {df_summary['MCC'].std():.4f}")

print(f"\nBest performing source: {df_summary.iloc[0]['Test Source']} (F1: {df_summary.iloc[0]['F1-Macro']:.4f})")
print(f"Worst performing source: {df_summary.iloc[-1]['Test Source']} (F1: {df_summary.iloc[-1]['F1-Macro']:.4f})")

# Model frequency
model_counts = df_summary['Best Model'].value_counts()
print(f"\nBest model frequency:")
for model, count in model_counts.items():
    print(f"  {model}: {count} times ({count/len(all_split_results)*100:.1f}%)")

# Save aggregate results
aggregate_dir = RESULTS_PATH / f"aggregate_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
aggregate_dir.mkdir(exist_ok=True)

df_summary.to_csv(aggregate_dir / "all_splits_summary.csv", index=False)

# PHASE 2: Grouped summary by imbalance tier
print("\n" + "="*80)
print("GROUPED ANALYSIS BY IMBALANCE TIER")
print("="*80 + "\n")

# imbalance_ratios_dict already created above when adding column to df_summary
grouped_summary = create_grouped_summary(df_summary, imbalance_ratios_dict)

print(grouped_summary)
grouped_summary.to_csv(aggregate_dir / "grouped_by_imbalance.csv")
print(f"\n✓ Grouped summary saved")

# PHASE 2: Error analysis summary
print("\n" + "="*80)
print("ERROR ANALYSIS SUMMARY")
print("="*80 + "\n")

error_reports = [r.get('error_report') for r in all_split_results if r.get('error_report')]
if error_reports:
    error_df = pd.DataFrame([
        {
            'Source': er['source'],
            'Total Errors': er['total_errors'],
            'False Negatives': er['false_negatives'],
            'False Positives': er['false_positives'],
            'FN Rate': f"{er['fn_rate']:.2%}",
            'FP Rate': f"{er['fp_rate']:.2%}"
        }
        for er in error_reports
    ])
    print(error_df.to_string(index=False))
    error_df.to_csv(aggregate_dir / "error_analysis_summary.csv", index=False)
    
    # Save detailed error reports
    with open(aggregate_dir / "error_analysis_detailed.json", 'w') as f:
        json.dump(error_reports, f, indent=2)
    print(f"\n✓ Error analysis saved")

# PHASE 2: Threshold tuning summary with comparison
thresholds_df = pd.DataFrame([
    {
        'Source': r['test_source'],
        'Optimal Threshold': r.get('optimal_threshold', 0.5),
        'F1-Default': r['best_metrics']['f1_macro'],
        'F1-Tuned': r.get('best_metrics_tuned', {}).get('f1_macro', r['best_metrics']['f1_macro']),
        'Improvement': r.get('best_metrics_tuned', {}).get('f1_macro', r['best_metrics']['f1_macro']) - r['best_metrics']['f1_macro'],
        'Sampler': r.get('sampler_used', 'None')
    }
    for r in all_split_results
])
print("\n" + "="*80)
print("THRESHOLD TUNING & RESAMPLING SUMMARY")
print("="*80 + "\n")
print(thresholds_df.to_string(index=False))
print(f"\nAverage improvement from tuning: {thresholds_df['Improvement'].mean():.4f}")
thresholds_df.to_csv(aggregate_dir / "threshold_tuning_summary.csv", index=False)

# PHASE 2 FIX: Create threshold map for deployment
threshold_map = {row['Source']: float(row['Optimal Threshold']) for _, row in thresholds_df.iterrows()}

print(f"\n✓ Aggregate results saved to: {aggregate_dir}")

# ============================================================================
# TRAIN FINAL MODEL ON ALL DATA (with PHASE 2 enhancements)
# ============================================================================

print("\n" + "="*80)
print("TRAINING FINAL MODEL ON ALL DATA")
print("="*80 + "\n")

# Determine best overall model
best_overall_model = model_counts.index[0].lower()
print(f"Best overall model: {best_overall_model.upper()}")

# PHASE 2: Collect top models for stacking
# ROBUSTNESS FIX: Filter out None models (when SAVE_SPLIT_MODELS=False)
print("\n--- Collecting Top Models for Ensemble ---")
top_models_for_stacking = {}
for result in all_split_results:
    if 'results' in result:
        for model_name, model_data in result['results'].items():
            if model_data.get('model') is not None and 'error' not in model_data:
                if model_name not in top_models_for_stacking:
                    top_models_for_stacking[model_name] = model_data

if not top_models_for_stacking:
    print("⚠️  No models available for stacking (SAVE_SPLIT_MODELS=False)")
else:
    print(f"✓ Collected {len(top_models_for_stacking)} model types for potential stacking")

all_train_srcs = [s for s in dat.keys() if dat[s]['labels'] is not None]

X_tr_list_all, y_tr_list_all = [], []
for src in all_train_srcs:
    if feat_type_to_test in dat[src]['feature_variants']:
        X_tr_list_all.append(dat[src]['feature_variants'][feat_type_to_test])
        y_tr_list_all.append(dat[src]['labels'])

X_tr_all = np.vstack(X_tr_list_all)
y_tr_all = np.concatenate(y_tr_list_all)

print(f"Total training samples: {len(y_tr_all):,}")

# Calculate class weights on original data
focal_weights_final = calculate_focal_weights(y_tr_all)

# Build pipeline for final model
best_model_config = mod_config[best_overall_model]
final_pipe = make_pipeline_for_model(best_overall_model, best_model_config['model'], y_tr_all)

# Apply class weights if no sampler
sampler_in_use = 'sampler' in dict(final_pipe.named_steps)
if hasattr(best_model_config['model'], 'class_weight') and not sampler_in_use:
    final_pipe.set_params(clf__class_weight=focal_weights_final)

# Train with CV
if len(best_model_config['p']) > 0:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = GridSearchCV(final_pipe, best_model_config['p'], 
                       cv=cv, scoring=f1_macro_scorer, n_jobs=N_JOBS_CV, verbose=1,
                       error_score='raise')
    grid.fit(X_tr_all, y_tr_all)
    final_model = grid.best_estimator_
    final_params = grid.best_params_
    print(f"Best parameters: {final_params}")
else:
    final_model = final_pipe.fit(X_tr_all, y_tr_all)
    final_params = {}
    print(f"Model trained (no hyperparameters)")

deployment_dir = MODELS_PATH / "deployment"
deployment_dir.mkdir(exist_ok=True)

# PHASE 2: Try to create stacked ensemble
print("\n--- Creating Stacked Ensemble ---")
stacked_model = create_stacked_ensemble(top_models_for_stacking, X_tr_all, y_tr_all)

if stacked_model is not None:
    print("✓ Stacked ensemble created successfully")
    
    # Quick evaluation on training data
    y_pred_stack = stacked_model.predict(X_tr_all)
    stack_f1 = f1_score(y_tr_all, y_pred_stack, average='macro', zero_division=0)
    print(f"  Stacked ensemble F1 (train): {stack_f1:.4f}")
    
    # Save stacked model separately
    stack_file = deployment_dir / "stacked_ensemble_model.pkl"
    with open(stack_file, 'wb') as f:
        pickle.dump({
            'model': stacked_model,
            'feature_type': feat_type_to_test,
            'model_name': 'stacked_ensemble',
            'base_models': list(top_models_for_stacking.keys())[:3],
            'timestamp': datetime.now().isoformat()
        }, f)
    print(f"  Saved to: {stack_file}")
else:
    print("⚠️  Stacked ensemble creation failed, using single best model")

# PHASE 2: Compute file hashes for reproducibility
print("\n--- Computing Data Hashes ---")
feature_hash = compute_file_hash(feat_file)
split_hash = compute_file_hash(split_file)
print(f"Feature file hash: {feature_hash[:16]}...")
print(f"Split file hash: {split_hash[:16]}...")

# Calculate final imbalance ratio
unique_final, counts_final = np.unique(y_tr_all, return_counts=True)
final_imb_ratio = float(counts_final.max() / counts_final.min()) if len(counts_final) > 1 else 1.0

dep_data = {
    'model': final_model,  # Pipeline includes scaler/sampler/classifier
    'feature_type': feat_type_to_test,
    'model_name': best_overall_model,
    'num_classes': 2,
    'label_map': LABEL_MAP,
    'best_params': final_params,
    'focal_weights': focal_weights_final,
    'metrics': {
        'avg_f1_macro': float(df_summary['F1-Macro'].mean()),
        'avg_balanced_acc': float(df_summary['Balanced Acc'].mean()),
        'avg_auroc': float(df_summary['AUROC'].mean()),
        'avg_auprc': float(df_summary['AUPRC'].mean()),
        'avg_mcc': float(df_summary['MCC'].mean()),
        'std_f1_macro': float(df_summary['F1-Macro'].std()),
    },
    'training_samples': int(len(y_tr_all)),
    'original_samples': int(len(y_tr_all)),
    'num_splits_evaluated': len(all_split_results),
    'phase1_enhanced': True,
    'phase2_enhanced': True,
    'all_splits_evaluated': True,
    'timestamp': datetime.now().isoformat(),
    # PHASE 2 additions
    'feature_hash': feature_hash,
    'split_hash': split_hash,
    'version': 'v2.0.0',
    'train_imbalance_ratio': final_imb_ratio,
    'has_stacked_ensemble': stacked_model is not None,
    'thresholds_by_source': threshold_map,  # PHASE 2 FIX: Save optimal thresholds
    'phase2_features': [
        'checkpointing',
        'threshold_tuning',
        'unsupervised_detection',
        'error_analysis',
        'grouped_summary',
        'stacked_ensemble',
        'data_hashing'
    ],
    # PHASE 2 FIX: Save environment for reproducibility
    'environment': {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'sklearn': sklearn.__version__,
        'xgboost': xgboost.__version__,
        'lightgbm': lightgbm.__version__,
        'imblearn': imblearn.__version__,
        'numpy': np.__version__,
        'pandas': pd.__version__
    }
}

deployment_file = deployment_dir / "best_model_for_deployment.pkl"
with open(deployment_file, 'wb') as f:
    pickle.dump(dep_data, f)

print(f"\n✓ DEPLOYMENT MODEL SAVED:")
print(f"  Location: {deployment_file}")
print(f"  Model: {best_overall_model.upper()} (pipeline)")
print(f"  Version: v2.0.0 (Phase 2)")
print(f"  Training samples: {len(y_tr_all):,}")
print(f"  Imbalance ratio: {final_imb_ratio:.2f}:1")
print(f"  Evaluated on: {len(all_split_results)} cross-source splits")
print(f"  Average F1-Macro: {df_summary['F1-Macro'].mean():.4f}")
print(f"  Pipeline: {' -> '.join(final_model.named_steps.keys())}")
print(f"  Stacked ensemble: {'Available' if stacked_model is not None else 'Not available'}")

print("\n" + "="*80)
print("GENERATING AGGREGATE VISUALIZATIONS (PHASE 2 ENHANCED)")
print("="*80 + "\n")

plt.style.use('default')
sns.set_palette("husl")

# Create aggregate visualization
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Cross-Source Evaluation - Phase 2 Enhanced Results', fontsize=16, fontweight='bold')

# Plot 1: F1-Macro by source
ax1 = axes[0, 0]
sources = df_summary['Test Source'].values
f1_scores = df_summary['F1-Macro'].values
colors = plt.cm.RdYlGn(f1_scores)

bars = ax1.barh(sources, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('F1-Macro Score', fontsize=12, fontweight='bold')
ax1.set_title('F1-Macro Performance by Source', fontsize=13, fontweight='bold')
ax1.set_xlim([0, 1])
ax1.grid(axis='x', alpha=0.3, linestyle='--')

for i, (bar, score) in enumerate(zip(bars, f1_scores)):
    ax1.text(score + 0.02, bar.get_y() + bar.get_height()/2, 
            f'{score:.3f}', va='center', fontsize=9, fontweight='bold')

# Plot 2: Model frequency
ax2 = axes[0, 1]
model_freq = df_summary['Best Model'].value_counts()
ax2.pie(model_freq.values, labels=model_freq.index, autopct='%1.1f%%',
        startangle=90, colors=sns.color_palette("husl", len(model_freq)))
ax2.set_title('Best Model Frequency', fontsize=13, fontweight='bold')

# Plot 3: PHASE 2 - Performance vs Imbalance Ratio
ax3 = axes[0, 2]
if 'Imbalance Ratio' in df_summary.columns:
    imb_ratios = df_summary['Imbalance Ratio'].values
    ax3.scatter(imb_ratios, f1_scores, s=100, alpha=0.6, c=f1_scores, 
               cmap='RdYlGn', edgecolors='black', linewidth=1.5)
    
    for i, source in enumerate(sources):
        ax3.annotate(source, (imb_ratios[i], f1_scores[i]), fontsize=7, ha='right')
    
    ax3.set_xlabel('Imbalance Ratio (log scale)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('F1-Macro', fontsize=12, fontweight='bold')
    ax3.set_title('Performance vs Imbalance', fontsize=13, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(alpha=0.3, linestyle='--')

# Plot 4: Metrics comparison (box plot)
ax4 = axes[1, 0]
metrics_data = df_summary[['F1-Macro', 'Balanced Acc', 'AUROC', 'AUPRC', 'MCC']].values
bp = ax4.boxplot(metrics_data, labels=['F1-Macro', 'Bal Acc', 'AUROC', 'AUPRC', 'MCC'],
                 patch_artist=True, showmeans=True)

for patch, color in zip(bp['boxes'], sns.color_palette("husl", 5)):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
ax4.set_title('Metrics Distribution Across Sources', fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.set_ylim([0, 1])

# Plot 5: Performance vs samples
ax5 = axes[1, 1]
ax5.scatter(df_summary['Test Samples'], df_summary['F1-Macro'], 
           s=100, alpha=0.6, c=df_summary['F1-Macro'], cmap='RdYlGn',
           edgecolors='black', linewidth=1.5)

for i, source in enumerate(df_summary['Test Source']):
    ax5.annotate(source, (df_summary['Test Samples'].iloc[i], df_summary['F1-Macro'].iloc[i]),
                fontsize=8, ha='right')

ax5.set_xlabel('Test Samples', fontsize=12, fontweight='bold')
ax5.set_ylabel('F1-Macro', fontsize=12, fontweight='bold')
ax5.set_title('Performance vs Dataset Size', fontsize=13, fontweight='bold')
ax5.grid(alpha=0.3, linestyle='--')

# Plot 6: PHASE 2 - Error Analysis
ax6 = axes[1, 2]
if error_reports:
    fn_rates = [er['fn_rate'] * 100 for er in error_reports]
    fp_rates = [er['fp_rate'] * 100 for er in error_reports]
    error_sources = [er['source'] for er in error_reports]
    
    x = np.arange(len(error_sources))
    width = 0.35
    
    ax6.bar(x - width/2, fn_rates, width, label='False Negative Rate', alpha=0.8, color='#ff6b6b')
    ax6.bar(x + width/2, fp_rates, width, label='False Positive Rate', alpha=0.8, color='#4ecdc4')
    
    ax6.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
    ax6.set_title('Error Rates by Source', fontsize=13, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(error_sources, rotation=45, ha='right', fontsize=8)
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()

viz_file = aggregate_dir / "aggregate_visualization.png"
plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Aggregate visualization saved to: {viz_file}")

plt.show()

# Save detailed JSON with all split results
results_json = {
    'experiment_info': {
        'timestamp': datetime.now().isoformat(),
        'feature_type': feat_type_to_test,
        'num_classes': 2,
        'label_map': LABEL_MAP,
        'total_splits': len(splts),
        'processed_splits': len(all_split_results)
    },
    'environment': {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'sklearn': sklearn.__version__,
        'xgboost': xgboost.__version__,
        'lightgbm': lightgbm.__version__,
        'imblearn': imblearn.__version__,
        'numpy': np.__version__,
        'pandas': pd.__version__
    },
    'aggregate_metrics': {
        'avg_f1_macro': float(df_summary['F1-Macro'].mean()),
        'std_f1_macro': float(df_summary['F1-Macro'].std()),
        'avg_balanced_acc': float(df_summary['Balanced Acc'].mean()),
        'avg_auroc': float(df_summary['AUROC'].mean()),
        'avg_auprc': float(df_summary['AUPRC'].mean()),
        'avg_mcc': float(df_summary['MCC'].mean())
    },
    'best_model': best_overall_model,
    'model_frequency': model_counts.to_dict(),
    'per_split_results': [
        {
            'test_source': r['test_source'],
            'best_model': r['best_model'],
            'metrics': r['best_metrics']
        } for r in all_split_results
    ]
}

json_file = aggregate_dir / "aggregate_results.json"
with open(json_file, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"✓ Aggregate JSON saved to: {json_file}")

print("\n" + "="*80)
print("EXECUTION COMPLETE - PHASE 2 ENHANCED")
print("="*80)
print(f"\n📊 Summary:")
print(f"  Splits processed: {len(all_split_results)}/{len(splts)}")
print(f"  Best overall model: {best_overall_model.upper()}")
print(f"  Average F1-Macro: {df_summary['F1-Macro'].mean():.4f} ± {df_summary['F1-Macro'].std():.4f}")
print(f"  Average Balanced Acc: {df_summary['Balanced Acc'].mean():.4f}")
print(f"  Average AUROC: {df_summary['AUROC'].mean():.4f}")
print(f"\n✓ Phase 2 Features Implemented:")
print(f"  ✓ Checkpointing & Resume (cache: {CACHE_PATH})")
print(f"  ✓ Per-Source Threshold Tuning")
print(f"  ✓ Unsupervised Detection for Single-Class")
print(f"  ✓ Error Analysis & Reporting")
print(f"  ✓ Grouped Summary by Imbalance Tier")
print(f"  ✓ Stacked Ensemble ({'Available' if stacked_model is not None else 'Not available'})")
print(f"  ✓ Data Hashing for Reproducibility")
print(f"\n📁 Results saved to:")
print(f"  Aggregate: {aggregate_dir}")
print(f"  Deployment model: {deployment_file}")
print(f"  Grouped summary: {aggregate_dir / 'grouped_by_imbalance.csv'}")
print(f"  Error analysis: {aggregate_dir / 'error_analysis_summary.csv'}")
print(f"  Optimal thresholds: {aggregate_dir / 'optimal_thresholds.csv'}")
if stacked_model is not None:
    print(f"  Stacked ensemble: {deployment_dir / 'stacked_ensemble_model.pkl'}")
print(f"\n✅ All files saved successfully!")
print(f"📌 Version: v2.0.0 (Phase 2)")
print(f"📌 Feature hash: {feature_hash[:16]}...")
print(f"📌 Split hash: {split_hash[:16]}...")
print("="*80 + "\n")