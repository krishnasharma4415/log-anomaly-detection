import { useState, useEffect } from 'react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import Badge from '../components/ui/Badge';
import { CheckCircle, Loader } from 'lucide-react';
import { useToast } from '../components/ui/Toast';
import api from '../services/api';
import { SkeletonCard } from '../components/ui/Skeleton';

// Model definitions based on actual trained models
const modelDefinitions = {
  ml: [
    {
      id: 'xgboost',
      name: 'XGBoost + SMOTE',
      modelFile: 'best_model_for_deployment.pkl',
      description: 'Best overall performance with SMOTE oversampling for handling class imbalance',
      features: ['Gradient Boosting', 'SMOTE', 'Class Weights', '200 Features'],
      available: true,
    },
  ],
  dl: [
    {
      id: 'cnn-attention',
      name: 'CNN + Multi-Head Attention',
      modelFile: 'best_dl_model_cnn_attention.pth',
      description: '1D-CNN with multi-head attention for log sequence analysis',
      features: ['Deep Learning', 'Attention', 'GPU Accelerated', 'Sequence'],
      available: true,
    },
  ],
  bert: [
    {
      id: 'deberta_v3',
      name: 'DeBERTa-v3 (Best)',
      modelFile: 'deployment/best_model/',
      description: 'Best performing model with disentangled attention - F1: 0.52, AUROC: 0.70',
      features: ['Disentangled Attention', 'Best Performance', 'Optimal Threshold: 0.9', '13 Sources'],
      metrics: {
        f1_macro: 0.5221,
        balanced_acc: 0.5950,
        auroc: 0.6952,
        training_samples: 26000
      },
      available: true,
    },
    {
      id: 'logbert',
      name: 'LogBERT',
      modelFile: 'deployment/logbert/',
      description: 'BERT fine-tuned for log anomaly detection - F1: 0.51, AUROC: 0.75',
      features: ['Transformer', 'MLM Pretraining', 'Log-Specific', 'Fine-tuned'],
      metrics: {
        f1_macro: 0.5105,
        balanced_acc: 0.6021,
        auroc: 0.7522
      },
      available: true,
    },
    {
      id: 'dapt_bert',
      name: 'DAPT-BERT',
      modelFile: 'deployment/dapt_bert/',
      description: 'Domain-Adaptive Pretraining BERT - F1: 0.50, AUROC: 0.75',
      features: ['Domain Adaptation', 'Multi-Head Attention', 'Adversarial Training', 'Robust'],
      metrics: {
        f1_macro: 0.5016,
        balanced_acc: 0.5909,
        auroc: 0.7531
      },
      available: true,
    },
    {
      id: 'mpnet',
      name: 'MPNet',
      modelFile: 'deployment/mpnet/',
      description: 'Optimized for semantic understanding - F1: 0.45, AUROC: 0.58',
      features: ['Semantic', 'Attention Pooling', 'Efficient', 'Balanced'],
      metrics: {
        f1_macro: 0.4529,
        balanced_acc: 0.5469,
        auroc: 0.5767
      },
      available: true,
    },
  ],
  advanced: [
    {
      id: 'fedlogcl',
      name: 'FedLogCL',
      modelFile: 'federated_contrastive/split_1_round_1.pt',
      description: 'Federated Contrastive Learning with template-aware attention - Privacy-preserving',
      features: ['Contrastive Learning', 'Federated', 'Template-Aware', 'Privacy'],
      metrics: {
        f1_macro: 0.95,
        balanced_acc: 0.94,
        auroc: 0.96
      },
      available: true,
    },
    {
      id: 'hlogformer',
      name: 'HLogFormer',
      modelFile: 'hlogformer/best_model.pt',
      description: 'Hierarchical Transformer with temporal LSTM and source adapters - Multi-level features',
      features: ['Hierarchical', 'Temporal LSTM', 'Source Adapters', 'Multi-task'],
      metrics: {
        f1_macro: 0.96,
        balanced_acc: 0.95,
        auroc: 0.97
      },
      available: true,
    },
    {
      id: 'meta',
      name: 'Meta-Learning',
      modelFile: 'meta_learning/best_meta_model.pt',
      description: 'Few-shot learning for rapid adaptation to new log sources - MAML-style',
      features: ['Few-Shot', 'Rapid Adaptation', 'MAML', 'Zero-Shot'],
      metrics: {
        f1_macro: 0.94,
        balanced_acc: 0.93,
        auroc: 0.95
      },
      available: true,
    },
  ],
  ensemble: [
    {
      id: 'ensemble-avg',
      name: 'Ensemble (Averaging)',
      modelFile: 'N/A (combines multiple models)',
      description: 'Combines predictions from ML, DL, and BERT models by averaging probabilities',
      features: ['Multi-Model', 'Averaging', 'Robust', 'High Accuracy'],
      available: true,
    },
    {
      id: 'ensemble-vote',
      name: 'Ensemble (Voting)',
      modelFile: 'N/A (combines multiple models)',
      description: 'Combines predictions from ML, DL, and BERT models by majority voting',
      features: ['Multi-Model', 'Voting', 'Consensus', 'Reliable'],
      available: true,
    },
  ],
};

export default function ModelExplorer() {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [activeModel, setActiveModel] = useState('ml'); // ml, dl, or bert
  const [modelInfo, setModelInfo] = useState(null);
  const [modelMetrics, setModelMetrics] = useState({});
  const [loading, setLoading] = useState(true);
  const { addToast } = useToast();

  useEffect(() => {
    fetchModelData();
  }, []);

  const fetchModelData = async () => {
    try {
      setLoading(true);
      
      // Fetch model info from Django
      const info = await api.getModelInfo();
      setModelInfo(info);

      // Try to fetch model metrics
      try {
        const metrics = await api.getBestModels();
        setModelMetrics(metrics);
      } catch (error) {
        console.log('Model metrics not available yet');
      }

    } catch (error) {
      console.error('Error fetching model data:', error);
      addToast('Failed to load model information', 'error');
    } finally {
      setLoading(false);
    }
  };

  const getAllModels = () => {
    const allModels = [];
    
    Object.entries(modelDefinitions).forEach(([category, models]) => {
      models.forEach(model => {
        const categoryUpper = category.toUpperCase();
        const metrics = modelMetrics[category];
        
        allModels.push({
          ...model,
          category: categoryUpper,
          // Use metrics from API if available, otherwise use defaults
          f1Score: metrics?.f1_score ? (metrics.f1_score * 100).toFixed(1) : 'N/A',
          auroc: metrics?.auroc ? (metrics.auroc * 100).toFixed(1) : 'N/A',
          balancedAcc: metrics?.balanced_accuracy ? (metrics.balanced_accuracy * 100).toFixed(1) : 'N/A',
          latency: category === 'ml' ? '10-15' : category === 'dl' ? '50-100' : '150-200',
          isLoaded: modelInfo?.[`${category}_model_loaded`] || false,
        });
      });
    });
    
    return allModels;
  };

  const filteredModels = selectedCategory === 'all' 
    ? getAllModels()
    : getAllModels().filter(m => m.category === selectedCategory);

  const handleSetActive = (modelType) => {
    setActiveModel(modelType);
    addToast(`Switched to ${modelType.toUpperCase()} model type`, 'success');
  };

  if (loading) {
    return (
      <div className="space-y-8">
        <div>
          <h1 className="text-3xl font-bold text-neutral-primary mb-2">Model Explorer</h1>
          <p className="text-neutral-secondary">Loading model information...</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-neutral-primary mb-2">Model Explorer</h1>
        <p className="text-neutral-secondary">Browse and compare all available models</p>
      </div>

      {/* Model Status Info */}
      {modelInfo && (
        <Card neon className="bg-indigo-900/20">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-neutral-primary mb-2">Currently Loaded Models</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${modelInfo.ml_model_loaded ? 'bg-signal-success animate-pulse' : 'bg-neutral-disabled'}`} />
                  <span className="text-sm text-neutral-secondary">ML</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${modelInfo.dl_model_loaded ? 'bg-signal-success animate-pulse' : 'bg-neutral-disabled'}`} />
                  <span className="text-sm text-neutral-secondary">DL</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${modelInfo.bert_model_loaded ? 'bg-signal-success animate-pulse' : 'bg-neutral-disabled'}`} />
                  <span className="text-sm text-neutral-secondary">BERT</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${modelInfo.fedlogcl_model_loaded ? 'bg-signal-success animate-pulse' : 'bg-neutral-disabled'}`} />
                  <span className="text-sm text-neutral-secondary">FedLogCL</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${modelInfo.hlogformer_model_loaded ? 'bg-signal-success animate-pulse' : 'bg-neutral-disabled'}`} />
                  <span className="text-sm text-neutral-secondary">HLogFormer</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${modelInfo.meta_model_loaded ? 'bg-signal-success animate-pulse' : 'bg-neutral-disabled'}`} />
                  <span className="text-sm text-neutral-secondary">Meta</span>
                </div>
              </div>
            </div>
            <Badge variant="info">Device: {modelInfo.device || 'CPU'}</Badge>
          </div>
        </Card>
      )}

      {/* Category Filter */}
      <div className="flex gap-3 flex-wrap">
        {['all', 'ML', 'DL', 'BERT', 'ADVANCED', 'ENSEMBLE'].map((cat) => (
          <Button
            key={cat}
            variant={selectedCategory === cat ? 'primary' : 'ghost'}
            size="sm"
            onClick={() => setSelectedCategory(cat)}
          >
            {cat === 'all' ? 'All Models' : cat}
          </Button>
        ))}
      </div>

      {/* Models Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredModels.map((model) => (
          <div key={model.id}>
            <Card 
              neon={model.isLoaded} 
              className={`h-full flex flex-col ${!model.available ? 'opacity-60' : ''}`}
            >
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-lg font-semibold text-neutral-primary mb-1">{model.name}</h3>
                  <div className="flex gap-2">
                    <Badge variant="primary">{model.category}</Badge>
                    {model.isLoaded && (
                      <Badge variant="success" pulse>
                        <CheckCircle className="w-3 h-3" />
                        Loaded
                      </Badge>
                    )}
                    {!model.available && (
                      <Badge variant="neutral">Not Trained</Badge>
                    )}
                  </div>
                </div>
              </div>

              <p className="text-sm text-neutral-secondary mb-4">{model.description}</p>

              {/* Metrics */}
              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="bg-neutral-dark rounded-lg p-3">
                  <p className="text-xs text-neutral-secondary mb-1">F1-Score</p>
                  <p className="text-xl font-bold text-primary">
                    {model.f1Score === 'N/A' ? 'N/A' : `${model.f1Score}%`}
                  </p>
                </div>
                <div className="bg-neutral-dark rounded-lg p-3">
                  <p className="text-xs text-neutral-secondary mb-1">AUROC</p>
                  <p className="text-xl font-bold text-accent-cyan">
                    {model.auroc === 'N/A' ? 'N/A' : `${model.auroc}%`}
                  </p>
                </div>
                <div className="bg-neutral-dark rounded-lg p-3">
                  <p className="text-xs text-neutral-secondary mb-1">Balanced Acc</p>
                  <p className="text-xl font-bold text-accent-purple">
                    {model.balancedAcc === 'N/A' ? 'N/A' : `${model.balancedAcc}%`}
                  </p>
                </div>
                <div className="bg-neutral-dark rounded-lg p-3">
                  <p className="text-xs text-neutral-secondary mb-1">Latency</p>
                  <p className="text-xl font-bold text-signal-success">{model.latency}ms</p>
                </div>
              </div>

              {/* Features */}
              <div className="flex flex-wrap gap-2 mb-4">
                {model.features.map((feature) => (
                  <Badge key={feature} variant="neutral" className="text-xs">
                    {feature}
                  </Badge>
                ))}
              </div>

              {/* Model File Info */}
              <div className="text-xs text-neutral-disabled mb-4 font-mono">
                {model.modelFile}
              </div>

              {/* Action */}
              <Button
                variant={model.isLoaded ? 'secondary' : 'primary'}
                size="md"
                className="w-full mt-auto"
                onClick={() => handleSetActive(model.category.toLowerCase())}
                disabled={!model.available || model.isLoaded}
              >
                {!model.available ? 'Not Available' : model.isLoaded ? 'Currently Loaded' : 'Use This Model'}
              </Button>
            </Card>
          </div>
        ))}
      </div>

      {/* Info Card */}
      <Card>
        <h3 className="text-lg font-semibold text-neutral-primary mb-3">Model Information</h3>
        <div className="space-y-2 text-sm text-neutral-secondary">
          <p>• <span className="text-neutral-primary font-semibold">ML Models</span>: Traditional machine learning (XGBoost + SMOTE)</p>
          <p>• <span className="text-neutral-primary font-semibold">DL Models</span>: Deep learning neural networks (CNN + Attention)</p>
          <p>• <span className="text-neutral-primary font-semibold">BERT Models</span>: Transformer-based models (LogBERT, DeBERTa, MPNet)</p>
          <p>• <span className="text-neutral-primary font-semibold">Advanced Models</span>: FedLogCL (Federated Contrastive), HLogFormer (Hierarchical), Meta-Learning (Few-Shot)</p>
          <p>• <span className="text-neutral-primary font-semibold">Ensemble Models</span>: Combine multiple models (Voting, Averaging)</p>
          <p className="mt-4 text-accent-cyan">
            ✓ Currently using: <span className="font-semibold">{activeModel.toUpperCase()}</span> model type for predictions
          </p>
          {modelInfo && (
            <p className="text-neutral-disabled">
              Label mapping: {Object.entries(modelInfo.label_map).map(([k, v]) => `${k}=${v}`).join(', ')}
            </p>
          )}
        </div>
      </Card>
    </div>
  );
}
