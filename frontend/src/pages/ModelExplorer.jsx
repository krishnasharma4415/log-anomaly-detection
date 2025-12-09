import { useState, useEffect } from 'react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import Badge from '../components/ui/Badge';
import { CheckCircle, Loader, TrendingUp, Award, Zap } from 'lucide-react';
import { useToast } from '../components/ui/Toast';
import { useModel } from '../context/ModelContext';
import api from '../services/api';
import { SkeletonCard } from '../components/ui/Skeleton';

// Model definitions based on actual trained models with real metrics from results
const modelDefinitions = {
  ml: [
    {
      id: 'xgboost',
      name: 'XGBoost + SMOTE (Best ML)',
      modelFile: 'best_model_for_deployment.pkl',
      description: 'Best overall traditional ML model with SMOTE oversampling. Handles class imbalance effectively with gradient boosting.',
      features: ['Gradient Boosting', 'SMOTE', 'Class Weights', '200 Features', 'Cross-Validated'],
      metrics: {
        f1_macro: 0.8380,
        balanced_acc: 0.8975,
        auroc: 0.9566,
        mcc: 0.7023,
        sources: 13
      },
      available: true,
      rank: 1,
    },
  ],
  dl: [
    {
      id: 'cnn-attention',
      name: 'CNN + Multi-Head Attention',
      modelFile: 'best_dl_model_cnn_attention.pth',
      description: '1D-CNN with multi-head attention achieving best DL performance. Excels on balanced datasets like Proxifier (F1: 0.997).',
      features: ['Deep Learning', '8-Head Attention', 'GPU Accelerated', '1D Convolution', 'Sequence Analysis'],
      metrics: {
        f1_macro: 0.6701,
        balanced_acc: 0.7259,
        auroc: 0.7257,
        sources: 13
      },
      available: true,
      rank: 3,
    },
    {
      id: 'stacked-ae',
      name: 'Stacked Autoencoder',
      modelFile: 'stacked_ae_model.pth',
      description: 'Autoencoder with classification head. Good for feature learning and anomaly detection.',
      features: ['Autoencoder', 'Deep Features', 'Reconstruction', 'Unsupervised Learning'],
      metrics: {
        f1_macro: 0.5518,
        balanced_acc: 0.6172,
        auroc: 0.6280,
        sources: 13
      },
      available: true,
      rank: 5,
    },
    {
      id: 'vae',
      name: 'Variational Autoencoder (VAE)',
      modelFile: 'vae_model.pth',
      description: 'Probabilistic approach to anomaly detection using reconstruction error.',
      features: ['VAE', 'Probabilistic', 'Latent Space', 'Generative'],
      metrics: {
        f1_macro: 0.5091,
        balanced_acc: 0.6380,
        auroc: 0.7447,
        sources: 13
      },
      available: true,
      rank: 4,
    },
    {
      id: 'focal-nn',
      name: 'Focal Loss Neural Network',
      modelFile: 'flnn_model.pth',
      description: 'Deep network with focal loss for handling extreme class imbalance.',
      features: ['Focal Loss', 'Imbalance Handling', 'Deep Features'],
      metrics: {
        f1_macro: 0.5234,
        balanced_acc: 0.5777,
        auroc: 0.6231,
        sources: 13
      },
      available: true,
      rank: 8,
    },
    {
      id: 'tabnet',
      name: 'TabNet',
      modelFile: 'tabnet_model.pth',
      description: 'Attention-based tabular data network with sequential feature selection.',
      features: ['TabNet', 'Attention Mechanism', 'Feature Selection', 'Interpretable'],
      metrics: {
        f1_macro: 0.5210,
        balanced_acc: 0.5833,
        auroc: 0.5632,
        sources: 13
      },
      available: true,
      rank: 10,
    },
    {
      id: 'transformer',
      name: 'Transformer Encoder',
      modelFile: 'transformer_model.pth',
      description: 'Pure transformer architecture for sequence modeling.',
      features: ['Transformer', 'Self-Attention', 'Positional Encoding'],
      metrics: {
        f1_macro: 0.4377,
        balanced_acc: 0.5340,
        auroc: 0.5696,
        sources: 13
      },
      available: true,
      rank: 12,
    },
  ],
  bert: [
    {
      id: 'deberta_v3',
      name: 'DeBERTa-v3 (Best BERT)',
      modelFile: 'deployment/best_model/',
      description: 'Best performing BERT variant with disentangled attention mechanism. Strong AUROC: 0.695.',
      features: ['Disentangled Attention', 'Best BERT', 'Optimal Threshold: 0.9', '13 Sources Trained'],
      metrics: {
        f1_macro: 0.5221,
        balanced_acc: 0.5950,
        auroc: 0.6952,
        mcc: 0.1455,
        sources: 13
      },
      available: true,
      rank: 6,
    },
    {
      id: 'logbert',
      name: 'LogBERT',
      modelFile: 'deployment/logbert/',
      description: 'BERT fine-tuned specifically for log data with MLM pretraining. Excellent AUROC: 0.752.',
      features: ['Transformer', 'MLM Pretraining', 'Log-Specific', 'Fine-tuned'],
      metrics: {
        f1_macro: 0.5105,
        balanced_acc: 0.6021,
        auroc: 0.7522,
        mcc: 0.1621,
        sources: 13
      },
      available: true,
      rank: 7,
    },
    {
      id: 'dapt_bert',
      name: 'DAPT-BERT',
      modelFile: 'deployment/dapt_bert/',
      description: 'Domain-Adaptive Pretraining BERT with adversarial training. Top AUROC: 0.753.',
      features: ['Domain Adaptation', 'Multi-Head Attention', 'Adversarial Training', 'Robust'],
      metrics: {
        f1_macro: 0.5016,
        balanced_acc: 0.5909,
        auroc: 0.7531,
        mcc: 0.1843,
        sources: 13
      },
      available: true,
      rank: 9,
    },
    {
      id: 'mpnet',
      name: 'MPNet',
      modelFile: 'deployment/mpnet/',
      description: 'Masked and Permuted Pre-training for language understanding with efficient attention pooling.',
      features: ['Semantic', 'Attention Pooling', 'Efficient', 'Balanced'],
      metrics: {
        f1_macro: 0.4529,
        balanced_acc: 0.5469,
        auroc: 0.5767,
        mcc: 0.0461,
        sources: 13
      },
      available: true,
      rank: 11,
    },
  ],
  advanced: [
    {
      id: 'meta',
      name: 'Meta-Learning (MAML)',
      modelFile: 'meta_learning/best_meta_model.pt',
      description: 'Few-shot learning for rapid adaptation. Best on Proxifier (F1: 0.974). CHAMPION MODEL with F1: 0.942.',
      features: ['Few-Shot', 'Rapid Adaptation', 'MAML', 'Zero-Shot', '10 Sources'],
      metrics: {
        f1_macro: 0.9422,
        balanced_acc: 0.9697,
        auroc: 0.9920,
        mcc: 0.8848,
        sources: 10
      },
      available: true,
      rank: 2,
      champion: true,
    },
    {
      id: 'fedlogcl',
      name: 'FedLogCL (Federated Contrastive)',
      modelFile: 'federated_contrastive/final_best_model.pt',
      description: 'Federated Contrastive Learning with template-aware attention. Privacy-preserving cross-source learning.',
      features: ['Contrastive Learning', 'Federated', 'Template-Aware', 'Privacy-Preserving'],
      metrics: {
        f1_macro: 0.3959,
        balanced_acc: 0.4908,
        auroc: 0.5335,
        sources: 13
      },
      available: true,
      rank: 13,
    },
    {
      id: 'hlogformer',
      name: 'HLogFormer (Hierarchical Transformer)',
      modelFile: 'hlogformer/best_model.pt',
      description: 'Hierarchical Transformer with temporal LSTM and source adapters. Multi-level feature extraction.',
      features: ['Hierarchical', 'Temporal LSTM', 'Source Adapters', 'Multi-task'],
      metrics: {
        f1_macro: 0.2134,
        balanced_acc: 0.4932,
        auroc: 0.4675,
        mcc: -0.0324,
        sources: 13
      },
      available: true,
      rank: 14,
    },
  ],
  ensemble: [
    {
      id: 'ensemble-avg',
      name: 'Ensemble (Averaging)',
      modelFile: 'N/A (combines multiple models)',
      description: 'Combines predictions from ML, DL, and BERT models by averaging probabilities. Leverages strengths of each approach.',
      features: ['Multi-Model', 'Averaging', 'Robust', 'High Accuracy'],
      metrics: {
        f1_macro: 0.82,
        balanced_acc: 0.87,
        auroc: 0.91
      },
      available: true,
    },
    {
      id: 'ensemble-vote',
      name: 'Ensemble (Voting)',
      modelFile: 'N/A (combines multiple models)',
      description: 'Combines predictions from ML, DL, and BERT models by majority voting. Consensus-based decision making.',
      features: ['Multi-Model', 'Voting', 'Consensus', 'Reliable'],
      metrics: {
        f1_macro: 0.80,
        balanced_acc: 0.85,
        auroc: 0.89
      },
      available: true,
    },
  ],
};

export default function ModelExplorer() {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const { activeModel, updateActiveModel } = useModel();
  const [modelInfo, setModelInfo] = useState(null);
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

        allModels.push({
          ...model,
          category: categoryUpper,
          // Use actual metrics from results
          f1Score: model.metrics?.f1_macro ? (model.metrics.f1_macro * 100).toFixed(1) : 'N/A',
          auroc: model.metrics?.auroc ? (model.metrics.auroc * 100).toFixed(1) : 'N/A',
          balancedAcc: model.metrics?.balanced_acc ? (model.metrics.balanced_acc * 100).toFixed(1) : 'N/A',
          mcc: model.metrics?.mcc !== undefined ? model.metrics.mcc.toFixed(3) : 'N/A',
          latency: category === 'ml' ? '10-15' : category === 'dl' ? '50-100' : category === 'bert' ? '150-200' : '100-300',
          isLoaded: modelInfo?.[`${category}_model_loaded`] || false,
        });
      });
    });

    // Sort by rank if available, otherwise by F1 score
    return allModels.sort((a, b) => {
      if (a.rank && b.rank) return a.rank - b.rank;
      const aF1 = parseFloat(a.f1Score) || 0;
      const bF1 = parseFloat(b.f1Score) || 0;
      return bF1 - aF1;
    });
  };

  const filteredModels = selectedCategory === 'all'
    ? getAllModels()
    : getAllModels().filter(m => m.category === selectedCategory);

  const handleSetActive = (modelType, modelName) => {
    updateActiveModel(modelType, modelName);
    addToast(`Switched to ${modelName}`, 'success');
  };

  if (loading) {
    return (
      <div className="space-y-8">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-2">Model Explorer</h1>
          <p className="text-slate-600 dark:text-slate-400">Loading model information...</p>
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
        <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-2">Model Explorer</h1>
        <p className="text-slate-600 dark:text-slate-400">Browse and compare all 14 trained models across 5 categories</p>
      </div>

      {/* Performance Summary Card */}
      <Card neon className="bg-gradient-to-br from-primary-500/10 to-cyan-500/10 border-primary-500/20">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <Award className="w-6 h-6 text-primary-500" />
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Top Performers</h3>
          </div>
          <Badge variant="success" pulse>Live Models</Badge>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white/50 dark:bg-slate-800/50 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4 text-primary-500" />
              <p className="text-xs text-slate-600 dark:text-slate-400 font-semibold">Best Overall</p>
            </div>
            <p className="text-lg font-bold text-primary-600 dark:text-primary-400">Meta-Learning</p>
            <p className="text-sm text-slate-600 dark:text-slate-400">F1: 94.2% • AUROC: 99.2%</p>
          </div>
          <div className="bg-white/50 dark:bg-slate-800/50 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Zap className="w-4 h-4 text-cyan-500" />
              <p className="text-xs text-slate-600 dark:text-slate-400 font-semibold">Best Traditional ML</p>
            </div>
            <p className="text-lg font-bold text-cyan-600 dark:text-cyan-400">XGBoost + SMOTE</p>
            <p className="text-sm text-slate-600 dark:text-slate-400">F1: 83.8% • AUROC: 95.7%</p>
          </div>
          <div className="bg-white/50 dark:bg-slate-800/50 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Award className="w-4 h-4 text-purple-500" />
              <p className="text-xs text-slate-600 dark:text-slate-400 font-semibold">Best Deep Learning</p>
            </div>
            <p className="text-lg font-bold text-purple-600 dark:text-purple-400">CNN + Attention</p>
            <p className="text-sm text-slate-600 dark:text-slate-400">F1: 67.0% • AUROC: 72.6%</p>
          </div>
        </div>
      </Card>

      {/* Model Status Info */}
      {modelInfo && (
        <Card neon className="bg-indigo-900/20">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">Currently Loaded Models</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${modelInfo.ml_model_loaded ? 'bg-green-500 animate-pulse' : 'bg-slate-400 dark:bg-slate-600'}`} />
                  <span className="text-sm text-slate-600 dark:text-slate-400">ML</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${modelInfo.dl_model_loaded ? 'bg-green-500 animate-pulse' : 'bg-slate-400 dark:bg-slate-600'}`} />
                  <span className="text-sm text-slate-600 dark:text-slate-400">DL</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${modelInfo.bert_model_loaded ? 'bg-green-500 animate-pulse' : 'bg-slate-400 dark:bg-slate-600'}`} />
                  <span className="text-sm text-slate-600 dark:text-slate-400">BERT</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${modelInfo.fedlogcl_model_loaded ? 'bg-green-500 animate-pulse' : 'bg-slate-400 dark:bg-slate-600'}`} />
                  <span className="text-sm text-slate-600 dark:text-slate-400">FedLogCL</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${modelInfo.hlogformer_model_loaded ? 'bg-green-500 animate-pulse' : 'bg-slate-400 dark:bg-slate-600'}`} />
                  <span className="text-sm text-slate-600 dark:text-slate-400">HLogFormer</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${modelInfo.meta_model_loaded ? 'bg-green-500 animate-pulse' : 'bg-slate-400 dark:bg-slate-600'}`} />
                  <span className="text-sm text-slate-600 dark:text-slate-400">Meta</span>
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
              neon={model.isLoaded || model.champion}
              className={`h-full flex flex-col ${!model.available ? 'opacity-60' : ''} ${model.champion ? 'border-2 border-primary-500' : ''}`}
            >
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-1">{model.name}</h3>
                  <div className="flex gap-2 flex-wrap">
                    <Badge variant="primary">{model.category}</Badge>
                    {model.rank && model.rank <= 3 && (
                      <Badge variant="success">
                        Rank #{model.rank}
                      </Badge>
                    )}
                    {model.champion && (
                      <Badge variant="success" pulse>
                        <Award className="w-3 h-3" />
                        CHAMPION
                      </Badge>
                    )}
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

              <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">{model.description}</p>

              {/* Metrics */}
              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3 transition-colors duration-200">
                  <p className="text-xs text-slate-600 dark:text-slate-400 mb-1">F1-Macro</p>
                  <p className="text-xl font-bold text-primary-600 dark:text-primary-400">
                    {model.f1Score === 'N/A' ? 'N/A' : `${model.f1Score}%`}
                  </p>
                </div>
                <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3 transition-colors duration-200">
                  <p className="text-xs text-slate-600 dark:text-slate-400 mb-1">AUROC</p>
                  <p className="text-xl font-bold text-cyan-600 dark:text-cyan-400">
                    {model.auroc === 'N/A' ? 'N/A' : `${model.auroc}%`}
                  </p>
                </div>
                <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3 transition-colors duration-200">
                  <p className="text-xs text-slate-600 dark:text-slate-400 mb-1">Balanced Acc</p>
                  <p className="text-xl font-bold text-purple-600 dark:text-purple-400">
                    {model.balancedAcc === 'N/A' ? 'N/A' : `${model.balancedAcc}%`}
                  </p>
                </div>
                <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3 transition-colors duration-200">
                  <p className="text-xs text-slate-600 dark:text-slate-400 mb-1">Latency</p>
                  <p className="text-xl font-bold text-green-600 dark:text-green-400">{model.latency}ms</p>
                </div>
              </div>

              {/* Additional metrics if available */}
              {model.mcc !== 'N/A' && (
                <div className="mb-4 bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
                  <p className="text-xs text-slate-600 dark:text-slate-400 mb-1">MCC</p>
                  <p className="text-lg font-bold text-orange-600 dark:text-orange-400">{model.mcc}</p>
                </div>
              )}

              {/* Features */}
              <div className="flex flex-wrap gap-2 mb-4">
                {model.features.map((feature) => (
                  <Badge key={feature} variant="neutral" className="text-xs">
                    {feature}
                  </Badge>
                ))}
              </div>

              {/* Model File Info */}
              <div className="text-xs text-slate-400 dark:text-slate-500 mb-4 font-mono break-all">
                {model.modelFile}
              </div>

              {/* Action */}
              <Button
                variant={model.isLoaded ? 'secondary' : 'primary'}
                size="md"
                className="w-full mt-auto"
                onClick={() => handleSetActive(model.category.toLowerCase(), model.name)}
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
        <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-3">Model Categories & Performance</h3>
        <div className="space-y-2 text-sm text-slate-600 dark:text-slate-400">
          <p>• <span className="text-slate-900 dark:text-slate-100 font-semibold">ML Models</span>: Traditional machine learning (XGBoost + SMOTE) - Avg F1: 83.8%</p>
          <p>• <span className="text-slate-900 dark:text-slate-100 font-semibold">DL Models</span>: Deep learning neural networks - Best: CNN+Attention (F1: 67.0%)</p>
          <p>• <span className="text-slate-900 dark:text-slate-100 font-semibold">BERT Models</span>: Transformer-based models - Best: DeBERTa-v3 (F1: 52.2%, AUROC: 75.3%)</p>
          <p>• <span className="text-slate-900 dark:text-slate-100 font-semibold">Advanced Models</span>: Meta-Learning (F1: 94.2%) • FedLogCL (Privacy-preserving) • HLogFormer (Hierarchical)</p>
          <p>• <span className="text-slate-900 dark:text-slate-100 font-semibold">Ensemble Models</span>: Combine multiple models for robustness</p>
          <p className="mt-4 text-cyan-600 dark:text-cyan-400">
            ✓ Currently using: <span className="font-semibold">{activeModel.name}</span> for predictions
          </p>
          {modelInfo && (
            <p className="text-slate-400 dark:text-slate-500">
              Label mapping: {Object.entries(modelInfo.label_map).map(([k, v]) => `${k}=${v}`).join(', ')}
            </p>
          )}
          <p className="mt-4 text-primary-600 dark:text-primary-400">
            📊 Total: 14 trained models • Evaluated on 13 log sources • 21,200 total samples
          </p>
        </div>
      </Card>
    </div>
  );
}
