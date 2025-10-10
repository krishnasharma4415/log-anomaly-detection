import React from 'react';
import { Brain, Layers } from 'lucide-react';
import { formatAnomalyType } from '../../utils/anomalyColors';

export default function ModelInfo({ modelInfo }) {
  if (!modelInfo) return null;
  
  const getModelDescription = (modelType) => {
    const descriptions = {
      'DANN-BERT': 'Domain Adversarial Neural Network with BERT for cross-domain log analysis.',
      'LORA-BERT': 'Low-Rank Adaptation of BERT for efficient parameter-tuning.',
      'HYBRID-BERT': 'Combines BERT embeddings with template features for enhanced accuracy.',
      'ML Model': 'Traditional machine learning with advanced feature engineering.'
    };
    return descriptions[modelType] || 'Advanced multi-class anomaly detection model.';
  };

  const classificationType = modelInfo.classification_type || 'multi-class';
  const numClasses = modelInfo.num_classes || 7;
  const labelMap = modelInfo.label_map || {};
  const modelType = modelInfo.model_type || 'Unknown';
  
  return (
    <div className="mt-6 p-4 bg-purple-900/30 rounded-lg border border-purple-500/30">
      <div className="flex items-center gap-2 mb-3">
        <Brain className="w-5 h-5 text-purple-400" />
        <p className="text-purple-200 font-semibold">Model Information</p>
      </div>
      
      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-400">Model Type:</span>
          <span className="text-purple-200 font-medium">{modelType}</span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-gray-400">Classification:</span>
          <span className="text-purple-200">{classificationType}</span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-gray-400">Classes:</span>
          <span className="text-purple-200">{numClasses}</span>
        </div>
        
        {modelInfo.uses_templates !== undefined && (
          <div className="flex justify-between">
            <span className="text-gray-400">Template Features:</span>
            <span className="text-purple-200">{modelInfo.uses_templates ? 'Yes' : 'No'}</span>
          </div>
        )}
        
        {modelInfo.device && (
          <div className="flex justify-between">
            <span className="text-gray-400">Device:</span>
            <span className="text-purple-200">{modelInfo.device}</span>
          </div>
        )}
      </div>
      
      <p className="text-purple-300 text-xs mt-3 pt-3 border-t border-purple-500/20">
        {getModelDescription(modelType)}
      </p>
      
      {}
      {Object.keys(labelMap).length > 0 && (
        <div className="mt-3 pt-3 border-t border-purple-500/20">
          <div className="flex items-center gap-2 mb-2">
            <Layers className="w-4 h-4 text-purple-400" />
            <span className="text-purple-200 text-xs font-semibold">Detectable Anomaly Types:</span>
          </div>
          <div className="grid grid-cols-2 gap-1">
            {Object.entries(labelMap).map(([id, name]) => (
              <div key={id} className="text-xs text-purple-300">
                • {formatAnomalyType(name)}
              </div>
            ))}
          </div>
        </div>
      )}
      
      {}
      {modelInfo.metrics && Object.keys(modelInfo.metrics).length > 0 && (
        <div className="mt-3 pt-3 border-t border-purple-500/20">
          <span className="text-purple-200 text-xs font-semibold">Performance Metrics:</span>
          <div className="grid grid-cols-2 gap-2 mt-2">
            {modelInfo.metrics.f1_macro && (
              <div className="text-xs">
                <span className="text-gray-400">F1 Macro:</span>{' '}
                <span className="text-green-400">{(modelInfo.metrics.f1_macro * 100).toFixed(1)}%</span>
              </div>
            )}
            {modelInfo.metrics.f1_weighted && (
              <div className="text-xs">
                <span className="text-gray-400">F1 Weighted:</span>{' '}
                <span className="text-green-400">{(modelInfo.metrics.f1_weighted * 100).toFixed(1)}%</span>
              </div>
            )}
            {modelInfo.metrics.balanced_accuracy && (
              <div className="text-xs">
                <span className="text-gray-400">Balanced Acc:</span>{' '}
                <span className="text-green-400">{(modelInfo.metrics.balanced_accuracy * 100).toFixed(1)}%</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}