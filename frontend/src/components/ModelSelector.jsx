import React from 'react';
import { Cpu, Zap, Brain, GitMerge } from 'lucide-react';

const MODEL_OPTIONS = [
  {
    id: 'ml',
    name: 'ML Model',
    icon: Cpu,
    description: 'Traditional Machine Learning',
    color: 'blue'
  },
  {
    id: 'dann_bert',
    name: 'DANN BERT',
    icon: Brain,
    description: 'Domain Adversarial Neural Network',
    color: 'purple'
  },
  {
    id: 'lora_bert',
    name: 'LoRA BERT',
    icon: Zap,
    description: 'Low-Rank Adaptation',
    color: 'pink'
  },
  {
    id: 'hybrid_bert',
    name: 'Hybrid BERT',
    icon: GitMerge,
    description: 'Combined Approach',
    color: 'green'
  }
];

export default function ModelSelector({ selectedModel, onModelChange, disabled }) {
  return (
    <div className="mb-4">
      <label className="block text-purple-200 mb-3 text-sm font-medium">
        Select Model
      </label>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {MODEL_OPTIONS.map((model) => {
          const Icon = model.icon;
          const isSelected = selectedModel === model.id;
          
          return (
            <button
              key={model.id}
              onClick={() => onModelChange(model.id)}
              disabled={disabled}
              className={`p-3 rounded-lg border-2 transition-all text-left ${
                isSelected
                  ? `border-${model.color}-500 bg-${model.color}-500/20`
                  : 'border-white/20 bg-slate-800/30 hover:border-white/40'
              } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
            >
              <div className="flex items-start gap-3">
                <Icon className={`w-5 h-5 flex-shrink-0 mt-0.5 ${
                  isSelected ? `text-${model.color}-400` : 'text-purple-300'
                }`} />
                <div className="flex-1 min-w-0">
                  <div className="font-semibold text-white text-sm mb-0.5">
                    {model.name}
                  </div>
                  <div className={`text-xs ${
                    isSelected ? `text-${model.color}-300` : 'text-purple-400'
                  }`}>
                    {model.description}
                  </div>
                </div>
                {isSelected && (
                  <div className={`w-2 h-2 rounded-full bg-${model.color}-500 flex-shrink-0 mt-1.5`} />
                )}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export { MODEL_OPTIONS };