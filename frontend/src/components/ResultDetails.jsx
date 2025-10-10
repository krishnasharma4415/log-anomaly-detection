import React from 'react';
import { Database, FileCode, Cpu } from 'lucide-react';

export default function ResultDetails({ predictedSource, template, embeddingDims, modelInfo }) {
  const modelType = modelInfo?.model_type || 'Unknown';
  
  return (
    <div className="space-y-3">
      {predictedSource && (
        <div className="bg-slate-800/50 p-4 rounded-lg border border-purple-500/20">
          <div className="flex items-center gap-2 mb-1">
            <Database className="w-4 h-4 text-purple-400" />
            <p className="text-purple-300 text-sm">Predicted Source</p>
          </div>
          <p className="text-white font-semibold text-lg">{predictedSource}</p>
        </div>
      )}

      {template && (
        <div className="bg-slate-800/50 p-4 rounded-lg border border-cyan-500/20">
          <div className="flex items-center gap-2 mb-1">
            <FileCode className="w-4 h-4 text-cyan-400" />
            <p className="text-purple-300 text-sm">Log Template</p>
          </div>
          <p className="text-white font-mono text-xs break-all">{template}</p>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3">
        {embeddingDims && (
          <div className="bg-slate-800/50 p-4 rounded-lg border border-blue-500/20">
            <div className="flex items-center gap-2 mb-1">
              <Cpu className="w-4 h-4 text-blue-400" />
              <p className="text-purple-300 text-sm">Embedding Dims</p>
            </div>
            <p className="text-white font-semibold">{embeddingDims}</p>
          </div>
        )}
        <div className="bg-slate-800/50 p-4 rounded-lg border border-green-500/20">
          <div className="flex items-center gap-2 mb-1">
            <Cpu className="w-4 h-4 text-green-400" />
            <p className="text-purple-300 text-sm">Model Type</p>
          </div>
          <p className="text-white font-semibold text-sm">{modelType}</p>
        </div>
      </div>
    </div>
  );
}
