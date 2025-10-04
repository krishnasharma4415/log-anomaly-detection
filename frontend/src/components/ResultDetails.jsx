import React from 'react';

export default function ResultDetails({ predictedSource, template, embeddingDims, modelUsed }) {
  return (
    <div className="space-y-3">
      <div className="bg-slate-800/50 p-4 rounded-lg">
        <p className="text-purple-300 text-sm mb-1">Predicted Source</p>
        <p className="text-white font-semibold text-lg">{predictedSource}</p>
      </div>

      <div className="bg-slate-800/50 p-4 rounded-lg">
        <p className="text-purple-300 text-sm mb-1">Log Template</p>
        <p className="text-white font-mono text-xs break-all">{template}</p>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-slate-800/50 p-4 rounded-lg">
          <p className="text-purple-300 text-sm mb-1">Embedding Dims</p>
          <p className="text-white font-semibold">{embeddingDims}</p>
        </div>
        <div className="bg-slate-800/50 p-4 rounded-lg">
          <p className="text-purple-300 text-sm mb-1">Model Used</p>
          <p className="text-white font-semibold text-sm">{modelUsed}</p>
        </div>
      </div>
    </div>
  );
}
