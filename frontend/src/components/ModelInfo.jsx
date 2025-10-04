import React from 'react';

export default function ModelInfo({ modelUsed }) {
  return (
    <div className="mt-6 p-4 bg-purple-900/30 rounded-lg border border-purple-500/30">
      <p className="text-purple-200 text-sm">
        <span className="font-semibold">Model:</span> {modelUsed} with BERT Embeddings
      </p>
      <p className="text-purple-300 text-xs mt-1">
        Trained on diverse log sources using cross-source transfer learning.
      </p>
    </div>
  );
}