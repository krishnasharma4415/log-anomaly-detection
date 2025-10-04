import React from 'react';
import { Loader2 } from 'lucide-react';

export default function LoadingState() {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="text-center">
        <Loader2 className="w-16 h-16 mx-auto mb-4 text-purple-400 animate-spin" />
        <p className="text-purple-300">Processing log data...</p>
        <p className="text-purple-400 text-sm mt-2">Extracting features and computing embeddings</p>
      </div>
    </div>
  );
}