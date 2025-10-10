import React from 'react';
import { Database } from 'lucide-react';

export default function EmptyState() {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="text-center text-purple-300">
        <Database className="w-16 h-16 mx-auto mb-4 opacity-50" />
        <p>Upload or paste a log to begin analysis</p>
        <p className="text-sm mt-2 text-purple-400">Supported formats: .log, .txt</p>
      </div>
    </div>
  );
}