import React from 'react';
import { AlertCircle } from 'lucide-react';

export default function ErrorDisplay({ error }) {
  if (!error) return null;
  
  return (
    <div className="mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg flex items-start gap-3">
      <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
      <p className="text-red-200 text-sm">{error}</p>
    </div>
  );
}