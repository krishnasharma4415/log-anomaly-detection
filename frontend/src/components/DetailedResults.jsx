import React from 'react';
import { Info } from 'lucide-react';

export default function DetailedResults({ detailedResults }) {
  if (!detailedResults || detailedResults.length === 0) return null;
  
  return (
    <div className="bg-slate-800/50 p-4 rounded-lg">
      <p className="text-purple-300 text-sm mb-3 flex items-center gap-2">
        <Info className="w-4 h-4" />
        Line-by-Line Analysis (First 10)
      </p>
      <div className="space-y-2 max-h-64 overflow-y-auto">
        {detailedResults.slice(0, 10).map((line, idx) => (
          <div 
            key={idx}
            className={`p-2 rounded text-xs ${
              line.is_anomaly 
                ? 'bg-red-900/30 border-l-2 border-red-500' 
                : 'bg-green-900/20 border-l-2 border-green-500'
            }`}
          >
            <div className="flex justify-between mb-1">
              <span className="text-purple-300">Line {line.line_number}</span>
              <span className={line.is_anomaly ? 'text-red-400' : 'text-green-400'}>
                {line.prediction} ({(line.confidence * 100).toFixed(0)}%)
              </span>
            </div>
            <p className="text-gray-300 font-mono truncate">{line.content}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
