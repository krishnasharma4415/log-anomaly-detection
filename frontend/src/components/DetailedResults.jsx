import React, { useState } from 'react';
import { Info, ChevronDown, ChevronUp, FileType } from 'lucide-react';
import { getAnomalyColor, formatAnomalyType, getProbabilityBarColor } from '../utils/anomalyColors';

export default function DetailedResults({ detailedResults }) {
  const [expandedLines, setExpandedLines] = useState(new Set());
  
  if (!detailedResults || detailedResults.length === 0) return null;
  
  const toggleExpanded = (index) => {
    const newExpanded = new Set(expandedLines);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedLines(newExpanded);
  };
  
  
  const logTypeDistribution = detailedResults.reduce((acc, line) => {
    const type = line.log_type || 'Unknown';
    acc[type] = (acc[type] || 0) + 1;
    return acc;
  }, {});
  
  const hasLogTypes = Object.keys(logTypeDistribution).length > 0 && 
                      !(Object.keys(logTypeDistribution).length === 1 && logTypeDistribution['Unknown']);
  
  return (
    <div className="bg-slate-800/50 p-4 rounded-lg">
      <div className="flex items-center justify-between mb-3">
        <p className="text-purple-300 text-sm flex items-center gap-2">
          <Info className="w-4 h-4" />
          Line-by-Line Analysis (First 10)
        </p>
        {hasLogTypes && (
          <div className="flex items-center gap-2 text-xs">
            <FileType className="w-3 h-3 text-blue-300" />
            <span className="text-gray-400">Log Types:</span>
            {Object.entries(logTypeDistribution).map(([type, count]) => (
              <span key={type} className="px-2 py-0.5 bg-blue-500/10 text-blue-300 rounded border border-blue-500/20">
                {type} ({count})
              </span>
            ))}
          </div>
        )}
      </div>
      <div className="space-y-2 max-h-[500px] overflow-y-auto">
        {detailedResults.slice(0, 10).map((line, idx) => {
          const colorScheme = getAnomalyColor(line.prediction);
          const isExpanded = expandedLines.has(idx);
          
          return (
            <div 
              key={idx}
              className={`p-2 rounded text-xs ${colorScheme.bg} border-l-2 ${colorScheme.border}`}
            >
              <div className="flex justify-between mb-1">
                <div className="flex items-center gap-2">
                  <span className="text-purple-300">Line {idx + 1}</span>
                  {line.log_type && (
                    <span className="px-2 py-0.5 bg-blue-500/20 text-blue-300 rounded text-xs border border-blue-500/30">
                      {line.log_type}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <span className={colorScheme.text}>
                    {formatAnomalyType(line.prediction)} ({(line.confidence * 100).toFixed(0)}%)
                  </span>
                  <button
                    onClick={() => toggleExpanded(idx)}
                    className="text-gray-400 hover:text-white transition-colors"
                    aria-label="Toggle probabilities"
                  >
                    {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>
                </div>
              </div>
              
              {}
              {line.parsed_content ? (
                <>
                  <p className="text-gray-400 text-xs mb-1">Parsed Content:</p>
                  <p className="text-gray-300 font-mono mb-2 break-words">{line.parsed_content}</p>
                  {isExpanded && line.log_text && (
                    <details className="mb-2">
                      <summary className="text-gray-500 text-xs cursor-pointer hover:text-gray-400">
                        View raw log
                      </summary>
                      <p className="text-gray-400 font-mono text-xs mt-1 break-words">{line.log_text}</p>
                    </details>
                  )}
                </>
              ) : (
                <p className="text-gray-300 font-mono mb-2 break-words">{line.log_text || line.content}</p>
              )}
              
              {}
              {isExpanded && line.probabilities && (
                <div className="mt-2 pt-2 border-t border-gray-700">
                  <p className="text-gray-400 text-xs mb-2">Class Probabilities:</p>
                  <div className="space-y-1">
                    {Object.entries(line.probabilities)
                      .sort((a, b) => b[1] - a[1]) 
                      .map(([className, prob]) => (
                        <div key={className} className="flex items-center gap-2">
                          <div className="w-32 text-xs text-gray-300 truncate">
                            {formatAnomalyType(className)}
                          </div>
                          <div className="flex-1 h-4 bg-gray-700 rounded-full overflow-hidden">
                            <div
                              className={`h-full transition-all ${getProbabilityBarColor(className, prob)}`}
                              style={{ width: `${prob * 100}%` }}
                            />
                          </div>
                          <div className="w-12 text-xs text-gray-400 text-right">
                            {(prob * 100).toFixed(1)}%
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}
              
              {}
              {line.template && (
                <div className="mt-2 pt-2 border-t border-gray-700">
                  <p className="text-gray-400 text-xs">Template: {line.template}</p>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
