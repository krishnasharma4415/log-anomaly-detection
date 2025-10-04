import React from 'react';
import { Loader2, Cpu } from 'lucide-react';

export default function ActionButtons({ onAnalyze, onClear, onLoadSample, analyzing, hasInput }) {
  return (
    <>
      <div className="flex gap-3 mb-4">
        <button
          onClick={onAnalyze}
          disabled={analyzing || !hasInput}
          className="flex-1 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold rounded-lg transition-all transform hover:scale-105 disabled:scale-100 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {analyzing ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Cpu className="w-5 h-5" />
              Analyze Log
            </>
          )}
        </button>
        <button
          onClick={onClear}
          disabled={analyzing}
          className="px-6 py-3 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 text-white font-semibold rounded-lg transition-colors"
        >
          Clear
        </button>
      </div>
      <button
        onClick={onLoadSample}
        disabled={analyzing}
        className="w-full px-4 py-2 bg-slate-800/50 hover:bg-slate-700/50 text-purple-300 text-sm rounded-lg transition-colors border border-purple-500/30"
      >
        Load Sample Log
      </button>
    </>
  );
}