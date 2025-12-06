import { Copy, Check } from 'lucide-react';
import { useState } from 'react';

export default function LogViewer({ log, className = '' }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(log);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Highlight error keywords
  const highlightLog = (text) => {
    const keywords = ['error', 'fail', 'exception', 'critical', 'warning', 'denied', 'invalid'];
    let highlighted = text;

    keywords.forEach(keyword => {
      const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
      highlighted = highlighted.replace(
        regex,
        `<span class="text-purple-600 dark:text-purple-400 font-semibold">${keyword}</span>`
      );
    });

    return highlighted;
  };

  return (
    <div className={`relative ${className}`}>
      <div className="absolute top-2 right-2 z-10">
        <button
          onClick={handleCopy}
          className="p-2 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 hover:border-primary-500 dark:hover:border-primary-500 transition-colors"
        >
          {copied ? (
            <Check className="w-4 h-4 text-green-500" />
          ) : (
            <Copy className="w-4 h-4 text-slate-600 dark:text-slate-400" />
          )}
        </button>
      </div>
      <pre className="bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg p-4 overflow-x-auto font-mono text-sm text-slate-900 dark:text-slate-100 transition-colors duration-200">
        <code dangerouslySetInnerHTML={{ __html: highlightLog(log) }} />
      </pre>
    </div>
  );
}
