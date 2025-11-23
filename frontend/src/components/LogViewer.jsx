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
        `<span class="text-accent-purple font-semibold">${keyword}</span>`
      );
    });
    
    return highlighted;
  };

  return (
    <div className={`relative ${className}`}>
      <div className="absolute top-2 right-2 z-10">
        <button
          onClick={handleCopy}
          className="p-2 bg-neutral-surface rounded-lg border border-neutral-border hover:border-primary transition-colors"
        >
          {copied ? (
            <Check className="w-4 h-4 text-signal-success" />
          ) : (
            <Copy className="w-4 h-4 text-neutral-secondary" />
          )}
        </button>
      </div>
      <pre className="bg-neutral-dark border border-neutral-border rounded-lg p-4 overflow-x-auto font-mono text-sm text-neutral-primary">
        <code dangerouslySetInnerHTML={{ __html: highlightLog(log) }} />
      </pre>
    </div>
  );
}
