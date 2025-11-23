import { useState } from 'react';
import { ChevronRight, ChevronDown, Copy, Check } from 'lucide-react';

export default function JsonViewer({ data, className = '' }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
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
      <div className="bg-neutral-dark border border-neutral-border rounded-lg p-4 overflow-x-auto font-mono text-sm">
        <JsonNode data={data} level={0} />
      </div>
    </div>
  );
}

function JsonNode({ data, level }) {
  const [isExpanded, setIsExpanded] = useState(level < 2);

  if (data === null) {
    return <span className="text-neutral-disabled">null</span>;
  }

  if (typeof data === 'boolean') {
    return <span className="text-accent-purple">{data.toString()}</span>;
  }

  if (typeof data === 'number') {
    return <span className="text-accent-cyan">{data}</span>;
  }

  if (typeof data === 'string') {
    return <span className="text-signal-success">"{data}"</span>;
  }

  if (Array.isArray(data)) {
    return (
      <div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="inline-flex items-center gap-1 text-neutral-secondary hover:text-neutral-primary"
        >
          {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          <span className="text-neutral-disabled">[{data.length}]</span>
        </button>
        {isExpanded && (
          <div className="ml-4 border-l border-neutral-border pl-4 mt-1">
            {data.map((item, index) => (
              <div key={index} className="py-1">
                <span className="text-neutral-disabled">{index}: </span>
                <JsonNode data={item} level={level + 1} />
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  if (typeof data === 'object') {
    const keys = Object.keys(data);
    return (
      <div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="inline-flex items-center gap-1 text-neutral-secondary hover:text-neutral-primary"
        >
          {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          <span className="text-neutral-disabled">{'{'}{keys.length}{'}'}</span>
        </button>
        {isExpanded && (
          <div className="ml-4 border-l border-neutral-border pl-4 mt-1">
            {keys.map((key) => (
              <div key={key} className="py-1">
                <span className="text-primary">"{key}"</span>
                <span className="text-neutral-disabled">: </span>
                <JsonNode data={data[key]} level={level + 1} />
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  return <span className="text-neutral-primary">{String(data)}</span>;
}
