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
          className="p-2 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 hover:border-primary-500 dark:hover:border-primary-500 transition-colors"
        >
          {copied ? (
            <Check className="w-4 h-4 text-green-500" />
          ) : (
            <Copy className="w-4 h-4 text-slate-600 dark:text-slate-400" />
          )}
        </button>
      </div>
      <div className="bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg p-4 overflow-x-auto font-mono text-sm transition-colors duration-200">
        <JsonNode data={data} level={0} />
      </div>
    </div>
  );
}

function JsonNode({ data, level }) {
  const [isExpanded, setIsExpanded] = useState(level < 2);

  if (data === null) {
    return <span className="text-slate-400 dark:text-slate-500">null</span>;
  }

  if (typeof data === 'boolean') {
    return <span className="text-purple-500 dark:text-purple-400">{data.toString()}</span>;
  }

  if (typeof data === 'number') {
    return <span className="text-cyan-500 dark:text-cyan-400">{data}</span>;
  }

  if (typeof data === 'string') {
    return <span className="text-green-500 dark:text-green-400">"{data}"</span>;
  }

  if (Array.isArray(data)) {
    return (
      <div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="inline-flex items-center gap-1 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 transition-colors"
        >
          {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          <span className="text-slate-400 dark:text-slate-500">[{data.length}]</span>
        </button>
        {isExpanded && (
          <div className="ml-4 border-l border-slate-200 dark:border-slate-700 pl-4 mt-1">
            {data.map((item, index) => (
              <div key={index} className="py-1">
                <span className="text-slate-400 dark:text-slate-500">{index}: </span>
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
          className="inline-flex items-center gap-1 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 transition-colors"
        >
          {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          <span className="text-slate-400 dark:text-slate-500">{'{'}{keys.length}{'}'}</span>
        </button>
        {isExpanded && (
          <div className="ml-4 border-l border-slate-200 dark:border-slate-700 pl-4 mt-1">
            {keys.map((key) => (
              <div key={key} className="py-1">
                <span className="text-primary-600 dark:text-primary-400">"{key}"</span>
                <span className="text-slate-400 dark:text-slate-500">: </span>
                <JsonNode data={data[key]} level={level + 1} />
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  return <span className="text-slate-900 dark:text-slate-100">{String(data)}</span>;
}
