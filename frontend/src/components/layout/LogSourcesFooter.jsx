import React from 'react';

export default function LogSourcesFooter() {
  const logSources = [
    'Windows', 'Linux', 'Mac', 'Hadoop', 'HDFS', 'Zookeeper', 
    'Spark', 'Apache', 'Thunderbird', 'Proxifier', 'HealthApp',
    'OpenStack', 'OpenSSH', 'BGL', 'HPC', 'Android'
  ];

  return (
    <div className="mt-8 bg-white/5 backdrop-blur-lg rounded-xl p-6 border border-white/10">
      <h3 className="font-semibold mb-3">Supported Log Sources</h3>
      <div className="flex flex-wrap gap-2">
        {logSources.map((source) => (
          <span
            key={source}
            className="px-3 py-1 bg-purple-600/30 text-purple-200 rounded-full text-sm border border-purple-500/30"
          >
            {source}
          </span>
        ))}
      </div>
    </div>
  );
}