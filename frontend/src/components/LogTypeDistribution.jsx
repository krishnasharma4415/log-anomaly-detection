import React from 'react';
import { FileType } from 'lucide-react';

export default function LogTypeDistribution({ logTypeDistribution }) {
  if (!logTypeDistribution || Object.keys(logTypeDistribution).length === 0) {
    return null;
  }
  
  
  if (Object.keys(logTypeDistribution).length === 1 && logTypeDistribution['Unknown']) {
    return null;
  }
  
  const total = Object.values(logTypeDistribution).reduce((sum, count) => sum + count, 0);
  const sortedTypes = Object.entries(logTypeDistribution).sort((a, b) => b[1] - a[1]);
  
  
  const getLogTypeColor = (type) => {
    const colors = {
      'OpenSSH': 'bg-blue-500/20 border-blue-500/30 text-blue-300',
      'Apache': 'bg-red-500/20 border-red-500/30 text-red-300',
      'HDFS': 'bg-green-500/20 border-green-500/30 text-green-300',
      'Hadoop': 'bg-yellow-500/20 border-yellow-500/30 text-yellow-300',
      'Linux': 'bg-purple-500/20 border-purple-500/30 text-purple-300',
      'Windows': 'bg-cyan-500/20 border-cyan-500/30 text-cyan-300',
      'Spark': 'bg-orange-500/20 border-orange-500/30 text-orange-300',
      'Android': 'bg-lime-500/20 border-lime-500/30 text-lime-300',
      'BGL': 'bg-indigo-500/20 border-indigo-500/30 text-indigo-300',
      'Mac': 'bg-pink-500/20 border-pink-500/30 text-pink-300',
      'OpenStack': 'bg-teal-500/20 border-teal-500/30 text-teal-300',
      'Zookeeper': 'bg-amber-500/20 border-amber-500/30 text-amber-300',
      'Unknown': 'bg-gray-500/20 border-gray-500/30 text-gray-300'
    };
    return colors[type] || 'bg-slate-500/20 border-slate-500/30 text-slate-300';
  };
  
  return (
    <div className="bg-slate-800/50 p-4 rounded-lg border border-purple-500/20">
      <div className="flex items-center gap-2 mb-3">
        <FileType className="w-4 h-4 text-purple-400" />
        <p className="text-purple-300 text-sm font-medium">Log Type Distribution</p>
      </div>
      
      <div className="space-y-2">
        {sortedTypes.map(([type, count]) => {
          const percentage = (count / total * 100).toFixed(1);
          const colorClass = getLogTypeColor(type);
          
          return (
            <div key={type} className="flex items-center gap-3">
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <span className={`px-2 py-0.5 rounded text-xs border ${colorClass}`}>
                    {type}
                  </span>
                  <span className="text-xs text-gray-400">
                    {count} ({percentage}%)
                  </span>
                </div>
                <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${colorClass.split(' ')[0].replace('/20', '/50')} transition-all duration-300`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>
      
      {sortedTypes.length > 1 && (
        <div className="mt-3 pt-3 border-t border-gray-700">
          <p className="text-xs text-gray-400">
            Detected {sortedTypes.length} different log type{sortedTypes.length !== 1 ? 's' : ''} across {total} log{total !== 1 ? 's' : ''}
          </p>
        </div>
      )}
    </div>
  );
}
