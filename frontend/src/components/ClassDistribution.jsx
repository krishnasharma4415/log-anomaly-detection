import React from 'react';
import { BarChart3 } from 'lucide-react';
import { getAnomalyColor, formatAnomalyType } from '../utils/anomalyColors';

export default function ClassDistribution({ classDistribution, totalLines }) {
  if (!classDistribution) return null;

  
  const sortedClasses = Object.entries(classDistribution)
    .sort((a, b) => b[1] - a[1]);

  return (
    <div className="bg-slate-800/50 p-4 rounded-lg">
      <p className="text-purple-300 text-sm mb-3 flex items-center gap-2">
        <BarChart3 className="w-4 h-4" />
        Class Distribution ({totalLines} total logs)
      </p>
      <div className="space-y-2">
        {sortedClasses.map(([className, count]) => {
          const percentage = totalLines > 0 ? (count / totalLines) * 100 : 0;
          const colorScheme = getAnomalyColor(className);

          return (
            <div key={className} className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className={`flex items-center gap-1 ${colorScheme.text}`}>
                  <span className="text-lg">{colorScheme.emoji}</span>
                  {formatAnomalyType(className)}
                </span>
                <span className="text-gray-300">
                  {count} ({percentage.toFixed(1)}%)
                </span>
              </div>
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-500 ${colorScheme.border.replace('border-', 'bg-')}`}
                  style={{ width: `${percentage}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
