import React from 'react';

export default function StatisticsGrid({ statistics, processingTime }) {
  if (!statistics) return null;
  
  return (
    <div className="grid grid-cols-2 gap-3">
      <div className="bg-slate-800/50 p-3 rounded-lg">
        <p className="text-purple-300 text-xs mb-1">Total Lines</p>
        <p className="text-white font-semibold text-lg">{statistics.total_lines}</p>
      </div>
      <div className="bg-slate-800/50 p-3 rounded-lg">
        <p className="text-purple-300 text-xs mb-1">Anomalous</p>
        <p className="text-white font-semibold text-lg">{statistics.anomalous_lines}</p>
      </div>
      <div className="bg-slate-800/50 p-3 rounded-lg">
        <p className="text-purple-300 text-xs mb-1">Anomaly Rate</p>
        <p className="text-white font-semibold text-lg">{statistics.anomaly_rate}</p>
      </div>
      <div className="bg-slate-800/50 p-3 rounded-lg">
        <p className="text-purple-300 text-xs mb-1">Processing Time</p>
        <p className="text-white font-semibold text-lg">{processingTime}s</p>
      </div>
    </div>
  );
}