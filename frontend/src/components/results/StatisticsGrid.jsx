import React from 'react';
import { Activity, AlertTriangle, FileText, Clock } from 'lucide-react';

export default function StatisticsGrid({ statistics, processingTime }) {
  if (!statistics) return null;
  
  const totalLines = statistics.total_lines || 0;
  const anomalousLines = statistics.anomalous_lines || 0;
  const normalLines = statistics.normal_lines || (totalLines - anomalousLines);
  const anomalyRate = statistics.anomaly_rate_percent || 0;
  
  return (
    <div className="grid grid-cols-2 gap-3">
      <div className="bg-slate-800/50 p-3 rounded-lg border border-purple-500/20">
        <div className="flex items-center gap-2 mb-1">
          <FileText className="w-4 h-4 text-purple-400" />
          <p className="text-purple-300 text-xs">Total Lines</p>
        </div>
        <p className="text-white font-semibold text-lg">{totalLines}</p>
      </div>
      
      <div className="bg-slate-800/50 p-3 rounded-lg border border-red-500/20">
        <div className="flex items-center gap-2 mb-1">
          <AlertTriangle className="w-4 h-4 text-red-400" />
          <p className="text-purple-300 text-xs">Anomalies</p>
        </div>
        <p className="text-white font-semibold text-lg">{anomalousLines}</p>
      </div>
      
      <div className="bg-slate-800/50 p-3 rounded-lg border border-green-500/20">
        <div className="flex items-center gap-2 mb-1">
          <Activity className="w-4 h-4 text-green-400" />
          <p className="text-purple-300 text-xs">Normal</p>
        </div>
        <p className="text-white font-semibold text-lg">{normalLines}</p>
      </div>
      
      <div className="bg-slate-800/50 p-3 rounded-lg border border-yellow-500/20">
        <div className="flex items-center gap-2 mb-1">
          <Activity className="w-4 h-4 text-yellow-400" />
          <p className="text-purple-300 text-xs">Anomaly Rate</p>
        </div>
        <p className="text-white font-semibold text-lg">{anomalyRate.toFixed(1)}%</p>
      </div>
      
      {processingTime !== undefined && (
        <div className="bg-slate-800/50 p-3 rounded-lg border border-blue-500/20 col-span-2">
          <div className="flex items-center gap-2 mb-1">
            <Clock className="w-4 h-4 text-blue-400" />
            <p className="text-purple-300 text-xs">Processing Time</p>
          </div>
          <p className="text-white font-semibold text-lg">{processingTime}s</p>
        </div>
      )}
      
      {statistics.unique_templates !== undefined && (
        <div className="bg-slate-800/50 p-3 rounded-lg border border-cyan-500/20 col-span-2">
          <div className="flex items-center gap-2 mb-1">
            <Activity className="w-4 h-4 text-cyan-400" />
            <p className="text-purple-300 text-xs">Unique Templates</p>
          </div>
          <p className="text-white font-semibold text-lg">{statistics.unique_templates}</p>
        </div>
      )}
    </div>
  );
}