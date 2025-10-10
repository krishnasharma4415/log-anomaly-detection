import React from 'react';
import { Database, TrendingUp, Cpu, Activity } from 'lucide-react';

export default function Header({ apiStatus }) {
  return (
    <div className="text-center mb-8">
      <div className="flex items-center justify-center gap-3 mb-4">
        <Cpu className="w-10 h-10 sm:w-12 sm:h-12 text-purple-400" />
        <h1 className="text-3xl sm:text-4xl font-bold">
          Log Anomaly Detection System
        </h1>
      </div>
      <p className="text-purple-200 text-md sm:text-lg">
        Cross-Source Transfer Learning for Rare Anomaly Detection
      </p>
      <div className="flex items-center justify-center gap-4 mt-4 text-sm text-purple-300">
        <span className="flex items-center gap-1.5">
          <Database className="w-4 h-4" />
          16 Log Sources
        </span>
        <span className="flex items-center gap-1.5">
          <TrendingUp className="w-4 h-4" />
          ML + BERT Model
        </span>
        <span className={`flex items-center gap-1.5 ${
          apiStatus === 'healthy' ? 'text-green-400' : 
          apiStatus === 'offline' ? 'text-red-400' : 'text-yellow-400'
        }`}>
          <Activity className="w-4 h-4" />
          API: {apiStatus}
        </span>
      </div>
    </div>
  );
}