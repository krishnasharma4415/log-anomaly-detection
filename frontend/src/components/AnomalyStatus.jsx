import React from 'react';
import { AlertCircle, CheckCircle } from 'lucide-react';

export default function AnomalyStatus({ anomalyDetected, confidence }) {
  return (
    <div className={`p-4 rounded-lg border-2 ${
      anomalyDetected 
        ? 'bg-red-500/20 border-red-500' 
        : 'bg-green-500/20 border-green-500'
    }`}>
      <div className="flex items-center gap-3">
        {anomalyDetected ? (
          <AlertCircle className="w-8 h-8 text-red-400" />
        ) : (
          <CheckCircle className="w-8 h-8 text-green-400" />
        )}
        <div>
          <h3 className="text-xl font-bold">
            {anomalyDetected ? 'Anomaly Detected' : 'Normal Log'}
          </h3>
          <p className={anomalyDetected ? 'text-red-300' : 'text-green-300'}>
            Confidence: {(confidence * 100).toFixed(1)}%
          </p>
        </div>
      </div>
    </div>
  );
}