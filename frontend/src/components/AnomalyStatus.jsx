import React from 'react';
import { AlertCircle, CheckCircle, Shield, XCircle, Zap, Wifi, Settings, Wrench } from 'lucide-react';
import { getAnomalyColor, formatAnomalyType } from '../utils/anomalyColors';

export default function AnomalyStatus({ anomalyDetected, confidence, prediction, predictionClassId }) {
  // Determine the anomaly type (default to 'normal' if not anomalous)
  const anomalyType = prediction || (anomalyDetected ? 'security_anomaly' : 'normal');
  const colorScheme = getAnomalyColor(anomalyType);
  
  // Select icon based on anomaly type
  const IconComponent = {
    normal: CheckCircle,
    security_anomaly: Shield,
    system_failure: XCircle,
    performance_issue: Zap,
    network_anomaly: Wifi,
    config_error: Settings,
    hardware_issue: Wrench
  }[anomalyType] || AlertCircle;
  
  return (
    <div className={`p-4 rounded-lg border-2 ${colorScheme.bg} ${colorScheme.border}`}>
      <div className="flex items-center gap-3">
        <IconComponent className={`w-8 h-8 ${colorScheme.icon}`} />
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h3 className="text-xl font-bold">
              {formatAnomalyType(anomalyType)}
            </h3>
            <span className="text-2xl">{colorScheme.emoji}</span>
          </div>
          <p className={colorScheme.text}>
            Confidence: {(confidence * 100).toFixed(1)}%
          </p>
          <p className="text-gray-300 text-xs mt-1">{colorScheme.description}</p>
          {predictionClassId !== undefined && (
            <p className="text-gray-400 text-xs mt-1">Class ID: {predictionClassId}</p>
          )}
        </div>
      </div>
    </div>
  );
}