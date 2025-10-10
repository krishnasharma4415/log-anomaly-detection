

export const ANOMALY_COLORS = {
  normal: {
    bg: 'bg-green-500/20',
    border: 'border-green-500',
    text: 'text-green-400',
    icon: 'text-green-400',
    label: 'Normal',
    emoji: '✅',
    description: 'No anomalies detected'
  },
  security_anomaly: {
    bg: 'bg-red-500/20',
    border: 'border-red-500',
    text: 'text-red-400',
    icon: 'text-red-400',
    label: 'Security Anomaly',
    emoji: '🔒',
    description: 'Security-related issue detected'
  },
  system_failure: {
    bg: 'bg-orange-500/20',
    border: 'border-orange-500',
    text: 'text-orange-400',
    icon: 'text-orange-400',
    label: 'System Failure',
    emoji: '💥',
    description: 'System crash or failure'
  },
  performance_issue: {
    bg: 'bg-yellow-500/20',
    border: 'border-yellow-500',
    text: 'text-yellow-400',
    icon: 'text-yellow-400',
    label: 'Performance Issue',
    emoji: '⚡',
    description: 'Performance degradation detected'
  },
  network_anomaly: {
    bg: 'bg-blue-500/20',
    border: 'border-blue-500',
    text: 'text-blue-400',
    icon: 'text-blue-400',
    label: 'Network Anomaly',
    emoji: '🌐',
    description: 'Network-related issue'
  },
  config_error: {
    bg: 'bg-purple-500/20',
    border: 'border-purple-500',
    text: 'text-purple-400',
    icon: 'text-purple-400',
    label: 'Config Error',
    emoji: '⚙️',
    description: 'Configuration error detected'
  },
  hardware_issue: {
    bg: 'bg-gray-500/20',
    border: 'border-gray-500',
    text: 'text-gray-400',
    icon: 'text-gray-400',
    label: 'Hardware Issue',
    emoji: '🔧',
    description: 'Hardware failure detected'
  }
};


export const getAnomalyColor = (anomalyType) => {
  return ANOMALY_COLORS[anomalyType] || ANOMALY_COLORS.normal;
};


export const getAnomalyIcon = (anomalyType) => {
  const icons = {
    normal: 'CheckCircle',
    security_anomaly: 'Shield',
    system_failure: 'XCircle',
    performance_issue: 'Zap',
    network_anomaly: 'Wifi',
    config_error: 'Settings',
    hardware_issue: 'Tool'
  };
  return icons[anomalyType] || 'AlertCircle';
};


export const formatAnomalyType = (anomalyType) => {
  if (!anomalyType) return 'Unknown';
  return ANOMALY_COLORS[anomalyType]?.label || anomalyType
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};


export const getProbabilityBarColor = (anomalyType, probability) => {
  const colors = {
    normal: 'bg-green-500',
    security_anomaly: 'bg-red-500',
    system_failure: 'bg-orange-500',
    performance_issue: 'bg-yellow-500',
    network_anomaly: 'bg-blue-500',
    config_error: 'bg-purple-500',
    hardware_issue: 'bg-gray-500'
  };
  
  
  const opacity = probability > 0.5 ? '' : '/70';
  return (colors[anomalyType] || 'bg-gray-500') + opacity;
};
