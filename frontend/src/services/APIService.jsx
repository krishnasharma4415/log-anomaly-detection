export const apiService = {
  checkHealth: async () => {
    try {
      const response = await fetch('http://localhost:5000/health');
      return response.ok ? 'healthy' : 'unhealthy';
    } catch (err) {
      return 'offline';
    }
  },

  getModelInfo: async () => {
    try {
      const response = await fetch('http://localhost:5000/model-info');
      if (!response.ok) {
        throw new Error('Failed to fetch model info');
      }
      return response.json();
    } catch (err) {
      console.error('Error fetching model info:', err);
      return null;
    }
  },

  analyzeLog: async (logText, selectedModel = null, includeTemplates = true) => {
    // Split log text into individual lines/logs
    const logs = logText
      .split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0);

    if (logs.length === 0) {
      throw new Error('No valid log lines to analyze');
    }

    const requestBody = {
      logs: logs,
      include_templates: includeTemplates,
      include_probabilities: true
    };

    // Map frontend model IDs to API parameters
    if (selectedModel) {
      if (selectedModel === 'ml') {
        requestBody.model_type = 'ml';
      } else if (selectedModel === 'dann_bert') {
        requestBody.model_type = 'bert';
        requestBody.bert_variant = 'dann';
      } else if (selectedModel === 'lora_bert') {
        requestBody.model_type = 'bert';
        requestBody.bert_variant = 'lora';
      } else if (selectedModel === 'hybrid_bert') {
        requestBody.model_type = 'bert';
        requestBody.bert_variant = 'hybrid';
      }
    }

    const response = await fetch('http://localhost:5000/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Analysis failed');
    }

    const data = await response.json();
    
    // Transform response to match frontend expectations
    return transformPredictionResponse(data);
  }
};

// Helper function to transform new API response to frontend format
function transformPredictionResponse(apiResponse) {
  const { logs, predictions, model_used, summary } = apiResponse;
  
  // Build detailed results for each log line using the new 'logs' array
  const detailed_results = logs.map((logData, index) => ({
    log_text: logData.raw,
    log_type: logData.log_type,
    parsed_content: logData.parsed_content,
    template: logData.template,
    prediction: logData.prediction.class_name,
    prediction_class_id: logData.prediction.class_index,
    confidence: logData.prediction.confidence,
    probabilities: logData.prediction.probabilities
  }));

  // Calculate overall statistics
  const anomalyCount = detailed_results.filter(r => r.prediction_class_id !== 0).length;
  const totalLogs = detailed_results.length;

  return {
    status: 'success',
    model_info: model_used,
    summary: {
      total_logs: totalLogs,
      anomaly_count: anomalyCount,
      normal_count: totalLogs - anomalyCount,
      anomaly_rate: summary?.anomaly_rate || (anomalyCount / totalLogs),
      class_distribution: summary?.class_distribution || {},
      log_type_distribution: summary?.log_type_distribution || {}
    },
    detailed_results: detailed_results,
    // For backward compatibility
    anomaly_detected: anomalyCount > 0,
    confidence: detailed_results.length > 0 ? detailed_results[0].confidence : 0,
    prediction: detailed_results.length > 0 ? detailed_results[0].prediction : 'normal'
  };
}