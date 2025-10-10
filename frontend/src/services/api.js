import { API_BASE_URL, API_ENDPOINTS } from '../utils/constants';

// Helper function to build full URL
const buildUrl = (endpoint) => `${API_BASE_URL}${endpoint}`;

// Helper function to map model selection to API parameters
const getModelParams = (selectedModel) => {
  const modelMap = {
    'ml': { model_type: 'ml' },
    'dann_bert': { model_type: 'bert', bert_variant: 'dann' },
    'lora_bert': { model_type: 'bert', bert_variant: 'lora' },
    'hybrid_bert': { model_type: 'bert', bert_variant: 'hybrid' }
  };
  return modelMap[selectedModel] || {};
};

// Transform backend response to frontend format
function transformPredictionResponse(apiResponse) {
  const { logs, model_used, summary } = apiResponse;

  // Map detailed results
  const detailed_results = logs.map((logData) => ({
    log_text: logData.raw,
    log_type: logData.log_type,
    parsed_content: logData.parsed_content,
    template: logData.template,
    prediction: logData.prediction.class_name,
    prediction_class_id: logData.prediction.class_index,
    is_anomaly: logData.prediction.class_index !== 0,
    confidence: logData.prediction.confidence,
    probabilities: logData.prediction.probabilities
  }));

  // Calculate summary statistics
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
    
    // Backward compatibility
    anomaly_detected: anomalyCount > 0,
    confidence: detailed_results.length > 0 ? detailed_results[0].confidence : 0,
    prediction: detailed_results.length > 0 ? detailed_results[0].prediction : 'normal'
  };
}

// API Service
export const apiService = {
  /**
   * Check API health status
   * @returns {Promise<string>} 'healthy', 'unhealthy', or 'offline'
   */
  checkHealth: async () => {
    try {
      const response = await fetch(buildUrl(API_ENDPOINTS.HEALTH));
      return response.ok ? 'healthy' : 'unhealthy';
    } catch (err) {
      return 'offline';
    }
  },

  /**
   * Get model information
   * @returns {Promise<object|null>} Model metadata or null on error
   */
  getModelInfo: async () => {
    try {
      const response = await fetch(buildUrl(API_ENDPOINTS.MODEL_INFO));
      if (!response.ok) {
        throw new Error('Failed to fetch model info');
      }
      return response.json();
    } catch (err) {
      console.error('Error fetching model info:', err);
      return null;
    }
  },

  /**
   * Analyze log data
   * @param {string} logText - Raw log text
   * @param {string} selectedModel - Model ID ('ml', 'dann_bert', 'lora_bert', 'hybrid_bert')
   * @param {boolean} includeTemplates - Include template extraction
   * @returns {Promise<object>} Analysis results
   */
  analyzeLog: async (logText, selectedModel = null, includeTemplates = true) => {
    // Split and clean logs
    const logs = logText
      .split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0);

    if (logs.length === 0) {
      throw new Error('No valid log lines to analyze');
    }

    // Build request body
    const requestBody = {
      logs: logs,
      include_templates: includeTemplates,
      include_probabilities: true,
      ...getModelParams(selectedModel)
    };

    // Make API request
    const response = await fetch(buildUrl(API_ENDPOINTS.PREDICT), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Analysis failed');
    }

    const data = await response.json();
    return transformPredictionResponse(data);
  }
};