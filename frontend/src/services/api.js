const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiService {
  async request(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || data.detail || 'API request failed');
      }

      return data;
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  }

  // Health check
  async checkHealth() {
    return this.request('/health');
  }

  // Get model info
  async getModelInfo() {
    return this.request('/model-info');
  }

  // Predict single or multiple logs
  async predict(logs, modelType = 'ml', saveToDb = false, bertModelKey = 'best') {
    return this.request('/api/predict', {
      method: 'POST',
      body: JSON.stringify({
        logs: Array.isArray(logs) ? logs : [logs],
        model_type: modelType,
        bert_model_key: bertModelKey,
        save_to_db: saveToDb,
      }),
    });
  }

  // Analyze logs with metadata
  async analyze(logs, modelType = 'ml', saveToDb = false, bertModelKey = 'best') {
    return this.request('/api/analyze', {
      method: 'POST',
      body: JSON.stringify({
        logs: Array.isArray(logs) ? logs : [logs],
        model_type: modelType,
        bert_model_key: bertModelKey,
        save_to_db: saveToDb,
      }),
    });
  }

  // Get log entries with pagination and filters
  async getLogs(params = {}) {
    const queryString = new URLSearchParams(params).toString();
    return this.request(`/api/logs/${queryString ? '?' + queryString : ''}`);
  }

  // Get specific log entry
  async getLog(id) {
    return this.request(`/api/logs/${id}/`);
  }

  // Get predictions with pagination and filters
  async getPredictions(params = {}) {
    const queryString = new URLSearchParams(params).toString();
    return this.request(`/api/predictions/${queryString ? '?' + queryString : ''}`);
  }

  // Get specific prediction
  async getPrediction(id) {
    return this.request(`/api/predictions/${id}/`);
  }

  // Get log sources
  async getLogSources() {
    return this.request('/api/sources/');
  }

  // Get model metrics
  async getModelMetrics(params = {}) {
    const queryString = new URLSearchParams(params).toString();
    return this.request(`/api/metrics/${queryString ? '?' + queryString : ''}`);
  }

  // Get best performing models
  async getBestModels() {
    return this.request('/api/metrics/best_models/');
  }

  // Get batch analysis jobs
  async getBatchAnalyses(params = {}) {
    const queryString = new URLSearchParams(params).toString();
    return this.request(`/api/batch/${queryString ? '?' + queryString : ''}`);
  }

  // Get specific batch analysis
  async getBatchAnalysis(id) {
    return this.request(`/api/batch/${id}/`);
  }

  // Create batch analysis
  async createBatchAnalysis(data) {
    return this.request('/api/batch/', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Get statistics for dashboard
  async getStatistics() {
    try {
      const [modelInfo, health] = await Promise.all([
        this.getModelInfo(),
        this.checkHealth(),
      ]);

      return {
        totalLogs: modelInfo.statistics?.total_logs || 0,
        totalPredictions: modelInfo.statistics?.total_predictions || 0,
        anomalyRate: modelInfo.statistics?.anomaly_rate || 0,
        predictionsByModel: modelInfo.statistics?.predictions_by_model || {},
        systemHealth: health.system || {},
        modelsLoaded: health.models || {},
      };
    } catch (error) {
      console.error('Error fetching statistics:', error);
      return {
        totalLogs: 0,
        totalPredictions: 0,
        anomalyRate: 0,
        predictionsByModel: {},
        systemHealth: {},
        modelsLoaded: {},
      };
    }
  }

  // Get recent logs for dashboard
  async getRecentLogs(limit = 10) {
    try {
      const response = await this.getLogs({ page_size: limit, ordering: '-created_at' });
      return response.results || [];
    } catch (error) {
      console.error('Error fetching recent logs:', error);
      return [];
    }
  }

  // Get anomaly distribution
  async getAnomalyDistribution() {
    try {
      const predictions = await this.getPredictions({ page_size: 1000 });
      const distribution = {};
      
      (predictions.results || []).forEach(pred => {
        const className = pred.predicted_class_name;
        distribution[className] = (distribution[className] || 0) + 1;
      });

      return distribution;
    } catch (error) {
      console.error('Error fetching anomaly distribution:', error);
      return {};
    }
  }

  // Get source distribution
  async getSourceDistribution() {
    try {
      const logs = await this.getLogs({ page_size: 1000 });
      const distribution = {};
      
      (logs.results || []).forEach(log => {
        const source = log.source_name || 'Unknown';
        distribution[source] = (distribution[source] || 0) + 1;
      });

      return distribution;
    } catch (error) {
      console.error('Error fetching source distribution:', error);
      return {};
    }
  }
}

export default new ApiService();
