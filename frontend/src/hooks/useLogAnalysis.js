import { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import { SAMPLE_LOG } from '../utils/constants';

export function useLogAnalysis() {
  const [logInput, setLogInput] = useState('');
  const [file, setFile] = useState(null);
  const [selectedModel, setSelectedModel] = useState('ml');
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState('unknown');

  // Check API health on mount
  useEffect(() => {
    apiService.checkHealth().then(setApiStatus);
  }, []);

  // Handle file upload
  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;

    setFile(uploadedFile);
    setError('');

    const reader = new FileReader();
    reader.onload = (event) => setLogInput(event.target.result);
    reader.onerror = () => setError('Failed to read file');
    reader.readAsText(uploadedFile);
  };

  // Handle model change
  const handleModelChange = (modelId) => {
    setSelectedModel(modelId);
    setError('');
  };

  // Process API response into results
  const processResults = (data) => {
    let primaryPrediction = 'normal';
    let primaryPredictionClassId = 0;

    if (data.detailed_results && data.detailed_results.length > 0) {
      // Find first anomaly or use first result
      const anomaly = data.detailed_results.find(r => r.is_anomaly);
      if (anomaly) {
        primaryPrediction = anomaly.prediction;
        primaryPredictionClassId = anomaly.prediction_class_id;
      } else {
        primaryPrediction = data.detailed_results[0].prediction;
        primaryPredictionClassId = data.detailed_results[0].prediction_class_id || 0;
      }
    }

    return {
      anomaly_detected: data.anomaly_detected,
      confidence: data.confidence,
      primary_prediction: primaryPrediction,
      primary_prediction_class_id: primaryPredictionClassId,
      predicted_source: data.predicted_source,
      template: data.template,
      embedding_dims: data.embedding_dims,
      processing_time_seconds: data.processing_time_seconds,
      statistics: {
        total_lines: data.summary?.total_logs || 0,
        anomalous_lines: data.summary?.anomaly_count || 0,
        normal_lines: data.summary?.normal_count || 0,
        anomaly_rate_percent: (data.summary?.anomaly_rate || 0) * 100,
        class_distribution: data.summary?.class_distribution || {},
        log_type_distribution: data.summary?.log_type_distribution || {}
      },
      model_info: data.model_info || {},
      detailed_results: data.detailed_results || []
    };
  };

  // Analyze log
  const analyzeLog = async () => {
    if (!logInput.trim()) {
      setError('Please enter log content or upload a log file.');
      return;
    }

    setAnalyzing(true);
    setError('');
    setResults(null);

    try {
      const data = await apiService.analyzeLog(logInput, selectedModel);
      setResults(processResults(data));
    } catch (err) {
      setError(`Analysis failed: ${err.message}. Please ensure the Flask backend is running on port 5000.`);
      setApiStatus('offline');
    } finally {
      setAnalyzing(false);
    }
  };

  // Clear all
  const clearAll = () => {
    setLogInput('');
    setFile(null);
    setResults(null);
    setError('');
  };

  // Load sample log
  const loadSampleLog = () => {
    setLogInput(SAMPLE_LOG);
    setError('');
  };

  return {
    // State
    logInput,
    file,
    selectedModel,
    analyzing,
    results,
    error,
    apiStatus,
    
    // Actions
    setLogInput,
    handleFileUpload,
    handleModelChange,
    analyzeLog,
    clearAll,
    loadSampleLog
  };
}