import React, { useState, useEffect } from 'react';
import { FileText, TrendingUp } from 'lucide-react';
import { apiService } from '../services/apiService';
import Header from './Header';
import FileUpload from './FileUpload';
import LogTextInput from './LogTextInput';
import ActionButtons from './ActionButtons';
import ErrorDisplay from './ErrorDisplay';
import EmptyState from './EmptyState';
import LoadingState from './LoadingState';
import ResultsDisplay from './ResultsDisplay';
import LogSourcesFooter from './LogSourcesFooter';

export default function LogAnomalyDetector() {
  const [logInput, setLogInput] = useState('');
  const [file, setFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState('unknown');

  useEffect(() => {
    apiService.checkHealth().then(setApiStatus);
  }, []);

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

  const analyzeLog = async () => {
    if (!logInput.trim()) {
      setError('Please enter log content or upload a log file.');
      return;
    }

    setAnalyzing(true);
    setError('');
    setResults(null);

    try {
      const data = await apiService.analyzeLog(logInput);
      setResults({
        anomaly_detected: data.anomaly_detected,
        confidence: data.confidence,
        predicted_source: data.predicted_source,
        template: data.template,
        embedding_dims: data.embedding_dims,
        processing_time: data.processing_time,
        statistics: data.statistics || {},
        model_used: data.model_used || 'Unknown',
        detailed_results: data.detailed_results || []
      });
    } catch (err) {
      setError(`Analysis failed: ${err.message}. Please ensure the Flask backend is running on port 5000.`);
      setApiStatus('offline');
    } finally {
      setAnalyzing(false);
    }
  };

  const clearAll = () => {
    setLogInput('');
    setFile(null);
    setResults(null);
    setError('');
  };

  const loadSampleLog = () => {
    const sampleLog = `2025-01-15 14:32:15 ERROR Connection timeout to server 192.168.1.100
2025-01-15 14:32:20 WARN Retrying connection attempt 3/5
2025-01-15 14:32:25 ERROR Failed to establish connection after 5 attempts
2025-01-15 14:32:30 INFO Switching to backup server 192.168.1.101
2025-01-15 14:32:35 INFO Connection established successfully
2025-01-15 14:32:40 DEBUG Processing request queue (125 pending)
2025-01-15 14:32:45 INFO Request processed successfully
2025-01-15 14:32:50 ERROR Unexpected null pointer exception in module auth.service
2025-01-15 14:32:55 CRITICAL System memory usage at 95%
2025-01-15 14:33:00 WARN High CPU usage detected (88%)`;
    setLogInput(sampleLog);
    setError('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-4 sm:p-6 font-sans">
      <div className="max-w-7xl mx-auto">
        <Header apiStatus={apiStatus} />

        <div className="grid md:grid-cols-2 gap-6">
          {/* Input Section */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
            <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
              <FileText className="w-6 h-6" />
              Input Log Data
            </h2>

            <FileUpload 
              onFileSelect={handleFileUpload}
              fileName={file?.name}
            />

            <div className="text-center text-purple-300 text-sm my-3">OR</div>

            <LogTextInput 
              value={logInput}
              onChange={(e) => setLogInput(e.target.value)}
            />

            <ActionButtons
              onAnalyze={analyzeLog}
              onClear={clearAll}
              onLoadSample={loadSampleLog}
              analyzing={analyzing}
              hasInput={logInput.trim()}
            />

            <ErrorDisplay error={error} />
          </div>

          {/* Results Section */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
            <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
              <TrendingUp className="w-6 h-6" />
              Analysis Results
            </h2>

            {!results && !analyzing && <EmptyState />}
            {analyzing && <LoadingState />}
            {results && <ResultsDisplay results={results} />}
          </div>
        </div>

        <LogSourcesFooter />
      </div>
    </div>
  );
}