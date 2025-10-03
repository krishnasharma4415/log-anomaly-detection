import React, { useState } from 'react';
import { Upload, FileText, AlertCircle, CheckCircle, Loader2, TrendingUp, Database, Cpu, Info, Activity } from 'lucide-react';

export default function LogAnomalyDetector() {
  const [logInput, setLogInput] = useState('');
  const [file, setFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState('unknown');

  const logSources = [
    'Windows', 'Linux', 'Mac', 'Hadoop', 'HDFS', 'Zookeeper', 
    'Spark', 'Apache', 'Thunderbird', 'Proxifier', 'HealthApp',
    'OpenStack', 'OpenSSH', 'BGL', 'HPC', 'Android'
  ];

  // Check API health on mount
  React.useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await fetch('http://localhost:5000/health');
      if (response.ok) {
        setApiStatus('healthy');
      } else {
        setApiStatus('unhealthy');
      }
    } catch (err) {
      setApiStatus('offline');
    }
  };

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      setError('');
      
      const reader = new FileReader();
      reader.onload = (event) => {
        setLogInput(event.target.result);
      };
      reader.onerror = () => {
        setError('Failed to read file');
      };
      reader.readAsText(uploadedFile);
    }
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
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          log_text: logInput,
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Analysis failed');
      }

      const data = await response.json();
      
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
      console.error(err);
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
        {/* Header */}
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

        {/* Main Content */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Input Section */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
            <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
              <FileText className="w-6 h-6" />
              Input Log Data
            </h2>

            {/* File Upload */}
            <div className="mb-4">
              <label className="block text-purple-200 mb-2 text-sm font-medium">
                Upload Log File
              </label>
              <div className="relative">
                <input
                  type="file"
                  accept=".log,.txt"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="flex items-center justify-center gap-2 w-full px-4 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg cursor-pointer transition-colors"
                >
                  <Upload className="w-5 h-5" />
                  {file ? file.name : 'Choose File'}
                </label>
              </div>
            </div>

            <div className="text-center text-purple-300 text-sm my-3">OR</div>

            {/* Text Input */}
            <div className="mb-4">
              <label className="block text-purple-200 mb-2 text-sm font-medium">
                Paste Log Content
              </label>
              <textarea
                value={logInput}
                onChange={(e) => setLogInput(e.target.value)}
                placeholder="Enter system log here..."
                className="w-full h-48 px-4 py-3 bg-slate-800/50 text-white rounded-lg border border-purple-500/30 focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 outline-none resize-none font-mono text-sm"
              />
            </div>

            {/* Action Buttons */}
            <div className="flex gap-3 mb-4">
              <button
                onClick={analyzeLog}
                disabled={analyzing || !logInput.trim()}
                className="flex-1 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold rounded-lg transition-all transform hover:scale-105 disabled:scale-100 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {analyzing ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Cpu className="w-5 h-5" />
                    Analyze Log
                  </>
                )}
              </button>
              <button
                onClick={clearAll}
                disabled={analyzing}
                className="px-6 py-3 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 text-white font-semibold rounded-lg transition-colors"
              >
                Clear
              </button>
            </div>

            <button
              onClick={loadSampleLog}
              disabled={analyzing}
              className="w-full px-4 py-2 bg-slate-800/50 hover:bg-slate-700/50 text-purple-300 text-sm rounded-lg transition-colors border border-purple-500/30"
            >
              Load Sample Log
            </button>

            {error && (
              <div className="mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <p className="text-red-200 text-sm">{error}</p>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
            <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
              <TrendingUp className="w-6 h-6" />
              Analysis Results
            </h2>

            {!results && !analyzing && (
              <div className="h-full flex items-center justify-center">
                <div className="text-center text-purple-300">
                  <Database className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p>Upload or paste a log to begin analysis</p>
                  <p className="text-sm mt-2 text-purple-400">Supported formats: .log, .txt</p>
                </div>
              </div>
            )}

            {analyzing && (
              <div className="h-full flex items-center justify-center">
                <div className="text-center">
                  <Loader2 className="w-16 h-16 mx-auto mb-4 text-purple-400 animate-spin" />
                  <p className="text-purple-300">Processing log data...</p>
                  <p className="text-purple-400 text-sm mt-2">Extracting features and computing embeddings</p>
                </div>
              </div>
            )}

            {results && (
              <div className="space-y-4 max-h-[600px] overflow-y-auto">
                {/* Anomaly Status */}
                <div className={`p-4 rounded-lg border-2 ${
                  results.anomaly_detected 
                    ? 'bg-red-500/20 border-red-500' 
                    : 'bg-green-500/20 border-green-500'
                }`}>
                  <div className="flex items-center gap-3">
                    {results.anomaly_detected ? (
                      <AlertCircle className="w-8 h-8 text-red-400" />
                    ) : (
                      <CheckCircle className="w-8 h-8 text-green-400" />
                    )}
                    <div>
                      <h3 className="text-xl font-bold">
                        {results.anomaly_detected ? 'Anomaly Detected' : 'Normal Log'}
                      </h3>
                      <p className={results.anomaly_detected ? 'text-red-300' : 'text-green-300'}>
                        Confidence: {(results.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </div>

                {/* Statistics */}
                {results.statistics && (
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-slate-800/50 p-3 rounded-lg">
                      <p className="text-purple-300 text-xs mb-1">Total Lines</p>
                      <p className="text-white font-semibold text-lg">{results.statistics.total_lines}</p>
                    </div>
                    <div className="bg-slate-800/50 p-3 rounded-lg">
                      <p className="text-purple-300 text-xs mb-1">Anomalous</p>
                      <p className="text-white font-semibold text-lg">{results.statistics.anomalous_lines}</p>
                    </div>
                    <div className="bg-slate-800/50 p-3 rounded-lg">
                      <p className="text-purple-300 text-xs mb-1">Anomaly Rate</p>
                      <p className="text-white font-semibold text-lg">{results.statistics.anomaly_rate}</p>
                    </div>
                    <div className="bg-slate-800/50 p-3 rounded-lg">
                      <p className="text-purple-300 text-xs mb-1">Processing Time</p>
                      <p className="text-white font-semibold text-lg">{results.processing_time}s</p>
                    </div>
                  </div>
                )}

                {/* Details */}
                <div className="space-y-3">
                  <div className="bg-slate-800/50 p-4 rounded-lg">
                    <p className="text-purple-300 text-sm mb-1">Predicted Source</p>
                    <p className="text-white font-semibold text-lg">{results.predicted_source}</p>
                  </div>

                  <div className="bg-slate-800/50 p-4 rounded-lg">
                    <p className="text-purple-300 text-sm mb-1">Log Template</p>
                    <p className="text-white font-mono text-xs break-all">{results.template}</p>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-slate-800/50 p-4 rounded-lg">
                      <p className="text-purple-300 text-sm mb-1">Embedding Dims</p>
                      <p className="text-white font-semibold">{results.embedding_dims}</p>
                    </div>
                    <div className="bg-slate-800/50 p-4 rounded-lg">
                      <p className="text-purple-300 text-sm mb-1">Model Used</p>
                      <p className="text-white font-semibold text-sm">{results.model_used}</p>
                    </div>
                  </div>
                </div>

                {/* Detailed Results */}
                {results.detailed_results && results.detailed_results.length > 0 && (
                  <div className="bg-slate-800/50 p-4 rounded-lg">
                    <p className="text-purple-300 text-sm mb-3 flex items-center gap-2">
                      <Info className="w-4 h-4" />
                      Line-by-Line Analysis (First 10)
                    </p>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {results.detailed_results.slice(0, 10).map((line, idx) => (
                        <div 
                          key={idx}
                          className={`p-2 rounded text-xs ${
                            line.is_anomaly 
                              ? 'bg-red-900/30 border-l-2 border-red-500' 
                              : 'bg-green-900/20 border-l-2 border-green-500'
                          }`}
                        >
                          <div className="flex justify-between mb-1">
                            <span className="text-purple-300">Line {line.line_number}</span>
                            <span className={line.is_anomaly ? 'text-red-400' : 'text-green-400'}>
                              {line.prediction} ({(line.confidence * 100).toFixed(0)}%)
                            </span>
                          </div>
                          <p className="text-gray-300 font-mono truncate">{line.content}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Model Info */}
                <div className="mt-6 p-4 bg-purple-900/30 rounded-lg border border-purple-500/30">
                  <p className="text-purple-200 text-sm">
                    <span className="font-semibold">Model:</span> {results.model_used} with BERT Embeddings
                  </p>
                  <p className="text-purple-300 text-xs mt-1">
                    Trained on diverse log sources using cross-source transfer learning.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Info Footer */}
        <div className="mt-8 bg-white/5 backdrop-blur-lg rounded-xl p-6 border border-white/10">
          <h3 className="font-semibold mb-3">Supported Log Sources</h3>
          <div className="flex flex-wrap gap-2">
            {logSources.map((source) => (
              <span
                key={source}
                className="px-3 py-1 bg-purple-600/30 text-purple-200 rounded-full text-sm border border-purple-500/30"
              >
                {source}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}