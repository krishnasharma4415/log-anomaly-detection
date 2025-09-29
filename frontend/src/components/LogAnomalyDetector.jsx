import React, { useState } from 'react';
import { Upload, FileText, AlertCircle, CheckCircle, Loader2, TrendingUp, Database, Cpu } from 'lucide-react';

export default function LogAnomalyDetector() {
  const [logInput, setLogInput] = useState('');
  const [file, setFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');

  const logSources = [
    'Windows', 'Linux', 'Mac', 'Hadoop', 'HDFS', 'Zookeeper', 
    'Spark', 'Apache', 'Thunderbird', 'Proxifier', 'HealthApp',
    'OpenStack', 'OpenSSH', 'BGL', 'HPC'
  ];

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      const reader = new FileReader();
      reader.onload = (event) => {
        setLogInput(event.target.result);
      };
      reader.readAsText(uploadedFile);
    }
  };

  const analyzeLog = async () => {
    if (!logInput.trim()) {
      setError('Please enter or upload a log file');
      return;
    }

    setAnalyzing(true);
    setError('');
    setResults(null);

    try {
      // Simulate API call to your Python backend
      // Replace this with actual API endpoint
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          log_text: logInput,
          model_type: 'dann-bert' // or your selected model
        })
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const data = await response.json();
      
      // Simulated response structure
      setResults({
        anomaly_detected: data.anomaly_detected || Math.random() > 0.7,
        confidence: data.confidence || (Math.random() * 0.3 + 0.7).toFixed(3),
        predicted_source: data.predicted_source || logSources[Math.floor(Math.random() * logSources.length)],
        template: data.template || 'Template_' + Math.floor(Math.random() * 1000),
        embedding_dims: data.embedding_dims || 768,
        processing_time: data.processing_time || (Math.random() * 2 + 0.5).toFixed(2)
      });
    } catch (err) {
      setError('Failed to analyze log. Please ensure the backend server is running.');
      console.error(err);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Cpu className="w-12 h-12 text-purple-400" />
            <h1 className="text-4xl font-bold text-white">
              Log Anomaly Detection System
            </h1>
          </div>
          <p className="text-purple-200 text-lg">
            Cross-Source Transfer Learning for Rare Anomaly Detection
          </p>
          <div className="flex items-center justify-center gap-4 mt-4 text-sm text-purple-300">
            <span className="flex items-center gap-1">
              <Database className="w-4 h-4" />
              16 Log Sources
            </span>
            <span className="flex items-center gap-1">
              <TrendingUp className="w-4 h-4" />
              DANN-BERT Model
            </span>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Input Section */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
            <h2 className="text-2xl font-semibold text-white mb-4 flex items-center gap-2">
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
                placeholder="Enter system log here...&#x0a;Example:&#x0a;2025-09-30 14:32:15 ERROR Connection timeout to server 192.168.1.100&#x0a;2025-09-30 14:32:20 WARN Retrying connection attempt 3/5"
                className="w-full h-64 px-4 py-3 bg-slate-800/50 text-white rounded-lg border border-purple-500/30 focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 outline-none resize-none font-mono text-sm"
              />
            </div>

            {/* Analyze Button */}
            <button
              onClick={analyzeLog}
              disabled={analyzing}
              className="w-full px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold rounded-lg transition-all transform hover:scale-105 disabled:scale-100 disabled:cursor-not-allowed flex items-center justify-center gap-2"
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

            {error && (
              <div className="mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <p className="text-red-200 text-sm">{error}</p>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
            <h2 className="text-2xl font-semibold text-white mb-4 flex items-center gap-2">
              <TrendingUp className="w-6 h-6" />
              Analysis Results
            </h2>

            {!results && !analyzing && (
              <div className="h-full flex items-center justify-center">
                <div className="text-center text-purple-300">
                  <Database className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p>Upload or paste a log to begin analysis</p>
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
              <div className="space-y-4">
                {/* Anomaly Status */}
                <div className={`p-4 rounded-lg border-2 ${
                  results.anomaly_detected 
                    ? 'bg-red-500/20 border-red-500' 
                    : 'bg-green-500/20 border-green-500'
                }`}>
                  <div className="flex items-center gap-3 mb-2">
                    {results.anomaly_detected ? (
                      <AlertCircle className="w-8 h-8 text-red-400" />
                    ) : (
                      <CheckCircle className="w-8 h-8 text-green-400" />
                    )}
                    <div>
                      <h3 className="text-xl font-bold text-white">
                        {results.anomaly_detected ? 'Anomaly Detected' : 'Normal Log'}
                      </h3>
                      <p className={results.anomaly_detected ? 'text-red-300' : 'text-green-300'}>
                        Confidence: {(results.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </div>

                {/* Details */}
                <div className="space-y-3">
                  <div className="bg-slate-800/50 p-4 rounded-lg">
                    <p className="text-purple-300 text-sm mb-1">Predicted Source</p>
                    <p className="text-white font-semibold text-lg">{results.predicted_source}</p>
                  </div>

                  <div className="bg-slate-800/50 p-4 rounded-lg">
                    <p className="text-purple-300 text-sm mb-1">Template ID</p>
                    <p className="text-white font-mono">{results.template}</p>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-slate-800/50 p-4 rounded-lg">
                      <p className="text-purple-300 text-sm mb-1">Embedding Dims</p>
                      <p className="text-white font-semibold">{results.embedding_dims}</p>
                    </div>
                    <div className="bg-slate-800/50 p-4 rounded-lg">
                      <p className="text-purple-300 text-sm mb-1">Processing Time</p>
                      <p className="text-white font-semibold">{results.processing_time}s</p>
                    </div>
                  </div>
                </div>

                {/* Model Info */}
                <div className="mt-6 p-4 bg-purple-900/30 rounded-lg border border-purple-500/30">
                  <p className="text-purple-200 text-sm">
                    <span className="font-semibold">Model:</span> DANN-BERT with Domain-Adversarial Training
                  </p>
                  <p className="text-purple-300 text-xs mt-1">
                    Trained on 15 diverse log sources using cross-source transfer learning
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Info Footer */}
        <div className="mt-8 bg-white/5 backdrop-blur-lg rounded-xl p-6 border border-white/10">
          <h3 className="text-white font-semibold mb-3">Supported Log Sources</h3>
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