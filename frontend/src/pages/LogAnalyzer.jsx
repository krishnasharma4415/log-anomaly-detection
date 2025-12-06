import { useState } from 'react';
import { TextArea } from '../components/ui/Input';
import Select from '../components/ui/Select';
import Button from '../components/ui/Button';
import Card from '../components/ui/Card';
import Badge from '../components/ui/Badge';
import { Scan, Sparkles, AlertCircle } from 'lucide-react';
import { AnimatePresence } from 'framer-motion';
import LogViewer from '../components/LogViewer';
import JsonViewer from '../components/JsonViewer';
import { useToast } from '../components/ui/Toast';
import api from '../services/api';

const modelOptions = [
  { value: 'ml', label: 'ML Models (XGBoost + SMOTE)' },
  { value: 'dl', label: 'Deep Learning (CNN + Attention)' },
  { value: 'bert', label: 'BERT Models (DeBERTa-v3)' },
  { value: 'fedlogcl', label: 'FedLogCL (Federated Contrastive Learning)' },
  { value: 'hlogformer', label: 'HLogFormer (Hierarchical Transformer)' },
  { value: 'meta', label: 'Meta-Learning (Few-Shot)' },
  { value: 'ensemble', label: 'Ensemble (Multiple Models)' },
];

const bertModelOptions = [
  { value: 'best', label: 'Best Model (DeBERTa-v3) - F1: 0.52' },
  { value: 'logbert', label: 'LogBERT - F1: 0.51' },
  { value: 'dapt_bert', label: 'DAPT-BERT - F1: 0.50' },
  { value: 'deberta_v3', label: 'DeBERTa-v3 - F1: 0.52' },
  { value: 'mpnet', label: 'MPNet - F1: 0.45' },
];

const ensembleMethodOptions = [
  { value: 'averaging', label: 'Averaging (Average probabilities)' },
  { value: 'voting', label: 'Voting (Majority vote)' },
];

export default function LogAnalyzer() {
  const [logInput, setLogInput] = useState('');
  const [selectedModel, setSelectedModel] = useState('ml');
  const [selectedBertModel, setSelectedBertModel] = useState('best');
  const [selectedEnsembleMethod, setSelectedEnsembleMethod] = useState('averaging');
  const [showMetadata, setShowMetadata] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const { addToast } = useToast();

  const handleAnalyze = async () => {
    if (!logInput.trim()) {
      addToast('Please enter a log message', 'warning');
      return;
    }

    setLoading(true);

    try {
      // Prepare request parameters
      const requestParams = {
        logs: [logInput],
        model_type: selectedModel,
        save_to_db: false,
      };

      // Add BERT model key if BERT is selected
      if (selectedModel === 'bert') {
        requestParams.bert_model_key = selectedBertModel;
      }

      // Add ensemble method if ensemble is selected
      if (selectedModel === 'ensemble') {
        requestParams.ensemble_method = selectedEnsembleMethod;
      }

      const response = await api.request('/api/predict', {
        method: 'POST',
        body: JSON.stringify(requestParams),
      });

      if (response.status === 'success' && response.logs && response.logs.length > 0) {
        const logResult = response.logs[0];

        // Transform Django API response to match our UI format
        const transformedResult = {
          raw: logResult.raw || logInput,
          log_type: logResult.source_type || 'Unknown',
          parsed_content: logResult.content || logResult.raw || logInput,
          prediction: {
            class_name: logResult.prediction?.predicted_class_name || 'unknown',
            predicted_class: logResult.prediction?.predicted_class || 0,
            confidence: logResult.prediction?.confidence || 0,
            probabilities: logResult.prediction?.probabilities || {},
            model_name: logResult.prediction?.model_name || selectedModel.toUpperCase(),
          },
          metadata: {
            timestamp: new Date().toISOString(),
            model_used: selectedModel,
            inference_time_ms: logResult.prediction?.inference_time_ms || response.processing_time_ms || 0,
            features_extracted: 848,
          },
        };

        setResult(transformedResult);
        addToast('Log analyzed successfully', 'success');
      } else {
        throw new Error('Invalid response format');
      }
    } catch (error) {
      console.error('Analysis error:', error);
      addToast(error.message || 'Failed to analyze log', 'error');
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return 'text-green-500';
    if (confidence >= 0.7) return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-2">Real-Time Log Analyzer</h1>
        <p className="text-slate-600 dark:text-slate-400">Analyze individual log entries with AI-powered anomaly detection</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Panel - Input */}
        <div className="space-y-6">
          <Card neon>
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4 flex items-center gap-2">
              <Scan className="w-5 h-5 text-primary-600 dark:text-primary-400" />
              Log Input
            </h3>

            <TextArea
              label="Enter log message"
              placeholder="Paste your log entry here...&#10;Example: Apr 15 12:34:56 server sshd[1234]: Failed password for admin from 192.168.1.100"
              rows={8}
              value={logInput}
              onChange={(e) => setLogInput(e.target.value)}
            />

            <div className="mt-4">
              <Select
                label="Model Type"
                options={modelOptions}
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
              />
            </div>

            {selectedModel === 'bert' && (
              <div className="mt-4">
                <Select
                  label="BERT Model Variant"
                  options={bertModelOptions}
                  value={selectedBertModel}
                  onChange={(e) => setSelectedBertModel(e.target.value)}
                />
                <p className="text-xs text-slate-500 dark:text-slate-500 mt-1">
                  Select which BERT model to use for prediction
                </p>
              </div>
            )}

            {selectedModel === 'ensemble' && (
              <div className="mt-4">
                <Select
                  label="Ensemble Method"
                  options={ensembleMethodOptions}
                  value={selectedEnsembleMethod}
                  onChange={(e) => setSelectedEnsembleMethod(e.target.value)}
                />
                <p className="text-xs text-slate-500 dark:text-slate-500 mt-1">
                  Choose how to combine predictions from multiple models
                </p>
              </div>
            )}

            <div className="mt-4 flex items-center gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showMetadata}
                  onChange={(e) => setShowMetadata(e.target.checked)}
                  className="w-4 h-4 rounded border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-primary-600 focus:ring-primary-500 focus:ring-offset-0"
                />
                <span className="text-sm text-slate-600 dark:text-slate-400">Show metadata</span>
              </label>
            </div>

            <Button
              variant="primary"
              size="lg"
              className="w-full mt-6"
              onClick={handleAnalyze}
              loading={loading}
              icon={Sparkles}
            >
              Analyze Log
            </Button>
          </Card>

          {/* Sample Logs */}
          <Card>
            <h4 className="text-sm font-semibold text-slate-900 dark:text-slate-100 mb-3">Sample Logs</h4>
            <div className="space-y-2">
              {[
                'Apr 15 12:34:56 server sshd[1234]: Failed password for admin',
                'ERROR: Connection timeout to database server 10.0.0.5',
                'INFO: User login successful from 192.168.1.100',
              ].map((sample, i) => (
                <button
                  key={i}
                  onClick={() => setLogInput(sample)}
                  className="w-full text-left px-3 py-2 bg-slate-50 dark:bg-slate-900 rounded-lg text-xs font-mono text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                >
                  {sample}
                </button>
              ))}
            </div>
          </Card>
        </div>

        {/* Right Panel - Results */}
        <div className="space-y-6">
          <AnimatePresence mode="wait">
            {result ? (
              <div
                key="result"
                className="space-y-6"
              >
                {/* Prediction Card */}
                <Card neon={result.prediction.class_name !== 'normal'}>
                  <div className="flex items-start justify-between mb-4">
                    <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Prediction Result</h3>
                    <Badge
                      variant={result.prediction.class_name === 'normal' ? 'success' : 'error'}
                      pulse={result.prediction.class_name !== 'normal'}
                    >
                      {result.prediction.class_name === 'normal' ? 'Normal' : 'Anomaly Detected'}
                    </Badge>
                  </div>

                  {result.prediction.class_name !== 'normal' && (
                    <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg flex items-start gap-3">
                      <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                      <div>
                        <p className="text-sm font-medium text-red-600 dark:text-red-400">Security Anomaly Detected</p>
                        <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">This log entry shows suspicious activity patterns</p>
                      </div>
                    </div>
                  )}

                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm text-slate-600 dark:text-slate-400">Confidence Score</span>
                        <span className={`text-lg font-bold ${getConfidenceColor(result.prediction.confidence)}`}>
                          {(result.prediction.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-3 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                        <div
                          style={{
                            width: `${result.prediction.confidence * 100}%`,
                            boxShadow: '0 0 10px currentColor',
                          }}
                          className={`h-full transition-all duration-800 ${result.prediction.confidence >= 0.9
                              ? 'bg-signal-success'
                              : result.prediction.confidence >= 0.7
                                ? 'bg-signal-warning'
                                : 'bg-signal-error'
                            }`}
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 pt-4 border-t border-slate-200 dark:border-slate-700">
                      <div>
                        <p className="text-xs text-slate-600 dark:text-slate-400 mb-1">Log Type</p>
                        <Badge variant="info">{result.log_type}</Badge>
                      </div>
                      <div>
                        <p className="text-xs text-slate-600 dark:text-slate-400 mb-1">Classification</p>
                        <Badge variant={result.prediction.class_name === 'normal' ? 'success' : 'error'}>
                          {result.prediction.class_name}
                        </Badge>
                      </div>
                    </div>
                  </div>
                </Card>

                {/* Parsed Content */}
                <Card>
                  <h4 className="text-sm font-semibold text-slate-900 dark:text-slate-100 mb-3">Parsed Content</h4>
                  <LogViewer log={result.parsed_content} />
                </Card>

                {/* Metadata */}
                {showMetadata && result.metadata && (
                  <Card>
                    <h4 className="text-sm font-semibold text-slate-900 dark:text-slate-100 mb-3">Analysis Metadata</h4>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-slate-600 dark:text-slate-400 mb-1">Model Used</p>
                        <p className="text-slate-900 dark:text-slate-100 font-medium">{result.metadata.model_used.toUpperCase()}</p>
                      </div>
                      <div>
                        <p className="text-slate-600 dark:text-slate-400 mb-1">Inference Time</p>
                        <p className="text-slate-900 dark:text-slate-100 font-medium">{result.metadata.inference_time_ms}ms</p>
                      </div>
                      <div>
                        <p className="text-slate-600 dark:text-slate-400 mb-1">Features Extracted</p>
                        <p className="text-slate-900 dark:text-slate-100 font-medium">{result.metadata.features_extracted}</p>
                      </div>
                      <div>
                        <p className="text-slate-600 dark:text-slate-400 mb-1">Timestamp</p>
                        <p className="text-slate-900 dark:text-slate-100 font-medium font-mono text-xs">
                          {new Date(result.metadata.timestamp).toLocaleString()}
                        </p>
                      </div>
                    </div>
                  </Card>
                )}

                {/* Full JSON Response */}
                <Card>
                  <h4 className="text-sm font-semibold text-slate-900 dark:text-slate-100 mb-3">Full JSON Response</h4>
                  <JsonViewer data={result} />
                </Card>
              </div>
            ) : (
              <div
                key="empty"
              >
                <Card className="h-full flex items-center justify-center min-h-[400px]">
                  <div className="text-center">
                    <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-primary-50 dark:bg-primary-900/30 flex items-center justify-center">
                      <Scan className="w-8 h-8 text-primary-600 dark:text-primary-400" />
                    </div>
                    <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">No Analysis Yet</h3>
                    <p className="text-slate-600 dark:text-slate-400 text-sm">Enter a log message and click "Analyze Log" to see results</p>
                  </div>
                </Card>
              </div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
