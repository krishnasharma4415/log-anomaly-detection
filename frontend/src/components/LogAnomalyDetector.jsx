import React from 'react';
import { FileText, TrendingUp } from 'lucide-react';
import { useLogAnalysis } from '../hooks/useLogAnalysis';

// Layout
import Header from './layout/Header';
import LogSourcesFooter from './layout/LogSourcesFooter';

// Input
import ModelSelector from './input/ModelSelector';
import FileUpload from './input/FileUpload';
import LogTextInput from './input/LogTextInput';
import ActionButtons from './input/ActionButtons';

// Common
import ErrorDisplay from './common/ErrorDisplay';
import EmptyState from './common/EmptyState';
import LoadingState from './common/LoadingState';

// Results
import ResultsDisplay from './results/ResultsDisplay';

export default function LogAnomalyDetector() {
  const {
    logInput,
    file,
    selectedModel,
    analyzing,
    results,
    error,
    apiStatus,
    setLogInput,
    handleFileUpload,
    handleModelChange,
    analyzeLog,
    clearAll,
    loadSampleLog
  } = useLogAnalysis();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-4 sm:p-6 font-sans">
      <div className="max-w-7xl mx-auto">
        <Header apiStatus={apiStatus} />

        <div className="grid md:grid-cols-2 gap-6">
          {/* Input Panel */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
            <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
              <FileText className="w-6 h-6" />
              Input Log Data
            </h2>

            <ModelSelector
              selectedModel={selectedModel}
              onModelChange={handleModelChange}
              disabled={analyzing}
            />

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

          {/* Results Panel */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
            <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
              <TrendingUp className="w-6 h-6" />
              Analysis Results
            </h2>

            {!results && !analyzing && <EmptyState />}
            {analyzing && <LoadingState selectedModel={selectedModel} />}
            {results && <ResultsDisplay results={results} />}
          </div>
        </div>

        <LogSourcesFooter />
      </div>
    </div>
  );
}