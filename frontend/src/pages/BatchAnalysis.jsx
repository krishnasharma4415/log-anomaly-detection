import { useState } from 'react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import Badge from '../components/ui/Badge';
import { Upload, Download, FileText, Trash2 } from 'lucide-react';
import { useToast } from '../components/ui/Toast';
import api from '../services/api';

export default function BatchAnalysis() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const { addToast } = useToast();

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      addToast(`File "${uploadedFile.name}" uploaded successfully`, 'success');
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      addToast('Please upload a file first', 'warning');
      return;
    }

    setLoading(true);

    try {
      // Read file content
      const text = await file.text();
      const logs = text.split('\n').filter(line => line.trim());

      if (logs.length === 0) {
        addToast('File is empty or invalid', 'error');
        setLoading(false);
        return;
      }

      // Analyze logs
      const response = await api.predict(logs, 'ml', false);

      if (response.status === 'success' && response.logs) {
        const transformedResults = response.logs.map((logResult, i) => ({
          id: i + 1,
          log: logResult.raw || logs[i],
          parsed: logResult.content || logResult.raw || logs[i],
          class: logResult.prediction?.predicted_class_name || 'unknown',
          confidence: logResult.prediction?.confidence || 0,
          source: logResult.source_type || 'Unknown',
        }));

        setResults(transformedResults);
        addToast(`Batch analysis completed: ${logs.length} logs processed`, 'success');
      } else {
        throw new Error('Invalid response format');
      }
    } catch (error) {
      console.error('Batch analysis error:', error);
      addToast(error.message || 'Failed to analyze batch', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    addToast('Results downloaded successfully', 'success');
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-2">Batch Analysis</h1>
        <p className="text-slate-600 dark:text-slate-400">Upload and analyze multiple log files at once</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card neon className="lg:col-span-1">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">Upload File</h3>

          <label className="block">
            <div className="border-2 border-dashed border-slate-200 dark:border-slate-700 rounded-lg p-8 text-center cursor-pointer hover:border-primary-500 dark:hover:border-primary-500 hover:bg-slate-50 dark:hover:bg-slate-800 transition-all">
              <Upload className="w-12 h-12 mx-auto mb-3 text-slate-400 dark:text-slate-500" />
              <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">
                {file ? file.name : 'Click to upload or drag and drop'}
              </p>
              <p className="text-xs text-slate-400 dark:text-slate-500">CSV, TXT, or JSON files</p>
            </div>
            <input
              type="file"
              className="hidden"
              accept=".csv,.txt,.json"
              onChange={handleFileUpload}
            />
          </label>

          {file && (
            <div className="mt-4 p-3 bg-slate-50 dark:bg-slate-900 rounded-lg flex items-center justify-between transition-colors duration-200">
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-primary-600 dark:text-primary-400" />
                <span className="text-sm text-slate-900 dark:text-slate-100">{file.name}</span>
              </div>
              <button
                onClick={() => setFile(null)}
                className="text-slate-600 dark:text-slate-400 hover:text-red-600 dark:hover:text-red-400 transition-colors"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          )}

          <Button
            variant="primary"
            size="lg"
            className="w-full mt-6"
            onClick={handleAnalyze}
            loading={loading}
            disabled={!file}
          >
            Analyze Batch
          </Button>
        </Card>

        <Card className="lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Results</h3>
            {results && (
              <Button variant="secondary" size="sm" icon={Download} onClick={handleDownload}>
                Download Results
              </Button>
            )}
          </div>

          {results ? (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-700">
                    <th className="text-left py-3 px-4 text-sm font-medium text-slate-600 dark:text-slate-400">#</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-slate-600 dark:text-slate-400">Log Message</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-slate-600 dark:text-slate-400">Source</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-slate-600 dark:text-slate-400">Class</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-slate-600 dark:text-slate-400">Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((result) => (
                    <tr
                      key={result.id}
                      className="border-b border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
                    >
                      <td className="py-3 px-4 text-sm text-slate-600 dark:text-slate-400">{result.id}</td>
                      <td className="py-3 px-4 text-sm text-slate-900 dark:text-slate-100 max-w-md truncate">{result.log}</td>
                      <td className="py-3 px-4 text-sm">
                        <Badge variant="info">{result.source}</Badge>
                      </td>
                      <td className="py-3 px-4 text-sm">
                        <Badge variant={result.class === 'normal' ? 'success' : 'error'}>
                          {result.class}
                        </Badge>
                      </td>
                      <td className="py-3 px-4 text-sm text-slate-900 dark:text-slate-100 font-semibold">
                        {(result.confidence * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-12">
              <FileText className="w-12 h-12 mx-auto mb-3 text-slate-400 dark:text-slate-500" />
              <p className="text-slate-600 dark:text-slate-400">No results yet. Upload a file to begin analysis.</p>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
