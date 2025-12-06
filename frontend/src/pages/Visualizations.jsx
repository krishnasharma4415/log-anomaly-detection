import Card from '../components/ui/Card';
import { BarChart, Bar, LineChart, Line, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Area, AreaChart } from 'recharts';
import { useChartTheme } from '../hooks/useChartTheme';

const confidenceData = [
  { range: '0-20%', count: 45 },
  { range: '20-40%', count: 120 },
  { range: '40-60%', count: 380 },
  { range: '60-80%', count: 1250 },
  { range: '80-100%', count: 3200 },
];

const featureImportance = [
  { feature: 'BERT Embeddings', importance: 0.45 },
  { feature: 'Template Features', importance: 0.25 },
  { feature: 'Temporal Patterns', importance: 0.18 },
  { feature: 'Statistical Features', importance: 0.08 },
  { feature: 'Text Complexity', importance: 0.04 },
];

const sourceHeatmap = [
  { source: 'Apache', normal: 4800, anomaly: 400 },
  { source: 'Linux', normal: 4200, anomaly: 600 },
  { source: 'HDFS', normal: 3500, anomaly: 400 },
  { source: 'OpenSSH', normal: 2900, anomaly: 300 },
  { source: 'Android', normal: 2700, anomaly: 100 },
  { source: 'Hadoop', normal: 3100, anomaly: 800 },
];

const modelComparison = [
  { model: 'XGBoost', f1: 88.5, precision: 89.2, recall: 87.8, accuracy: 91.2 },
  { model: 'LightGBM', f1: 87.3, precision: 88.1, recall: 86.5, accuracy: 90.8 },
  { model: 'RF', f1: 86.1, precision: 87.0, recall: 85.2, accuracy: 89.5 },
  { model: 'FLNN', f1: 84.7, precision: 85.5, recall: 83.9, accuracy: 88.3 },
  { model: 'TabNet', f1: 83.2, precision: 84.0, recall: 82.4, accuracy: 87.1 },
  { model: 'LogBERT', f1: 82.5, precision: 83.3, recall: 81.7, accuracy: 86.8 },
];

const aurocCurve = [
  { fpr: 0, tpr: 0 },
  { fpr: 0.05, tpr: 0.65 },
  { fpr: 0.1, tpr: 0.82 },
  { fpr: 0.15, tpr: 0.89 },
  { fpr: 0.2, tpr: 0.93 },
  { fpr: 0.3, tpr: 0.96 },
  { fpr: 0.5, tpr: 0.98 },
  { fpr: 1, tpr: 1 },
];

export default function Visualizations() {
  const chartTheme = useChartTheme();

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-2">Visualization Center</h1>
        <p className="text-slate-600 dark:text-slate-400">Advanced analytics and model performance visualizations</p>
      </div>

      {/* Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card neon>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">Confidence Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={confidenceData}>
              <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.gridColor} />
              <XAxis dataKey="range" stroke={chartTheme.axisColor} />
              <YAxis stroke={chartTheme.axisColor} />
              <Tooltip contentStyle={chartTheme.tooltipStyle} />
              <Bar dataKey="count" fill="url(#confidenceGradient)" radius={[8, 8, 0, 0]} />
              <defs>
                <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#00C6FF" />
                  <stop offset="100%" stopColor="#22E1FF" />
                </linearGradient>
              </defs>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">Feature Importance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={featureImportance} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.gridColor} />
              <XAxis type="number" stroke={chartTheme.axisColor} />
              <YAxis dataKey="feature" type="category" stroke={chartTheme.axisColor} width={150} />
              <Tooltip contentStyle={chartTheme.tooltipStyle} />
              <Bar dataKey="importance" fill="#A259FF" radius={[0, 8, 8, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">Per-Source Heatmap</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={sourceHeatmap}>
              <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.gridColor} />
              <XAxis dataKey="source" stroke={chartTheme.axisColor} />
              <YAxis stroke={chartTheme.axisColor} />
              <Tooltip contentStyle={chartTheme.tooltipStyle} />
              <Legend />
              <Bar dataKey="normal" stackId="a" fill="#34D399" radius={[0, 0, 0, 0]} />
              <Bar dataKey="anomaly" stackId="a" fill="#F43F5E" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card neon>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">Multi-Model Radar Chart</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={modelComparison.slice(0, 4)}>
              <PolarGrid stroke={chartTheme.gridColor} />
              <PolarAngleAxis dataKey="model" stroke={chartTheme.axisColor} />
              <PolarRadiusAxis stroke={chartTheme.axisColor} />
              <Radar name="F1-Score" dataKey="f1" stroke="#4B5DFF" fill="#4B5DFF" fillOpacity={0.3} />
              <Radar name="Accuracy" dataKey="accuracy" stroke="#00C6FF" fill="#00C6FF" fillOpacity={0.3} />
              <Tooltip contentStyle={chartTheme.tooltipStyle} />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Row 3 */}
      <Card className="lg:col-span-2">
        <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">AUROC Curve</h3>
        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={aurocCurve}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.gridColor} />
            <XAxis dataKey="fpr" label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5 }} stroke={chartTheme.axisColor} />
            <YAxis label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft' }} stroke={chartTheme.axisColor} />
            <Tooltip contentStyle={chartTheme.tooltipStyle} />
            <Area type="monotone" dataKey="tpr" stroke="#4B5DFF" fill="url(#aurocGradient)" strokeWidth={3} />
            <Line type="linear" dataKey="fpr" stroke="#7C7F92" strokeDasharray="5 5" dot={false} />
            <defs>
              <linearGradient id="aurocGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#4B5DFF" stopOpacity={0.8} />
                <stop offset="100%" stopColor="#A259FF" stopOpacity={0.2} />
              </linearGradient>
            </defs>
          </AreaChart>
        </ResponsiveContainer>
        <p className="text-center text-sm text-slate-600 dark:text-slate-400 mt-4">
          AUROC: 0.94 | AUC represents the model's ability to distinguish between classes
        </p>
      </Card>
    </div>
  );
}
