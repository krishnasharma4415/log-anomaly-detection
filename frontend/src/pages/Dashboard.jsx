import { useState, useEffect } from 'react';
import { MetricCard } from '../components/ui/Card';
import Card from '../components/ui/Card';
import { Activity, Clock, Brain, AlertTriangle } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import Badge from '../components/ui/Badge';
import api from '../services/api';
import { SkeletonCard } from '../components/ui/Skeleton';
import { useChartTheme } from '../hooks/useChartTheme';

const COLORS = ['#4B5DFF', '#00C6FF', '#A259FF', '#22E1FF', '#34D399', '#FBBF24'];

export default function Dashboard() {
  const chartTheme = useChartTheme();
  const [stats, setStats] = useState({
    totalLogs: 0,
    anomalyRate: 0,
    avgLatency: 0,
    activeModel: 'Loading...',
  });
  const [loading, setLoading] = useState(true);
  const [recentLogs, setRecentLogs] = useState([]);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const [statistics, health, logs] = await Promise.all([
        api.getStatistics(),
        api.checkHealth(),
        api.getRecentLogs(10),
      ]);

      setStats({
        totalLogs: statistics.totalLogs || 0,
        anomalyRate: statistics.anomalyRate || 0,
        avgLatency: health.system?.avg_response_time_ms || 145,
        activeModel: 'XGBoost + SMOTE',
      });

      // Transform logs for display
      const transformedLogs = logs.map((log, index) => ({
        id: log.id || index + 1,
        timestamp: new Date(log.created_at).toLocaleString(),
        source: log.source_name || 'Unknown',
        message: log.parsed_content || log.raw_content || 'No content',
        class: log.predictions?.[0]?.predicted_class_name || 'Normal',
        confidence: log.predictions?.[0]?.confidence || 0,
      }));

      setRecentLogs(transformedLogs);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const [timelineData] = useState([
    { time: '00:00', normal: 120, anomaly: 15 },
    { time: '04:00', normal: 95, anomaly: 8 },
    { time: '08:00', normal: 180, anomaly: 22 },
    { time: '12:00', normal: 210, anomaly: 35 },
    { time: '16:00', normal: 165, anomaly: 18 },
    { time: '20:00', normal: 140, anomaly: 12 },
  ]);

  const [classDistribution] = useState([
    { name: 'Normal', value: 28500, color: '#34D399' },
    { name: 'Security', value: 1850, color: '#F43F5E' },
    { name: 'Performance', value: 1200, color: '#FBBF24' },
    { name: 'Network', value: 650, color: '#00C6FF' },
    { name: 'System', value: 347, color: '#A259FF' },
  ]);

  const [sourceDistribution] = useState([
    { source: 'Apache', count: 5200 },
    { source: 'Linux', count: 4800 },
    { source: 'HDFS', count: 3900 },
    { source: 'OpenSSH', count: 3200 },
    { source: 'Android', count: 2800 },
    { source: 'Others', count: 12647 },
  ]);



  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-2">Dashboard</h1>
        <p className="text-slate-600 dark:text-slate-400">Real-time overview of log anomaly detection system</p>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Logs Analyzed"
          value={stats.totalLogs.toLocaleString()}
          subtitle="Last 24 hours"
          icon={Activity}
          trend={8.2}
          neon
        />
        <MetricCard
          title="Anomaly Rate"
          value={`${stats.anomalyRate}%`}
          subtitle="Detection accuracy"
          icon={AlertTriangle}
          trend={-2.1}
        />
        <MetricCard
          title="Avg Inference Time"
          value={`${stats.avgLatency}ms`}
          subtitle="Per log analysis"
          icon={Clock}
        />
        <MetricCard
          title="Active Model"
          value={stats.activeModel}
          subtitle="88.5% F1-Score"
          icon={Brain}
        />
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Anomaly Timeline */}
        <Card neon>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">Anomaly Timeline</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={timelineData}>
              <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.gridColor} />
              <XAxis dataKey="time" stroke={chartTheme.axisColor} />
              <YAxis stroke={chartTheme.axisColor} />
              <Tooltip contentStyle={chartTheme.tooltipStyle} />
              <Legend />
              <Line type="monotone" dataKey="normal" stroke="#34D399" strokeWidth={2} dot={{ fill: '#34D399' }} />
              <Line type="monotone" dataKey="anomaly" stroke="#F43F5E" strokeWidth={2} dot={{ fill: '#F43F5E' }} />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        {/* Class Distribution */}
        <Card>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">Class Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={classDistribution}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={2}
                dataKey="value"
              >
                {classDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip contentStyle={chartTheme.tooltipStyle} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Charts Row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Source Distribution */}
        <Card className="lg:col-span-2">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">Source Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={sourceDistribution}>
              <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.gridColor} />
              <XAxis dataKey="source" stroke={chartTheme.axisColor} />
              <YAxis stroke={chartTheme.axisColor} />
              <Tooltip contentStyle={chartTheme.tooltipStyle} />
              <Bar dataKey="count" fill="url(#colorGradient)" radius={[8, 8, 0, 0]} />
              <defs>
                <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#4B5DFF" />
                  <stop offset="100%" stopColor="#A259FF" />
                </linearGradient>
              </defs>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Model Performance */}
        <Card>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">Model Performance</h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-600 dark:text-slate-400">F1-Score</span>
                <span className="text-slate-900 dark:text-slate-100 font-semibold">88.5%</span>
              </div>
              <div className="h-2 bg-neutral-dark rounded-full overflow-hidden">
                <div
                  style={{ width: '88.5%' }}
                  className="h-full bg-gradient-primary transition-all duration-1000"
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-600 dark:text-slate-400">Balanced Accuracy</span>
                <span className="text-slate-900 dark:text-slate-100 font-semibold">91.2%</span>
              </div>
              <div className="h-2 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                <div
                  style={{ width: '91.2%' }}
                  className="h-full bg-gradient-accent transition-all duration-1000"
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-600 dark:text-slate-400">AUROC</span>
                <span className="text-slate-900 dark:text-slate-100 font-semibold">94.0%</span>
              </div>
              <div className="h-2 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                <div
                  style={{ width: '94%' }}
                  className="h-full bg-signal-success transition-all duration-1000"
                />
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Recent Logs Table */}
      <Card>
        <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">Recent Analyzed Logs</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-700">
                <th className="text-left py-3 px-4 text-sm font-medium text-slate-600 dark:text-slate-400">Timestamp</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-slate-600 dark:text-slate-400">Source</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-slate-600 dark:text-slate-400">Message</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-slate-600 dark:text-slate-400">Class</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-slate-600 dark:text-slate-400">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {recentLogs.map((log) => (
                <tr
                  key={log.id}
                  className="border-b border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
                >
                  <td className="py-3 px-4 text-sm text-slate-600 dark:text-slate-400 font-mono">{log.timestamp}</td>
                  <td className="py-3 px-4 text-sm">
                    <Badge variant="info">{log.source}</Badge>
                  </td>
                  <td className="py-3 px-4 text-sm text-slate-900 dark:text-slate-100 max-w-md truncate">{log.message}</td>
                  <td className="py-3 px-4 text-sm">
                    <Badge variant={log.class === 'Normal' ? 'success' : 'error'}>{log.class}</Badge>
                  </td>
                  <td className="py-3 px-4 text-sm text-slate-900 dark:text-slate-100 font-semibold">{(log.confidence * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
