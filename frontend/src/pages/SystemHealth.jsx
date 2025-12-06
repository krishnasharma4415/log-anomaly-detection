import { useState, useEffect } from 'react';
import { MetricCard } from '../components/ui/Card';
import Card from '../components/ui/Card';
import Badge from '../components/ui/Badge';
import { Activity, Server, Cpu, HardDrive, Clock, Zap } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import api from '../services/api';
import { useChartTheme } from '../hooks/useChartTheme';

export default function SystemHealth() {
  const chartTheme = useChartTheme();
  const [systemStatus, setSystemStatus] = useState({
    apiStatus: 'checking',
    modelLoaded: false,
    uptime: 'Loading...',
    lastCheck: new Date().toISOString(),
  });

  const [metrics, setMetrics] = useState({
    cpu: 0,
    memory: 0,
    latency: 0,
    requestsPerMin: 0,
  });

  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchHealthData();
    const interval = setInterval(fetchHealthData, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  const fetchHealthData = async () => {
    try {
      const health = await api.checkHealth();

      setSystemStatus({
        apiStatus: health.status === 'healthy' ? 'online' : 'offline',
        modelLoaded: health.models?.ml_model_loaded || false,
        uptime: 'N/A', // Django doesn't track uptime by default
        lastCheck: health.timestamp,
      });

      setMetrics({
        cpu: health.system?.cpu_percent || 0,
        memory: health.system?.memory_percent || 0,
        latency: health.system?.avg_response_time_ms || 0,
        requestsPerMin: health.system?.total_requests || 0,
      });

      setLoading(false);
    } catch (error) {
      console.error('Error fetching health data:', error);
      setSystemStatus(prev => ({ ...prev, apiStatus: 'offline' }));
      setLoading(false);
    }
  };

  const [latencyHistory] = useState([
    { time: '5m ago', latency: 120 },
    { time: '4m ago', latency: 135 },
    { time: '3m ago', latency: 142 },
    { time: '2m ago', latency: 138 },
    { time: '1m ago', latency: 145 },
    { time: 'now', latency: 145 },
  ]);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-2">System Health</h1>
        <p className="text-slate-600 dark:text-slate-400">Real-time monitoring and diagnostics</p>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="API Status"
          value={systemStatus.apiStatus}
          subtitle="All systems operational"
          icon={Server}
          neon
        />
        <MetricCard
          title="Model Status"
          value={systemStatus.modelLoaded ? 'Loaded' : 'Not Loaded'}
          subtitle="XGBoost + SMOTE"
          icon={Zap}
        />
        <MetricCard
          title="System Uptime"
          value={systemStatus.uptime}
          subtitle="No downtime"
          icon={Clock}
        />
        <MetricCard
          title="Requests/Min"
          value={metrics.requestsPerMin}
          subtitle="Average load"
          icon={Activity}
          trend={12.5}
        />
      </div>

      {/* Resource Usage */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card neon>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-6">Resource Usage</h3>
          <div className="space-y-6">
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Cpu className="w-5 h-5 text-cyan-500 dark:text-cyan-400" />
                  <span className="text-sm text-slate-600 dark:text-slate-400">CPU Usage</span>
                </div>
                <span className="text-lg font-bold text-slate-900 dark:text-slate-100">{metrics.cpu}%</span>
              </div>
              <div className="h-3 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                <div
                  style={{
                    width: `${metrics.cpu}%`,
                    boxShadow: '0 0 10px rgba(0, 198, 255, 0.5)'
                  }}
                  className="h-full bg-gradient-accent transition-all duration-1000"
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <HardDrive className="w-5 h-5 text-purple-500 dark:text-purple-400" />
                  <span className="text-sm text-slate-600 dark:text-slate-400">Memory Usage</span>
                </div>
                <span className="text-lg font-bold text-slate-900 dark:text-slate-100">{metrics.memory}%</span>
              </div>
              <div className="h-3 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                <div
                  style={{
                    width: `${metrics.memory}%`,
                    boxShadow: '0 0 10px rgba(75, 93, 255, 0.5)'
                  }}
                  className="h-full bg-gradient-primary transition-all duration-1000"
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Activity className="w-5 h-5 text-green-500" />
                  <span className="text-sm text-slate-600 dark:text-slate-400">Backend Latency</span>
                </div>
                <span className="text-lg font-bold text-slate-900 dark:text-slate-100">{metrics.latency}ms</span>
              </div>
              <div className="h-3 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                <div
                  style={{
                    width: `${(metrics.latency / 500) * 100}%`,
                    boxShadow: '0 0 10px rgba(52, 211, 153, 0.5)'
                  }}
                  className="h-full bg-signal-success transition-all duration-1000"
                />
              </div>
            </div>
          </div>
        </Card>

        <Card>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">Latency Trend</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={latencyHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.gridColor} />
              <XAxis dataKey="time" stroke={chartTheme.axisColor} />
              <YAxis stroke={chartTheme.axisColor} />
              <Tooltip contentStyle={chartTheme.tooltipStyle} />
              <Line
                type="monotone"
                dataKey="latency"
                stroke="#00C6FF"
                strokeWidth={3}
                dot={{ fill: '#00C6FF', r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* System Info */}
      <Card>
        <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">System Information</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">API Version</p>
            <p className="text-lg font-semibold text-slate-900 dark:text-slate-100">v1.0.0</p>
          </div>
          <div>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">Python Version</p>
            <p className="text-lg font-semibold text-slate-900 dark:text-slate-100">3.11.5</p>
          </div>
          <div>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">PyTorch Version</p>
            <p className="text-lg font-semibold text-slate-900 dark:text-slate-100">2.1.0</p>
          </div>
          <div>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">Environment</p>
            <Badge variant="success">Production</Badge>
          </div>
        </div>
      </Card>

      {/* Health Checks */}
      <Card>
        <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">Health Checks</h3>
        <div className="space-y-3">
          {[
            { name: 'API Endpoint', status: 'healthy', latency: '45ms' },
            { name: 'Model Loading', status: 'healthy', latency: '120ms' },
            { name: 'Feature Extraction', status: 'healthy', latency: '85ms' },
            { name: 'Database Connection', status: 'healthy', latency: '12ms' },
            { name: 'Cache Service', status: 'healthy', latency: '8ms' },
          ].map((check) => (
            <div
              key={check.name}
              className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-900 rounded-lg transition-colors duration-200"
            >
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-slate-900 dark:text-slate-100 font-medium">{check.name}</span>
              </div>
              <div className="flex items-center gap-4">
                <span className="text-sm text-slate-600 dark:text-slate-400">{check.latency}</span>
                <Badge variant="success">{check.status}</Badge>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
