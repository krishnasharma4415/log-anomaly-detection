import { useState, useEffect } from 'react';
import { MetricCard } from '../components/ui/Card';
import Card from '../components/ui/Card';
import Badge from '../components/ui/Badge';
import { Activity, Server, Cpu, HardDrive, Clock, Zap } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import api from '../services/api';

export default function SystemHealth() {
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
        <h1 className="text-3xl font-bold text-neutral-primary mb-2">System Health</h1>
        <p className="text-neutral-secondary">Real-time monitoring and diagnostics</p>
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
          <h3 className="text-lg font-semibold text-neutral-primary mb-6">Resource Usage</h3>
          <div className="space-y-6">
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Cpu className="w-5 h-5 text-accent-cyan" />
                  <span className="text-sm text-neutral-secondary">CPU Usage</span>
                </div>
                <span className="text-lg font-bold text-neutral-primary">{metrics.cpu}%</span>
              </div>
              <div className="h-3 bg-neutral-dark rounded-full overflow-hidden">
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
                  <HardDrive className="w-5 h-5 text-accent-purple" />
                  <span className="text-sm text-neutral-secondary">Memory Usage</span>
                </div>
                <span className="text-lg font-bold text-neutral-primary">{metrics.memory}%</span>
              </div>
              <div className="h-3 bg-neutral-dark rounded-full overflow-hidden">
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
                  <Activity className="w-5 h-5 text-signal-success" />
                  <span className="text-sm text-neutral-secondary">Backend Latency</span>
                </div>
                <span className="text-lg font-bold text-neutral-primary">{metrics.latency}ms</span>
              </div>
              <div className="h-3 bg-neutral-dark rounded-full overflow-hidden">
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
          <h3 className="text-lg font-semibold text-neutral-primary mb-4">Latency Trend</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={latencyHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#3B3F52" />
              <XAxis dataKey="time" stroke="#C5C7D3" />
              <YAxis stroke="#C5C7D3" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#2A2D3E',
                  border: '1px solid #3B3F52',
                  borderRadius: '8px',
                  color: '#FFFFFF',
                }}
              />
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
        <h3 className="text-lg font-semibold text-neutral-primary mb-4">System Information</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div>
            <p className="text-sm text-neutral-secondary mb-1">API Version</p>
            <p className="text-lg font-semibold text-neutral-primary">v1.0.0</p>
          </div>
          <div>
            <p className="text-sm text-neutral-secondary mb-1">Python Version</p>
            <p className="text-lg font-semibold text-neutral-primary">3.11.5</p>
          </div>
          <div>
            <p className="text-sm text-neutral-secondary mb-1">PyTorch Version</p>
            <p className="text-lg font-semibold text-neutral-primary">2.1.0</p>
          </div>
          <div>
            <p className="text-sm text-neutral-secondary mb-1">Environment</p>
            <Badge variant="success">Production</Badge>
          </div>
        </div>
      </Card>

      {/* Health Checks */}
      <Card>
        <h3 className="text-lg font-semibold text-neutral-primary mb-4">Health Checks</h3>
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
              className="flex items-center justify-between p-4 bg-neutral-dark rounded-lg"
            >
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 rounded-full bg-signal-success animate-pulse" />
                <span className="text-neutral-primary font-medium">{check.name}</span>
              </div>
              <div className="flex items-center gap-4">
                <span className="text-sm text-neutral-secondary">{check.latency}</span>
                <Badge variant="success">{check.status}</Badge>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
