import { useState } from 'react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import Select from '../components/ui/Select';
import Badge from '../components/ui/Badge';
import { Moon, Sun, Zap, Shield, Save } from 'lucide-react';
import { useToast } from '../components/ui/Toast';

export default function Settings() {
  const [theme, setTheme] = useState('dark');
  const [activeModel, setActiveModel] = useState('xgboost');
  const [threshold, setThreshold] = useState(0.7);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const { addToast } = useToast();

  const handleSave = () => {
    addToast('Settings saved successfully', 'success');
  };

  return (
    <div className="space-y-8 max-w-4xl">
      <div>
        <h1 className="text-3xl font-bold text-neutral-primary mb-2">Settings</h1>
        <p className="text-neutral-secondary">Configure your system preferences</p>
      </div>

      {/* Appearance */}
      <Card neon>
        <h3 className="text-lg font-semibold text-neutral-primary mb-4 flex items-center gap-2">
          <Moon className="w-5 h-5 text-primary" />
          Appearance
        </h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-neutral-secondary mb-2">
              Theme
            </label>
            <div className="flex gap-3">
              <button
                onClick={() => setTheme('dark')}
                className={`flex-1 p-4 rounded-lg border-2 transition-all ${
                  theme === 'dark'
                    ? 'border-primary bg-primary/10'
                    : 'border-neutral-border bg-neutral-dark hover:border-neutral-surface'
                }`}
              >
                <Moon className="w-6 h-6 mx-auto mb-2 text-primary" />
                <p className="text-sm font-medium text-neutral-primary">Dark</p>
              </button>
              <button
                onClick={() => setTheme('light')}
                className={`flex-1 p-4 rounded-lg border-2 transition-all ${
                  theme === 'light'
                    ? 'border-primary bg-primary/10'
                    : 'border-neutral-border bg-neutral-dark hover:border-neutral-surface'
                }`}
              >
                <Sun className="w-6 h-6 mx-auto mb-2 text-signal-warning" />
                <p className="text-sm font-medium text-neutral-primary">Light</p>
              </button>
            </div>
            <p className="text-xs text-neutral-disabled mt-2">
              Light theme coming soon. Currently only dark theme is supported.
            </p>
          </div>
        </div>
      </Card>

      {/* Model Configuration */}
      <Card>
        <h3 className="text-lg font-semibold text-neutral-primary mb-4 flex items-center gap-2">
          <Zap className="w-5 h-5 text-accent-cyan" />
          Model Configuration
        </h3>
        <div className="space-y-4">
          <Select
            label="Active Model"
            value={activeModel}
            onChange={(e) => setActiveModel(e.target.value)}
            options={[
              { value: 'xgboost', label: 'XGBoost + SMOTE (Best Overall)' },
              { value: 'lightgbm', label: 'LightGBM + Focal Loss (Fast)' },
              { value: 'balanced-rf', label: 'Balanced Random Forest (Interpretable)' },
              { value: 'flnn', label: 'Focal Loss Neural Network (Deep Learning)' },
              { value: 'logbert', label: 'LogBERT (Semantic Understanding)' },
            ]}
          />

          <div>
            <label className="block text-sm font-medium text-neutral-secondary mb-2">
              Anomaly Detection Threshold: {threshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0.5"
              max="0.95"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full h-2 bg-neutral-dark rounded-lg appearance-none cursor-pointer accent-primary"
            />
            <div className="flex justify-between text-xs text-neutral-disabled mt-1">
              <span>More Sensitive (0.5)</span>
              <span>More Specific (0.95)</span>
            </div>
          </div>

          <div className="p-4 bg-neutral-dark rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-neutral-primary mb-1">Performance Mode</p>
                <p className="text-xs text-neutral-secondary">
                  {activeModel === 'xgboost' ? 'Balanced - Best accuracy' : 
                   activeModel === 'lightgbm' ? 'Fast - Quick inference' : 
                   'Custom configuration'}
                </p>
              </div>
              <Badge variant="primary">
                {activeModel === 'lightgbm' ? 'Speed' : 'Accuracy'}
              </Badge>
            </div>
          </div>
        </div>
      </Card>

      {/* Advanced Settings */}
      <Card>
        <h3 className="text-lg font-semibold text-neutral-primary mb-4 flex items-center gap-2">
          <Shield className="w-5 h-5 text-accent-purple" />
          Advanced Settings
        </h3>
        <div className="space-y-4">
          <label className="flex items-center justify-between p-4 bg-neutral-dark rounded-lg cursor-pointer">
            <div>
              <p className="text-sm font-medium text-neutral-primary mb-1">Auto-refresh Dashboard</p>
              <p className="text-xs text-neutral-secondary">Automatically update metrics every 30 seconds</p>
            </div>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="w-5 h-5 rounded border-neutral-border bg-neutral-surface text-primary focus:ring-primary focus:ring-offset-0"
            />
          </label>

          <label className="flex items-center justify-between p-4 bg-neutral-dark rounded-lg cursor-pointer">
            <div>
              <p className="text-sm font-medium text-neutral-primary mb-1">Enable Notifications</p>
              <p className="text-xs text-neutral-secondary">Get alerts for critical anomalies</p>
            </div>
            <input
              type="checkbox"
              defaultChecked
              className="w-5 h-5 rounded border-neutral-border bg-neutral-surface text-primary focus:ring-primary focus:ring-offset-0"
            />
          </label>

          <label className="flex items-center justify-between p-4 bg-neutral-dark rounded-lg cursor-pointer">
            <div>
              <p className="text-sm font-medium text-neutral-primary mb-1">Detailed Logging</p>
              <p className="text-xs text-neutral-secondary">Store detailed analysis logs for debugging</p>
            </div>
            <input
              type="checkbox"
              className="w-5 h-5 rounded border-neutral-border bg-neutral-surface text-primary focus:ring-primary focus:ring-offset-0"
            />
          </label>
        </div>
      </Card>

      {/* Save Button */}
      <div className="flex justify-end gap-3">
        <Button variant="ghost" size="lg">
          Reset to Defaults
        </Button>
        <Button variant="primary" size="lg" icon={Save} onClick={handleSave}>
          Save Settings
        </Button>
      </div>
    </div>
  );
}
