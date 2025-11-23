import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Scan, 
  Layers, 
  Brain, 
  BarChart3, 
  Activity,
  Settings,
  Zap
} from 'lucide-react';

const navItems = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/analyzer', icon: Scan, label: 'Log Analyzer' },
  { path: '/batch', icon: Layers, label: 'Batch Analysis' },
  { path: '/models', icon: Brain, label: 'Model Explorer' },
  { path: '/visualizations', icon: BarChart3, label: 'Visualizations' },
  { path: '/health', icon: Activity, label: 'System Health' },
  { path: '/settings', icon: Settings, label: 'Settings' },
];

export default function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 h-screen w-64 bg-indigo-900 border-r border-neutral-border flex flex-col z-30">
      {/* Logo */}
      <div className="p-6 border-b border-neutral-border">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-primary flex items-center justify-center shadow-neon">
            <Zap className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white">LogAI</h1>
            <p className="text-xs text-neutral-secondary">Anomaly Detection</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => `
              flex items-center gap-3 px-4 py-3 rounded-lg transition-all group
              ${isActive 
                ? 'bg-primary text-white shadow-neon' 
                : 'text-neutral-secondary hover:bg-neutral-surface hover:text-white'
              }
            `}
          >
            {({ isActive }) => (
              <>
                <item.icon className={`w-5 h-5 ${isActive ? 'text-white' : 'text-accent-cyan group-hover:text-accent-cyan'}`} />
                <span className="font-medium">{item.label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-neutral-border">
        <div className="bg-neutral-surface rounded-lg p-3 border border-neutral-border">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 rounded-full bg-signal-success animate-pulse" />
            <span className="text-xs font-medium text-neutral-secondary">System Online</span>
          </div>
          <p className="text-xs text-neutral-disabled">v1.0.0 • Production</p>
        </div>
      </div>
    </aside>
  );
}
