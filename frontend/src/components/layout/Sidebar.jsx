import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Scan,
  Layers,
  Brain,
  BarChart3,
  Activity,
  Settings
} from 'lucide-react';
import hackerLogo from '../../hacker.png';

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
    <aside className="fixed left-0 top-0 h-screen w-64 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-700 flex flex-col z-30 shadow-sm dark:shadow-none transition-colors duration-200">
      {/* Logo */}
      <div className="p-6 border-b border-slate-200 dark:border-slate-700">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center shadow-sm overflow-hidden">
            <img src={hackerLogo} alt="Rakshak Logo" className="w-full h-full object-cover" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-slate-900 dark:text-white">Rakshak</h1>
            <p className="text-xs text-slate-500 dark:text-slate-400">Anomaly Detection</p>
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
                ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 font-medium shadow-sm dark:shadow-none'
                : 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-200'
              }
            `}
          >
            {({ isActive }) => (
              <>
                <item.icon className={`w-5 h-5 ${isActive ? 'text-primary-600 dark:text-primary-400' : 'text-slate-400 dark:text-slate-500 group-hover:text-slate-600 dark:group-hover:text-slate-300'}`} />
                <span className="font-medium">{item.label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-slate-200 dark:border-slate-700">
        <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-3 border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-xs font-medium text-slate-600 dark:text-slate-300">System Online</span>
          </div>
          <p className="text-xs text-slate-400 dark:text-slate-500">v1.0.0 • Production</p>
        </div>
      </div>
    </aside>
  );
}
