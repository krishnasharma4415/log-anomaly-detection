import { Search, Bell, User, Sun, Moon } from 'lucide-react';
import Badge from '../ui/Badge';
import { useModel } from '../../context/ModelContext';
import { useTheme } from '../../context/ThemeContext';
import hackerLogo from '../../hacker.png';

export default function TopBar() {
  const { activeModel } = useModel();
  const { isDarkMode, toggleTheme } = useTheme();

  return (
    <header className="sticky top-0 z-20 bg-slate-900 border-b border-slate-700 shadow-lg">
      <div className="flex items-center justify-between px-6 py-3">
        {/* Logo - Left Section */}
        <div className="flex items-center gap-3 min-w-[200px]">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center shadow-sm overflow-hidden">
            <img src={hackerLogo} alt="Rakshak Logo" className="w-full h-full object-cover" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white">Rakshak</h1>
            <p className="text-xs text-slate-400">Anomaly Detection</p>
          </div>
        </div>

        {/* Search - Center Section */}
        <div className="flex-1 max-w-xl mx-6">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input
              type="text"
              placeholder="Search logs, models, or documentation..."
              className="w-full pl-10 pr-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-100 placeholder:text-slate-500 focus:border-primary-500 focus:ring-2 focus:ring-primary-900 focus:outline-none transition-all text-sm"
            />
          </div>
        </div>

        {/* Right Section */}
        <div className="flex items-center gap-3">
          {/* Model Status */}
          <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 rounded-lg border border-slate-700">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-sm text-slate-200 font-medium">{activeModel.name} Active</span>
          </div>

          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition-colors"
            aria-label="Toggle theme"
          >
            {isDarkMode ? (
              <Sun className="w-5 h-5" />
            ) : (
              <Moon className="w-5 h-5" />
            )}
          </button>

          {/* Notifications */}
          <button className="relative p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition-colors">
            <Bell className="w-5 h-5" />
            <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
          </button>

          {/* User */}
          <button className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 rounded-lg border border-slate-700 hover:border-primary-600 hover:bg-slate-700 transition-colors">
            <div className="w-7 h-7 rounded-full bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center shadow-sm">
              <User className="w-4 h-4 text-white" />
            </div>
            <span className="text-sm text-slate-200 font-medium">Admin</span>
          </button>
        </div>
      </div>
    </header>
  );
}
