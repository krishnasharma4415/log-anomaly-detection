import { Search, Bell, User } from 'lucide-react';
import Badge from '../ui/Badge';

export default function TopBar() {
  return (
    <header className="sticky top-0 z-20 bg-neutral-dark/80 backdrop-blur-lg border-b border-neutral-border">
      <div className="flex items-center justify-between px-8 py-4">
        {/* Search */}
        <div className="flex-1 max-w-xl">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-disabled" />
            <input
              type="text"
              placeholder="Search logs, models, or documentation..."
              className="w-full pl-11 pr-4 py-2.5 bg-neutral-surface border border-neutral-border rounded-lg text-neutral-primary placeholder:text-neutral-disabled focus:border-primary focus:shadow-neon focus:outline-none"
            />
          </div>
        </div>

        {/* Right Section */}
        <div className="flex items-center gap-4">
          {/* Model Status */}
          <div className="flex items-center gap-2 px-3 py-2 bg-neutral-surface rounded-lg border border-neutral-border">
            <div className="w-2 h-2 rounded-full bg-signal-success animate-pulse" />
            <span className="text-sm text-neutral-secondary">XGBoost Active</span>
          </div>

          {/* Notifications */}
          <button className="relative p-2 text-neutral-secondary hover:text-neutral-primary transition-colors">
            <Bell className="w-5 h-5" />
            <span className="absolute top-1 right-1 w-2 h-2 bg-signal-error rounded-full" />
          </button>

          {/* User */}
          <button className="flex items-center gap-2 px-3 py-2 bg-neutral-surface rounded-lg border border-neutral-border hover:border-primary transition-colors">
            <div className="w-8 h-8 rounded-full bg-gradient-primary flex items-center justify-center">
              <User className="w-4 h-4 text-white" />
            </div>
            <span className="text-sm text-neutral-primary">Admin</span>
          </button>
        </div>
      </div>
    </header>
  );
}
