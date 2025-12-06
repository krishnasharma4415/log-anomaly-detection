export default function Card({
  children,
  className = '',
  neon = false,
  ...props
}) {
  return (
    <div
      className={`
        bg-white dark:bg-slate-800 rounded-xl p-6 shadow-card
        ${neon ? 'border border-primary/30 shadow-neon' : 'border border-slate-200 dark:border-slate-700'}
        transition-colors duration-200
        ${className}
      `}
      {...props}
    >
      {children}
    </div>
  );
}

export function MetricCard({ title, value, subtitle, icon: Icon, trend, neon = false }) {
  return (
    <Card neon={neon} className="relative overflow-hidden">
      {neon && (
        <div className="absolute inset-0 bg-gradient-primary opacity-5" />
      )}
      <div className="relative z-10">
        <div className="flex items-start justify-between mb-3">
          <div className="text-slate-600 dark:text-slate-400 text-sm font-medium">{title}</div>
          {Icon && (
            <div className={`p-2 rounded-lg ${neon ? 'bg-primary/10' : 'bg-slate-100 dark:bg-slate-700'}`}>
              <Icon className={`w-5 h-5 ${neon ? 'text-primary' : 'text-slate-600 dark:text-slate-400'}`} />
            </div>
          )}
        </div>
        <div className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-1">{value}</div>
        {subtitle && (
          <div className="flex items-center gap-2 text-sm">
            <span className="text-slate-600 dark:text-slate-400">{subtitle}</span>
            {trend && (
              <span className={trend > 0 ? 'text-green-500' : 'text-red-500'}>
                {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}%
              </span>
            )}
          </div>
        )}
      </div>
    </Card>
  );
}
