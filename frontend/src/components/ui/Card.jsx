export default function Card({ 
  children, 
  className = '', 
  neon = false,
  ...props 
}) {
  return (
    <div
      className={`
        bg-neutral-surface rounded-xl p-6 shadow-card
        ${neon ? 'border border-primary/30 shadow-neon' : 'border border-neutral-border'}
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
          <div className="text-neutral-secondary text-sm font-medium">{title}</div>
          {Icon && (
            <div className={`p-2 rounded-lg ${neon ? 'bg-primary/10' : 'bg-neutral-dark'}`}>
              <Icon className={`w-5 h-5 ${neon ? 'text-primary' : 'text-neutral-secondary'}`} />
            </div>
          )}
        </div>
        <div className="text-3xl font-bold text-neutral-primary mb-1">{value}</div>
        {subtitle && (
          <div className="flex items-center gap-2 text-sm">
            <span className="text-neutral-secondary">{subtitle}</span>
            {trend && (
              <span className={trend > 0 ? 'text-signal-success' : 'text-signal-error'}>
                {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}%
              </span>
            )}
          </div>
        )}
      </div>
    </Card>
  );
}
