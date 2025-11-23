const variants = {
  success: 'bg-signal-success/10 text-signal-success border-signal-success/30',
  warning: 'bg-signal-warning/10 text-signal-warning border-signal-warning/30',
  error: 'bg-signal-error/10 text-signal-error border-signal-error/30',
  info: 'bg-accent-blue/10 text-accent-blue border-accent-blue/30',
  primary: 'bg-primary/10 text-primary border-primary/30',
  neutral: 'bg-neutral-surface text-neutral-secondary border-neutral-border',
};

export default function Badge({ 
  children, 
  variant = 'neutral', 
  pulse = false,
  className = '' 
}) {
  return (
    <span
      className={`
        inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium
        border ${variants[variant]} ${className}
      `}
    >
      {pulse && (
        <span className="relative flex h-2 w-2">
          <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${variant === 'success' ? 'bg-signal-success' : variant === 'error' ? 'bg-signal-error' : 'bg-primary'}`}></span>
          <span className={`relative inline-flex rounded-full h-2 w-2 ${variant === 'success' ? 'bg-signal-success' : variant === 'error' ? 'bg-signal-error' : 'bg-primary'}`}></span>
        </span>
      )}
      {children}
    </span>
  );
}
