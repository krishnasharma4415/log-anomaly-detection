const variants = {
  primary: 'bg-gradient-primary text-white hover:shadow-neon-hover',
  secondary: 'bg-neutral-surface text-neutral-primary border border-neutral-border hover:border-primary hover:shadow-neon',
  ghost: 'bg-transparent text-neutral-secondary hover:bg-neutral-surface hover:text-neutral-primary',
  danger: 'bg-signal-error text-white hover:bg-red-600',
  success: 'bg-signal-success text-white hover:bg-green-500',
};

const sizes = {
  sm: 'px-3 py-1.5 text-sm',
  md: 'px-4 py-2 text-base',
  lg: 'px-6 py-3 text-lg',
};

export default function Button({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  className = '', 
  disabled = false,
  loading = false,
  icon: Icon,
  ...props 
}) {
  return (
    <button
      className={`
        inline-flex items-center justify-center gap-2 rounded-lg font-medium
        transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed
        ${variants[variant]} ${sizes[size]} ${className}
      `}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
      ) : Icon ? (
        <Icon className="w-4 h-4" />
      ) : null}
      {children}
    </button>
  );
}
