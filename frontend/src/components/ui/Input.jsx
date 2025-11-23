export default function Input({ 
  label, 
  error, 
  icon: Icon,
  className = '',
  ...props 
}) {
  return (
    <div className="w-full">
      {label && (
        <label className="block text-sm font-medium text-neutral-secondary mb-2">
          {label}
        </label>
      )}
      <div className="relative">
        {Icon && (
          <div className="absolute left-3 top-1/2 -translate-y-1/2 text-neutral-disabled">
            <Icon className="w-5 h-5" />
          </div>
        )}
        <input
          className={`
            w-full px-4 py-2.5 bg-neutral-dark border border-neutral-border rounded-lg
            text-neutral-primary placeholder:text-neutral-disabled
            focus:border-primary focus:shadow-neon focus:outline-none
            disabled:opacity-50 disabled:cursor-not-allowed
            ${Icon ? 'pl-11' : ''}
            ${error ? 'border-signal-error' : ''}
            ${className}
          `}
          {...props}
        />
      </div>
      {error && (
        <p className="mt-1.5 text-sm text-signal-error">{error}</p>
      )}
    </div>
  );
}

export function TextArea({ 
  label, 
  error, 
  className = '',
  rows = 4,
  ...props 
}) {
  return (
    <div className="w-full">
      {label && (
        <label className="block text-sm font-medium text-neutral-secondary mb-2">
          {label}
        </label>
      )}
      <textarea
        rows={rows}
        className={`
          w-full px-4 py-2.5 bg-neutral-dark border border-neutral-border rounded-lg
          text-neutral-primary placeholder:text-neutral-disabled font-mono text-sm
          focus:border-primary focus:shadow-neon focus:outline-none
          disabled:opacity-50 disabled:cursor-not-allowed resize-none
          ${error ? 'border-signal-error' : ''}
          ${className}
        `}
        {...props}
      />
      {error && (
        <p className="mt-1.5 text-sm text-signal-error">{error}</p>
      )}
    </div>
  );
}
