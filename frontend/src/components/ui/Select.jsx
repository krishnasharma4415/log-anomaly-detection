import { ChevronDown } from 'lucide-react';

export default function Select({ 
  label, 
  options = [], 
  error,
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
        <select
          className={`
            w-full px-4 py-2.5 bg-neutral-dark border border-neutral-border rounded-lg
            text-neutral-primary appearance-none cursor-pointer
            focus:border-primary focus:shadow-neon focus:outline-none
            disabled:opacity-50 disabled:cursor-not-allowed
            ${error ? 'border-signal-error' : ''}
            ${className}
          `}
          {...props}
        >
          {options.map((option) => (
            <option 
              key={option.value} 
              value={option.value}
              disabled={option.disabled}
            >
              {option.label}
            </option>
          ))}
        </select>
        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-disabled pointer-events-none" />
      </div>
      {error && (
        <p className="mt-1.5 text-sm text-signal-error">{error}</p>
      )}
    </div>
  );
}
