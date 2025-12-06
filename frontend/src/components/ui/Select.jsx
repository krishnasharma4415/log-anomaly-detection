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
        <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">
          {label}
        </label>
      )}
      <div className="relative">
        <select
          className={`
            w-full px-4 py-2.5 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg
            text-slate-900 dark:text-slate-100 appearance-none cursor-pointer
            focus:border-primary-500 focus:ring-2 focus:ring-primary-100 dark:focus:ring-primary-900 focus:outline-none
            disabled:opacity-50 disabled:cursor-not-allowed
            transition-colors duration-200
            ${error ? 'border-red-500 dark:border-red-500' : ''}
            ${className}
          `}
          {...props}
        >
          {options.map((option) => (
            <option
              key={option.value}
              value={option.value}
              disabled={option.disabled}
              className="bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100"
            >
              {option.label}
            </option>
          ))}
        </select>
        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400 dark:text-slate-500 pointer-events-none" />
      </div>
      {error && (
        <p className="mt-1.5 text-sm text-red-500">{error}</p>
      )}
    </div>
  );
}
