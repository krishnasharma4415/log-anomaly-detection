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
        <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">
          {label}
        </label>
      )}
      <div className="relative">
        {Icon && (
          <div className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 dark:text-slate-500">
            <Icon className="w-5 h-5" />
          </div>
        )}
        <input
          className={`
            w-full px-4 py-2.5 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg
            text-slate-900 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-500
            focus:border-primary-500 focus:ring-2 focus:ring-primary-100 dark:focus:ring-primary-900 focus:outline-none
            disabled:opacity-50 disabled:cursor-not-allowed
            transition-colors duration-200
            ${Icon ? 'pl-11' : ''}
            ${error ? 'border-red-500 dark:border-red-500' : ''}
            ${className}
          `}
          {...props}
        />
      </div>
      {error && (
        <p className="mt-1.5 text-sm text-red-500">{error}</p>
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
        <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">
          {label}
        </label>
      )}
      <textarea
        rows={rows}
        className={`
          w-full px-4 py-2.5 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg
          text-slate-900 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-500 font-mono text-sm
          focus:border-primary-500 focus:ring-2 focus:ring-primary-100 dark:focus:ring-primary-900 focus:outline-none
          disabled:opacity-50 disabled:cursor-not-allowed resize-none
          transition-colors duration-200
          ${error ? 'border-red-500 dark:border-red-500' : ''}
          ${className}
        `}
        {...props}
      />
      {error && (
        <p className="mt-1.5 text-sm text-red-500">{error}</p>
      )}
    </div>
  );
}
