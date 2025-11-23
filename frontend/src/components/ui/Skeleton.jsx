export default function Skeleton({ className = '', variant = 'default' }) {
  const variants = {
    default: 'h-4 w-full',
    text: 'h-4 w-3/4',
    title: 'h-8 w-1/2',
    circle: 'h-12 w-12 rounded-full',
    card: 'h-32 w-full',
  };

  return (
    <div
      className={`
        bg-neutral-surface rounded shimmer
        ${variants[variant]} ${className}
      `}
    />
  );
}

export function SkeletonCard() {
  return (
    <div className="bg-neutral-surface rounded-xl p-6 border border-neutral-border">
      <Skeleton variant="title" className="mb-4" />
      <Skeleton variant="text" className="mb-2" />
      <Skeleton variant="text" className="mb-2" />
      <Skeleton variant="default" />
    </div>
  );
}

export function SkeletonTable({ rows = 5 }) {
  return (
    <div className="space-y-3">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="flex gap-4">
          <Skeleton className="w-1/4" />
          <Skeleton className="w-1/2" />
          <Skeleton className="w-1/4" />
        </div>
      ))}
    </div>
  );
}
