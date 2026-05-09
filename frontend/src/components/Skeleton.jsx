export function Skeleton({ className = "" }) {
  return <div className={`skeleton ${className}`} />;
}

export function MetricCardSkeleton() {
  return (
    <div className="metric-card">
      <span className="metric-accent skeleton-bar" />
      <Skeleton className="h-3 w-24 mt-3" />
      <Skeleton className="h-8 w-32 mt-3" />
      <Skeleton className="h-3 w-40 mt-2" />
    </div>
  );
}

export function ChartSkeleton({ height = 320 }) {
  return (
    <div className="chart-skeleton" style={{ height }}>
      <div className="chart-skeleton-bars">
        {Array.from({ length: 8 }).map((_, i) => (
          <div
            key={i}
            className="chart-skeleton-bar skeleton"
            style={{ height: `${30 + Math.sin(i * 0.8) * 40 + 40}%` }}
          />
        ))}
      </div>
    </div>
  );
}
