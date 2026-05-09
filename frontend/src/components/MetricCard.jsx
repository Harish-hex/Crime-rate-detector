import { MetricCardSkeleton } from "./Skeleton";

export function MetricCard({ title, value, helper, accent, trend, loading }) {
  if (loading) return <MetricCardSkeleton />;

  return (
    <article className="metric-card">
      <span className="metric-accent-bar" style={{ background: accent }} aria-hidden="true" />
      <p className="metric-title">{title}</p>
      <strong className="metric-value">{value}</strong>
      <div className="metric-footer">
        {trend !== undefined && (
          <span className={`metric-trend ${trend >= 0 ? "trend-up" : "trend-down"}`}>
            {trend >= 0 ? (
              <svg width="9" height="9" viewBox="0 0 10 10" fill="currentColor" aria-hidden="true">
                <polygon points="5,1 9,9 1,9" />
              </svg>
            ) : (
              <svg width="9" height="9" viewBox="0 0 10 10" fill="currentColor" aria-hidden="true">
                <polygon points="5,9 9,1 1,1" />
              </svg>
            )}
            {Math.abs(trend).toFixed(1)}%
          </span>
        )}
        <p className="metric-helper">{helper}</p>
      </div>
    </article>
  );
}
