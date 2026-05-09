import { ErrorBoundary } from "./ErrorBoundary";

export function Panel({ title, subtitle, badge, children, loading, className = "" }) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-header">
        <div className="panel-header-text">
          <h2 className="panel-title">{title}</h2>
          {subtitle && <p className="panel-subtitle">{subtitle}</p>}
        </div>
        {badge && <span className="pill">{badge}</span>}
        {loading && <span className="panel-loading-dot" />}
      </div>
      <ErrorBoundary>{children}</ErrorBoundary>
    </section>
  );
}
