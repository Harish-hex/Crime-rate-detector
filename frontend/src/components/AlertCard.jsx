import { ALERT_COLORS, formatNumber } from "../utils/format";

const SEVERITY_LABEL = {
  critical: "Critical",
  warning:  "Warning",
  watch:    "Watch",
};

export function AlertCard({ alert }) {
  const color = ALERT_COLORS[alert.severity];

  return (
    <article className="alert-card">
      <div className="alert-card-header">
        <span
          className="alert-chip"
          style={{ background: `${color}18`, color }}
        >
          <span
            className="alert-chip-dot"
            style={{ background: color }}
            aria-hidden="true"
          />
          {SEVERITY_LABEL[alert.severity] ?? alert.severity}
        </span>
        <strong className="alert-state">{alert.state}</strong>
      </div>
      <p className="alert-message">{alert.message}</p>
      <div className="alert-meta">
        <div className="alert-stat">
          <span className="alert-stat-label">Previous</span>
          <span className="alert-stat-value">{formatNumber(alert.previous_value)}</span>
        </div>
        <svg
          className="alert-arrow"
          viewBox="0 0 24 24"
          fill="none"
          stroke={color}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M5 12h14M12 5l7 7-7 7" />
        </svg>
        <div className="alert-stat alert-stat--right">
          <span className="alert-stat-label">Latest</span>
          <span className="alert-stat-value" style={{ color }}>{formatNumber(alert.latest_value)}</span>
        </div>
      </div>
    </article>
  );
}
