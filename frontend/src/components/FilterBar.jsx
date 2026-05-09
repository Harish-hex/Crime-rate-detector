import { METRIC_LABELS } from "../utils/format";

function FilterSelect({ label, value, onChange, options, disabled, wide }) {
  return (
    <label className={`filter-label${wide ? " filter-label--wide" : ""}`}>
      <span className="filter-title">{label}</span>
      <select
        className="filter-select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function FilterBar({
  filters,
  crimeType,
  setCrimeType,
  range,
  setStartYear,
  setEndYear,
  selectedYear,
  setSelectedYear,
  selectedState,
  setSelectedState,
  disabled,
}) {
  const yearOptions = (filters?.years ?? []).map((y) => ({ value: y, label: String(y) }));
  const crimeOptions = (filters?.crime_types ?? []).map((c) => ({
    value: c,
    label: METRIC_LABELS[c] ?? c,
  }));
  const stateOptions = (filters?.states ?? []).map((s) => ({ value: s, label: s }));

  return (
    <div className="filter-bar">
      <FilterSelect
        label="Crime Type"
        value={crimeType}
        onChange={setCrimeType}
        options={crimeOptions}
        disabled={disabled}
      />
      <FilterSelect
        label="Start Year"
        value={range.startYear}
        onChange={(v) => setStartYear(Number(v))}
        options={yearOptions}
        disabled={disabled}
      />
      <FilterSelect
        label="End Year"
        value={range.endYear}
        onChange={(v) => setEndYear(Number(v))}
        options={yearOptions}
        disabled={disabled}
      />
      <FilterSelect
        label="Map Year"
        value={selectedYear}
        onChange={(v) => setSelectedYear(Number(v))}
        options={yearOptions}
        disabled={disabled}
      />
      <FilterSelect
        label="Prediction State"
        value={selectedState}
        onChange={setSelectedState}
        options={stateOptions}
        disabled={disabled}
        wide
      />
    </div>
  );
}
