import { CircleMarker, MapContainer, Popup, TileLayer } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { formatNumber, mapRadius, METRIC_LABELS } from "../utils/format";

export function CrimeMap({ mapData, crimeType }) {
  return (
    <MapContainer center={[22.5, 80]} zoom={4.5} scrollWheelZoom className="leaflet-map">
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
      />
      {mapData?.points.map((point) => (
        <CircleMarker
          key={point.state}
          center={[point.latitude, point.longitude]}
          radius={mapRadius(point.value)}
          pathOptions={{
            color: point.risk_color,
            fillColor: point.risk_color,
            fillOpacity: 0.65,
            weight: 1.5,
          }}
        >
          <Popup className="crime-popup">
            <div className="popup-content">
              <strong className="popup-state">{point.state}</strong>
              <div className="popup-row">
                <span>{METRIC_LABELS[crimeType] ?? crimeType}</span>
                <span>{formatNumber(point.value)}</span>
              </div>
              <div className="popup-row">
                <span>Risk level</span>
                <span style={{ color: point.risk_color }}>{point.risk}</span>
              </div>
              <div className="popup-row">
                <span>Forecast {point.forecast_year}</span>
                <span>{formatNumber(point.forecast_value)}</span>
              </div>
              <div className="popup-row">
                <span>Confidence</span>
                <span>{point.confidence}%</span>
              </div>
            </div>
          </Popup>
        </CircleMarker>
      ))}
    </MapContainer>
  );
}
