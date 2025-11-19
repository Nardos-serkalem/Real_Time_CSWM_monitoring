"use client";

import { useEffect, useState } from "react";

export default function Dashboard() {
  const [plots, setPlots] = useState([]);
  const [status, setStatus] = useState({
    system: "Operational",
    lastUpdate: "",
    kpIndex: 0,
    solarFlux: 0,
    protonFlux: 0,
    magneticField: 0,
  });

  useEffect(() => {
    // Fetch plot images from backend
    fetch("http://127.0.0.1:5000/plots")
      .then((res) => res.json())
      .then((data) => {
        // Ensure data.plots exists and is an array
        if (data && Array.isArray(data.plots)) {
          setPlots(data.plots);
        } else {
          console.error("Backend returned invalid plots:", data);
          setPlots([]);
        }
      })
      .catch((err) => {
        console.error("Error fetching plots:", err);
        setPlots([]);
      });

    // Fetch system status
    fetch("http://127.0.0.1:5000/status")
      .then((res) => res.json())
      .then((data) => setStatus(data))
      .catch((err) => console.error("Error fetching status:", err));
  }, []);

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      {/* Header */}
      <header className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Real-time Monitoring
        </h1>
        <p className="text-gray-700 mb-1">
          Ethiopia Space Weather Early Warning System
        </p>
        <p className="text-gray-700">
          System Status: <span className="font-semibold">{status.system}</span>
          &nbsp;| Last Update: {status.lastUpdate}
        </p>
      </header>

      {/* Alerts / Quick Info */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-red-100 text-red-800 p-4 rounded shadow">
          <p className="text-sm">Kp Index</p>
          <p className="text-xl font-bold">{status.kpIndex}</p>
        </div>
        <div className="bg-blue-100 text-blue-800 p-4 rounded shadow">
          <p className="text-sm">S4_Pi</p>
          <p className="text-xl font-bold">{status.solarFlux}</p>
        </div>
        <div className="bg-yellow-100 text-yellow-800 p-4 rounded shadow">
          <p className="text-sm">VTEC</p>
          <p className="text-xl font-bold">{status.protonFlux}</p>
        </div>
        <div className="bg-green-100 text-green-800 p-4 rounded shadow">
          <p className="text-sm">Magnetic Field</p>
          <p className="text-xl font-bold">{status.magneticField} nT</p>
        </div>
      </div>

      {/* Main Plots */}
      {Array.isArray(plots) && plots.length > 0 ? (
        plots.map((plot, idx) => (
          <div key={idx} className="mb-6">
            <img
              src={`http://127.0.0.1:5000/plots/${plot}`}
              alt={plot}
              className="w-full rounded shadow-lg"
            />
          </div>
        ))
      ) : (
        <p className="text-gray-500">No plots available</p>
      )}
    </div>
  );
}
