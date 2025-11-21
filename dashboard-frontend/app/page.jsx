"use client";

import { useEffect, useState } from "react";
import Header from "../components/Header";
import SystemStatus from "../components/SystemStatus";
import DataCards from "../components/DataCards";
import PlotViewer from "../components/PlotViewer";

export default function Dashboard() {
  const [plots, setPlots] = useState([]);
  const [status, setStatus] = useState({
    system: "Operational",
    kpIndex: 0,
    solarFlux: 0,
    protonFlux: 0,
    magneticField: 0,
  });

  useEffect(() => {
    fetch("http://127.0.0.1:5000/plots")
      .then((res) => res.json())
      .then((data) => setPlots(data.plots))
      .catch(() => setPlots([]));

    fetch("http://127.0.0.1:5000/status")
      .then((res) => res.json())
      .then((data) => setStatus(data))
      .catch(() => {});
  }, []);

  return (
    <div className="min-h-screen bg-[#0B0F19] text-white p-6">
      <Header />
      <SystemStatus status={status} />
      <DataCards status={status} />
      <PlotViewer plots={plots} />
    </div>
  );
}
