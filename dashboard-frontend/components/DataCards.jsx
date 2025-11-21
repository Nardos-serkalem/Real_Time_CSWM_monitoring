export default function DataCards({ status }) {
  const items = [
    { label: "Kp Index", value: status.kpIndex, icon: "⚡" },
    { label: "Solar Flux", value: status.solarFlux, icon: "☀️" },
    { label: "Proton Flux", value: status.protonFlux, icon: "⚛️" },
    { label: "Magnetic Field", value: status.magneticField + " nT", icon: "🧲" }
  ];

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Live Data Feed Active</h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {items.map((item, idx) => (
          <div key={idx} className="p-5 bg-[#1c2233] border border-gray-700 rounded-xl shadow">
            <div className="text-4xl mb-2">{item.icon}</div>
            <p className="font-semibold">{item.label}</p>
            <p className="text-2xl">{item.value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
