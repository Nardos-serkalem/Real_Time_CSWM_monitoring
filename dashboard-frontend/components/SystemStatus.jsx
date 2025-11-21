export default function SystemStatus({ status }) {
  return (
    <div className="bg-[#111725] border border-gray-700 rounded-xl p-5 mb-10 text-white">
      <div className="flex gap-4 items-center">
        <span className="w-3 h-3 rounded-full bg-green-400"></span>
        <p className="text-gray-300">System Status:</p>
        <span className="font-semibold">{status.system}</span>
      </div>

      <p className="mt-2 text-gray-400">
        Data Sources: 5/5 Active | Stations: 4/5 Online | Alerts: None
      </p>
    </div>
  );
}
