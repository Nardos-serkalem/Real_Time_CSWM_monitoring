export default function PlotViewer({ plots }) {
  return (
    <div className="mt-10">
      <h2 className="text-xl font-semibold mb-4">Visual Data</h2>

      {plots.length === 0 ? (
        <p>No plots available</p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {plots.map((file, i) => (
            <div key={i} className="p-4 bg-[#1c2233] border border-gray-700 rounded-xl shadow">
              <img src={`/${file}`} className="w-full border rounded-lg" />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
