'use client';

import { useEffect, useState } from 'react';

export default function Home() {
  const [plots, setPlots] = useState([]); // remove <string[]>
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchPlots = async () => {
      try {
        const res = await fetch('http://127.0.0.1:5000/plots');
        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const data = await res.json(); // remove : string[]
        setPlots(data);
      } catch (err) {
        console.error('Error fetching plots:', err);
        setError(err.message || 'Unknown error'); // remove : any
      } finally {
        setLoading(false);
      }
    };
    fetchPlots();
  }, []);

  if (loading) return <p style={{ color: '#fff' }}>Loading plots...</p>;
  if (error) return <p style={{ color: 'red' }}>Error: {error}</p>;

  return (
    <div style={{ padding: '2rem', background: '#111', color: '#fff', minHeight: '100vh' }}>
      <h1>Space Weather Real Time Monitering</h1>

      {plots.length === 0 ? (
        <p>No plots available.</p>
      ) : (
        plots.map((plot) => (
          <div key={plot} style={{ marginBottom: '2rem' }}>
            <h3>{plot}</h3>
            <img
              src={`http://127.0.0.1:5000/plots/${plot}`}
              alt={plot}
              style={{ maxWidth: '100%', borderRadius: '8px' }}
            />
          </div>
        ))
      )}
    </div>
  );
}
