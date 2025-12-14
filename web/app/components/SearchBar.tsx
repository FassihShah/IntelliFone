import React from 'react';

export default function SearchBar({ onSearch, placeholder = "Search phones..." }: { onSearch: (q: string) => void; placeholder?: string }) {
  const [query, setQuery] = React.useState('');

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    onSearch(query);
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-3 w-full max-w-2xl mx-auto">
      <div className="flex-1 relative">
        <span className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400">🔎</span>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={placeholder}
          className="w-full pl-12 pr-4 py-3 rounded-xl glass-panel text-white placeholder-gray-400 border border-gray-500 "
        />
      </div>
      <button 
        type="submit"
        className="px-8 py-3 rounded-xl text-black font-semibold flex items-center justify-center gap-2 disabled:opacity-50 transition neon-glow"
         style={{
            backgroundColor: "#f7f434",
          }}

      >
        Search
      </button>
    </form>
  );
}