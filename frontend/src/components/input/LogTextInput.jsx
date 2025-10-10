import React from 'react';

export default function LogTextInput({ value, onChange }) {
  return (
    <div className="mb-4">
      <label className="block text-purple-200 mb-2 text-sm font-medium">
        Paste Log Content
      </label>
      <textarea
        value={value}
        onChange={onChange}
        placeholder="Enter system log here..."
        className="w-full h-48 px-4 py-3 bg-slate-800/50 text-white rounded-lg border border-purple-500/30 focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 outline-none resize-none font-mono text-sm"
      />
    </div>
  );
}