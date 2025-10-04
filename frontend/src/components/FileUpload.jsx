import React from 'react';
import { Upload } from 'lucide-react';

export default function FileUpload({ onFileSelect, fileName }) {
  return (
    <div className="mb-4">
      <label className="block text-purple-200 mb-2 text-sm font-medium">
        Upload Log File
      </label>
      <div className="relative">
        <input
          type="file"
          accept=".log,.txt"
          onChange={onFileSelect}
          className="hidden"
          id="file-upload"
        />
        <label
          htmlFor="file-upload"
          className="flex items-center justify-center gap-2 w-full px-4 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg cursor-pointer transition-colors"
        >
          <Upload className="w-5 h-5" />
          {fileName || 'Choose File'}
        </label>
      </div>
    </div>
  );
}