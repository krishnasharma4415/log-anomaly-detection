export const apiService = {
  checkHealth: async () => {
    try {
      const response = await fetch('http://localhost:5000/health');
      return response.ok ? 'healthy' : 'unhealthy';
    } catch (err) {
      return 'offline';
    }
  },

  analyzeLog: async (logText) => {
    const response = await fetch('http://localhost:5000/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ log_text: logText })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Analysis failed');
    }

    return response.json();
  }
};