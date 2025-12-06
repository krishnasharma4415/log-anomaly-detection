import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ToastProvider } from './components/ui/Toast';
import { ModelProvider } from './context/ModelContext';
import { ThemeProvider } from './context/ThemeContext';
import Layout from './components/layout/Layout';
import Dashboard from './pages/Dashboard';
import LogAnalyzer from './pages/LogAnalyzer';
import BatchAnalysis from './pages/BatchAnalysis';
import ModelExplorer from './pages/ModelExplorer';
import Visualizations from './pages/Visualizations';
import SystemHealth from './pages/SystemHealth';
import Settings from './pages/Settings';

export default function App() {
  return (
    <ThemeProvider>
      <ToastProvider>
        <ModelProvider>
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<Layout />}>
                <Route index element={<Dashboard />} />
                <Route path="analyzer" element={<LogAnalyzer />} />
                <Route path="batch" element={<BatchAnalysis />} />
                <Route path="models" element={<ModelExplorer />} />
                <Route path="visualizations" element={<Visualizations />} />
                <Route path="health" element={<SystemHealth />} />
                <Route path="settings" element={<Settings />} />
              </Route>
            </Routes>
          </BrowserRouter>
        </ModelProvider>
      </ToastProvider>
    </ThemeProvider>
  );
}
