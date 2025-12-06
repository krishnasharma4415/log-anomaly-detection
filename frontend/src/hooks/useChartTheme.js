import { useTheme } from '../context/ThemeContext';

/**
 * Custom hook to provide theme-aware chart styling for Recharts
 * Returns colors and styles that adapt to current theme (light/dark)
 */
export function useChartTheme() {
    const { isDarkMode } = useTheme();

    const tooltipStyle = {
        backgroundColor: isDarkMode ? '#1E293B' : '#FFFFFF',
        border: isDarkMode ? '1px solid #334155' : '1px solid #E2E8F0',
        borderRadius: '8px',
        color: isDarkMode ? '#F1F5F9' : '#0F172A',
        boxShadow: isDarkMode
            ? '0 4px 6px rgba(0, 0, 0, 0.3)'
            : '0 4px 6px rgba(0, 0, 0, 0.1)',
    };

    const gridColor = isDarkMode ? '#334155' : '#E2E8F0';
    const axisColor = isDarkMode ? '#94A3B8' : '#64748B';
    const textColor = isDarkMode ? '#F1F5F9' : '#0F172A';

    return {
        tooltipStyle,
        gridColor,
        axisColor,
        textColor,
        isDarkMode,
    };
}
