import { createContext, useContext, useState, useEffect } from 'react';

const ThemeContext = createContext();

export function ThemeProvider({ children }) {
    const [isDarkMode, setIsDarkMode] = useState(() => {
        // Check localStorage first, default to light mode
        const saved = localStorage.getItem('theme');
        return saved === 'dark';
    });

    useEffect(() => {
        // Update localStorage and html class when theme changes
        localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');

        console.log('🎨 Theme changed to:', isDarkMode ? 'DARK' : 'LIGHT');
        console.log('📄 HTML classes before:', document.documentElement.className);

        if (isDarkMode) {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }

        console.log('📄 HTML classes after:', document.documentElement.className);
    }, [isDarkMode]);

    const toggleTheme = () => {
        console.log('🔄 Toggle theme clicked!');
        setIsDarkMode(prev => !prev);
    };

    return (
        <ThemeContext.Provider value={{ isDarkMode, toggleTheme }}>
            {children}
        </ThemeContext.Provider>
    );
}

export function useTheme() {
    const context = useContext(ThemeContext);
    if (!context) {
        throw new Error('useTheme must be used within ThemeProvider');
    }
    return context;
}
