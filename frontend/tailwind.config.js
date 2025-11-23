export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Cyber Indigo Palette
        primary: {
          DEFAULT: '#4B5DFF',
          50: '#F0F2FF',
          100: '#E0E4FF',
          200: '#C7CDFF',
          300: '#A5AFFF',
          400: '#7D8AFF',
          500: '#4B5DFF',
          600: '#4B5DFF',
          700: '#2C318A',
          800: '#1F2461',
          900: '#131740',
        },
        indigo: {
          600: '#4B5DFF',
          700: '#2C318A',
          900: '#131740',
        },
        slate: {
          900: '#0F111A',
        },
        accent: {
          blue: '#00C6FF',
          cyan: '#22E1FF',
          purple: '#A259FF',
        },
        neutral: {
          primary: '#FFFFFF',
          secondary: '#C5C7D3',
          disabled: '#7C7F92',
          surface: '#2A2D3E',
          dark: '#0F111A',
          border: '#3B3F52',
        },
        signal: {
          success: '#34D399',
          warning: '#FBBF24',
          error: '#F43F5E',
        },
      },
      fontFamily: {
        sans: ['Manrope', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      boxShadow: {
        'card': '0px 4px 16px rgba(0, 0, 0, 0.3)',
        'neon': '0px 0px 14px rgba(75, 93, 255, 0.6)',
        'neon-hover': '0px 0px 20px rgba(0, 198, 255, 0.5)',
        'neon-cyan': '0px 0px 16px rgba(34, 225, 255, 0.4)',
        'neon-purple': '0px 0px 16px rgba(162, 89, 255, 0.4)',
      },
      backgroundImage: {
        'gradient-primary': 'linear-gradient(135deg, #4B5DFF 0%, #A259FF 100%)',
        'gradient-accent': 'linear-gradient(135deg, #00C6FF 0%, #22E1FF 100%)',
        'gradient-hero': 'linear-gradient(135deg, rgba(75, 93, 255, 0.25) 0%, rgba(162, 89, 255, 0.20) 100%)',
      },
      animation: {
        'pulse-neon': 'pulse-neon 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'shimmer': 'shimmer 2s linear infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        'pulse-neon': {
          '0%, 100%': { opacity: '1', boxShadow: '0px 0px 14px rgba(75, 93, 255, 0.6)' },
          '50%': { opacity: '0.8', boxShadow: '0px 0px 20px rgba(75, 93, 255, 0.8)' },
        },
        'shimmer': {
          '0%': { backgroundPosition: '-1000px 0' },
          '100%': { backgroundPosition: '1000px 0' },
        },
        'glow': {
          'from': { boxShadow: '0 0 10px rgba(75, 93, 255, 0.5)' },
          'to': { boxShadow: '0 0 20px rgba(75, 93, 255, 0.8)' },
        },
      },
    },
  },
  plugins: [],
}