/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,jsx}',
    './components/**/*.{js,jsx}',
  ],
  theme: {
    extend: {
      colors: {
        // CAFE24 브랜드 컬러
        cafe24: {
          blue: '#5B9BF5',
          navy: '#4A8AE5',
          dark: '#1A1A2E',
          light: '#F8FAFC',
          gray: '#E8ECF0',
          accent: '#00C853',
          warning: '#FF9800',
          error: '#F44336',
          white: '#FFFFFF',
          slate: '#64748B',
          yellow: '#7CB9F7',
          orange: '#5B9BF5',
          brown: '#1A1A2E',
          cream: '#F5F7FA',
          beige: '#E8ECF0',
          pink: '#F472B6',
        },
      },
      fontFamily: {
        sans: ['Pretendard', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Noto Sans KR', 'sans-serif'],
      },
      boxShadow: {
        'cafe24': '0 4px 12px 0 rgba(91, 155, 245, 0.18)',
        'cafe24-lg': '0 8px 24px -3px rgba(91, 155, 245, 0.22)',
        'cafe24-sm': '0 4px 12px rgba(91, 155, 245, 0.15)',
        'soft': '0 2px 8px rgba(26, 26, 46, 0.05)',
        'soft-lg': '0 8px 24px rgba(26, 26, 46, 0.08)',
      },
      transitionTimingFunction: {
        'smooth': 'cubic-bezier(0.4, 0, 0.2, 1)',
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
};
