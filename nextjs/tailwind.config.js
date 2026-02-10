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
          light: '#F5F7FA',
          gray: '#E8ECF0',
          accent: '#00C853',
          warning: '#FF9800',
          error: '#F44336',
          white: '#FFFFFF',
          slate: '#64748B',
        },
        // 기존 cookie-* 클래스 → cafe24 매핑 (호환성)
        cookie: {
          primary: '#5B9BF5',
          secondary: '#4A8AE5',
          accent: '#00C853',
          success: '#00C853',
          warning: '#FF9800',
          error: '#F44336',
          info: '#7CB9F7',
          yellow: '#7CB9F7',
          orange: '#5B9BF5',
          brown: '#1A1A2E',
          cream: '#F5F7FA',
          beige: '#E8ECF0',
          light: '#F8FAFC',
        },
        // 셀러 등급별 컬러
        grade: {
          common: '#9CA3AF',
          rare: '#42A5F5',
          superrare: '#7C3AED',
          epic: '#AB47BC',
          legendary: '#1B6FF0',
          ancient: '#0D47A1',
        },
      },
      fontFamily: {
        sans: ['Pretendard', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Noto Sans KR', 'sans-serif'],
      },
      backgroundImage: {
        'cafe24-gradient': 'linear-gradient(135deg, #7CB9F7 0%, #5B9BF5 100%)',
        'cookie-gradient': 'linear-gradient(135deg, #7CB9F7 0%, #5B9BF5 100%)',
        'dark-gradient': 'linear-gradient(135deg, #1A1A2E 0%, #16213E 100%)',
      },
      boxShadow: {
        'cafe24': '0 4px 12px 0 rgba(91, 155, 245, 0.18)',
        'cafe24-lg': '0 8px 24px -3px rgba(91, 155, 245, 0.22)',
        'cookie': '0 4px 12px 0 rgba(91, 155, 245, 0.18)',
        'cookie-lg': '0 8px 24px -3px rgba(91, 155, 245, 0.22)',
        'soft': '0 2px 8px rgba(26, 26, 46, 0.05)',
        'soft-lg': '0 8px 24px rgba(26, 26, 46, 0.08)',
      },
      animation: {
        'bounce-slow': 'bounce 3s infinite',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-in': 'slideIn 0.25s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideIn: {
          '0%': { opacity: '0', transform: 'translateX(-10px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
      },
      transitionTimingFunction: {
        'smooth': 'cubic-bezier(0.4, 0, 0.2, 1)',
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
};
