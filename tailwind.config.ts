import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}', './lib/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#0B1020',
        amd: '#00C853',
        cobalt: '#3457D5'
      },
      boxShadow: {
        glow: '0 0 80px rgba(0, 200, 83, 0.20)'
      }
    }
  },
  plugins: [require('@tailwindcss/forms')]
};

export default config;
