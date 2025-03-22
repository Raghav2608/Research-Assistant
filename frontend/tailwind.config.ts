import type { Config } from "tailwindcss";

export default {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        lightest: '#697688', 
        lighter: '#525f6e',  
        light: '#3a4755',
        dark: '#000100',
        primary: '#c0a2b5',
        secondary: '#d74c4c',
        info: '#598da8',
        success: '#40b581',
        warning: '#dcd439',
        danger: '#e74420',
      }
    },
  },
  plugins: [],
} satisfies Config;
