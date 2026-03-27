import { create } from 'zustand';

export type Theme = 'dark' | 'light';

interface ThemeColors {
  bg: string;
  bgSurface: string;
  bgCanvas: string;
  text: string;
  textSecondary: string;
  border: string;
  nodeBg: number; // opacity multiplier for node pill backgrounds
  nodeText: string;
  linkAlphaMultiplier: number;
}

export const THEMES: Record<Theme, ThemeColors> = {
  dark: {
    bg: '#050510',
    bgSurface: 'rgba(13, 17, 23, 0.8)',
    bgCanvas: '#0a0a1a',
    text: '#e6edf3',
    textSecondary: '#8b949e',
    border: 'rgba(0, 212, 255, 0.12)',
    nodeBg: 0.85,
    nodeText: '#ffffff',
    linkAlphaMultiplier: 1,
  },
  light: {
    bg: '#f0f2f5',
    bgSurface: 'rgba(255, 255, 255, 0.9)',
    bgCanvas: '#e8ecf1',
    text: '#1a1a2e',
    textSecondary: '#5a6070',
    border: 'rgba(0, 0, 0, 0.1)',
    nodeBg: 0.95,
    nodeText: '#ffffff',
    linkAlphaMultiplier: 2.5,
  },
};

interface ThemeState {
  theme: Theme;
  colors: ThemeColors;
  toggleTheme: () => void;
}

export const useThemeStore = create<ThemeState>((set) => ({
  theme: 'dark',
  colors: THEMES.dark,
  toggleTheme: () =>
    set((state) => {
      const next = state.theme === 'dark' ? 'light' : 'dark';
      return { theme: next, colors: THEMES[next] };
    }),
}));
