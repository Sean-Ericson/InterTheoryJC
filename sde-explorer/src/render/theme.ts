/** Colours the canvases need, read from the CSS custom properties in style.css. */
export interface Theme {
  dark: boolean;
  surface: string;
  ink: string;
  ink2: string;
  muted: string;
  grid: string;
  axis: string;
  /** Fokker–Planck density (and the heatmap hue). */
  fp: string;
  /** Sample-path ensemble. */
  ens: string;
  /** Stationary density. */
  stat: string;
  /** Exact solution. */
  exact: string;
  font: string;
  /** 256-entry RGBA lookup table for the heatmap. */
  lut: Uint8ClampedArray;
}

export function readTheme(): Theme {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string) => css.getPropertyValue(name).trim();
  const dark = v('--is-dark') === '1';
  const stops = v('--heat-stops')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  return {
    dark,
    surface: v('--surface'),
    ink: v('--ink'),
    ink2: v('--ink-2'),
    muted: v('--muted'),
    grid: v('--grid'),
    axis: v('--axis'),
    fp: v('--series-fp'),
    ens: v('--series-ens'),
    stat: v('--series-stat'),
    exact: v('--series-exact'),
    font: css.getPropertyValue('font-family') || 'system-ui, sans-serif',
    lut: buildLut(stops),
  };
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  const n = parseInt(h.length === 3 ? h.replace(/./g, (c) => c + c) : h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Piecewise-linear ramp through evenly spaced stops (low → high). */
function buildLut(stops: string[]): Uint8ClampedArray {
  const rgb = (stops.length >= 2 ? stops : ['#ffffff', '#000000']).map(hexToRgb);
  const lut = new Uint8ClampedArray(256 * 4);
  const segs = rgb.length - 1;
  for (let i = 0; i < 256; i++) {
    const u = (i / 255) * segs;
    const s = Math.min(segs - 1, Math.floor(u));
    const f = u - s;
    for (let c = 0; c < 3; c++) lut[i * 4 + c] = rgb[s][c] + (rgb[s + 1][c] - rgb[s][c]) * f;
    lut[i * 4 + 3] = 255;
  }
  return lut;
}

/** Shared axis/label font. */
export function labelFont(theme: Theme, size = 11): string {
  return `${size}px ${theme.font}`;
}
