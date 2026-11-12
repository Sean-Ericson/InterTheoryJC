/**
 * Colour roles a chart can ask for. Each entity keeps its role on every chart:
 * blue for the wall at 0, orange for the wall at L, aqua for the
 * quasi-stationary state and slowest mode, ink for the process as a whole.
 * The figures of the notes add two ordered families, as in make_figures.py:
 * seq1–seq5 for time snapshots (early to late, rising contrast) and div1–div5
 * for a drift swept through zero (toward 0, none, toward +x).
 */
export type Role =
  | 'ink'
  | 'ink2'
  | 'muted'
  | 'blue'
  | 'orange'
  | 'aqua'
  | 'seq1'
  | 'seq2'
  | 'seq3'
  | 'seq4'
  | 'seq5'
  | 'div1'
  | 'div2'
  | 'div3'
  | 'div4'
  | 'div5';

const ROLE_VARS: Record<Role, string> = {
  ink: '--ink',
  ink2: '--ink-2',
  muted: '--muted',
  blue: '--series-1',
  orange: '--series-2',
  aqua: '--series-3',
  seq1: '--seq-1',
  seq2: '--seq-2',
  seq3: '--seq-3',
  seq4: '--seq-4',
  seq5: '--seq-5',
  div1: '--div-1',
  div2: '--div-2',
  div3: '--div-3',
  div4: '--div-4',
  div5: '--div-5',
};

export const SEQ: Role[] = ['seq1', 'seq2', 'seq3', 'seq4', 'seq5'];
export const DIV: Role[] = ['div1', 'div2', 'div3', 'div4', 'div5'];

/** The CSS colour for a role, for legends and other HTML. */
export function roleVar(role: Role): string {
  return `var(${ROLE_VARS[role]})`;
}

/** Colours the canvases need, read from the CSS custom properties in style.css. */
export interface Theme {
  dark: boolean;
  surface: string;
  ink2: string;
  muted: string;
  grid: string;
  axis: string;
  font: string;
  color(role: Role): string;
  /** Stops of the heatmap ramp, from zero (the surface) to the top of the scale. */
  heat: string[];
}

export function readTheme(): Theme {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string) => css.getPropertyValue(name).trim();
  const roles = Object.fromEntries(Object.entries(ROLE_VARS).map(([role, name]) => [role, v(name)])) as Record<Role, string>;
  return {
    dark: v('--is-dark') === '1',
    surface: v('--surface'),
    ink2: v('--ink-2'),
    muted: v('--muted'),
    grid: v('--grid'),
    axis: v('--axis'),
    font: css.getPropertyValue('font-family') || 'system-ui, sans-serif',
    color: (role) => roles[role],
    heat: [v('--surface'), ...SEQ.map((r) => roles[r])],
  };
}

/** Shared axis/label font. */
export function labelFont(theme: Theme, size = 11): string {
  return `${size}px ${theme.font}`;
}
