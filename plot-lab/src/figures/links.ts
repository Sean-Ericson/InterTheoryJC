import { num } from '../ui/controls';

/** Boundary conditions as SDE Explorer names them. */
export type BC = 'open' | 'reflecting' | 'absorbing' | 'periodic';

/** A process for SDE Explorer: dx = a dt + b dW with Gardiner's B = b². */
export interface SdeLink {
  /** Drift a(x, t) and noise b(x, t); free symbols are parameters. */
  a: string;
  b: string;
  params?: Record<string, number>;
  domain: [number, number];
  bc: [BC, BC];
  x0: number;
  T: number;
  /** Show the density conditioned on survival, q = p/S. */
  cond?: boolean;
}

/**
 * SDE Explorer's address. Built, the two apps are deployed side by side
 * (…/plot-lab/ and …/sde-explorer/); in development each runs its own Vite
 * server (see vite.config.ts in both). VITE_SDE_EXPLORER_URL overrides both.
 */
export const SDE_EXPLORER_URL: string = import.meta.env.VITE_SDE_EXPLORER_URL ?? (import.meta.env.DEV ? 'http://localhost:5173/' : '../sde-explorer/');

export function sdeExplorerUrl(link: SdeLink): string {
  const q = new URLSearchParams({
    a: link.a,
    b: link.b,
    lo: num(link.domain[0]),
    hi: num(link.domain[1]),
    bc: link.bc.join(','),
    x0: num(link.x0),
    T: num(link.T),
  });
  if (link.params && Object.keys(link.params).length) q.set('p', Object.entries(link.params).map(([k, v]) => `${k}:${num(v)}`).join(','));
  if (link.cond) q.set('cond', '1');
  const theme = document.documentElement.dataset.theme;
  if (theme) q.set('theme', theme);
  return `${SDE_EXPLORER_URL}?${q}`;
}
