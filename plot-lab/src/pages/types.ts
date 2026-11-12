import type { Theme } from '../plot/theme';

/** What the shell gives a page when it mounts. */
export interface PageHost {
  /** Sidebar container for the page's controls. */
  controls: HTMLElement;
  /** Main area for the page's charts. */
  main: HTMLElement;
  /** URL parameters the page was opened with. */
  query: URLSearchParams;
  /** Replace the page's URL parameters, so the address bar always links to the current view. */
  setQuery(params: Record<string, string>): void;
  /** Current colours (they change with the system theme). */
  theme(): Theme;
}

export interface PageInstance {
  redraw(): void;
  destroy(): void;
}

/** One interactive topic. Add new ones to PAGES in main.ts. */
export interface PageDef {
  id: string;
  title: string;
  /** Menu group: the part of the notes the page belongs to. */
  group?: string;
  mount(host: PageHost): PageInstance;
}
