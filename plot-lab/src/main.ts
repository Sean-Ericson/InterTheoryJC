import './style.css';
import { FIGURES } from './figures';
import type { PageDef, PageInstance } from './pages/types';
import { readTheme } from './plot/theme';
import { SDE_EXPLORER_URL } from './figures/links';

/** Every page, in the order of the notes; the menu groups them by part. */
const PAGES: PageDef[] = FIGURES;

// Deep links: ?page=interval&x0=0.1&start=uniform&theme=dark. Each page reads
// its own parameters and keeps the address bar in step with what is on screen.
const query = new URLSearchParams(location.search);
const themeParam = query.get('theme');
if (themeParam === 'light' || themeParam === 'dark') document.documentElement.dataset.theme = themeParam;

const select = document.getElementById('page') as HTMLSelectElement;
const prev = document.getElementById('page-prev') as HTMLButtonElement;
const next = document.getElementById('page-next') as HTMLButtonElement;
const controls = document.getElementById('page-controls')!;
const main = document.getElementById('page-main')!;

(document.getElementById('sde-link') as HTMLAnchorElement).href = SDE_EXPLORER_URL;

const groups = new Map<string, PageDef[]>();
for (const p of PAGES) groups.set(p.group ?? '', [...(groups.get(p.group ?? '') ?? []), p]);
select.replaceChildren(
  ...[...groups].map(([name, pages]) => {
    const g = document.createElement('optgroup');
    g.label = name;
    g.append(...pages.map((p) => new Option(p.title, p.id)));
    return g;
  }),
);

let theme = readTheme();
let current: PageInstance | null = null;

function mount(def: PageDef, q: URLSearchParams): void {
  current?.destroy();
  controls.replaceChildren();
  main.replaceChildren();
  select.value = def.id;
  const i = PAGES.indexOf(def);
  prev.disabled = i === 0;
  next.disabled = i === PAGES.length - 1;
  prev.title = i > 0 ? PAGES[i - 1].title : '';
  next.title = i < PAGES.length - 1 ? PAGES[i + 1].title : '';
  document.title = `${def.title} · Plot Lab`;
  current = def.mount({
    controls,
    main,
    query: q,
    theme: () => theme,
    setQuery: (params) => {
      const next = new URLSearchParams({ page: def.id, ...params });
      const t = document.documentElement.dataset.theme;
      if (t) next.set('theme', t);
      history.replaceState(null, '', `${location.pathname}?${next}`);
    },
  });
}

function go(def: PageDef | undefined): void {
  if (!def) return;
  mount(def, new URLSearchParams());
  history.replaceState(null, '', `${location.pathname}?page=${def.id}`);
  window.scrollTo(0, 0);
}

select.addEventListener('change', () => go(PAGES.find((p) => p.id === select.value)));
prev.addEventListener('click', () => go(PAGES[PAGES.findIndex((p) => p.id === select.value) - 1]));
next.addEventListener('click', () => go(PAGES[PAGES.findIndex((p) => p.id === select.value) + 1]));

window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  theme = readTheme();
  current?.redraw();
});

mount(PAGES.find((p) => p.id === query.get('page')) ?? PAGES[0], query);
