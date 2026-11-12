import type { ChartSpec } from '../plot/chart';
import { Scale } from '../plot/scale';
import { Card } from '../ui/card';
import { button, checkbox, group, hint, num, segmented, selectField } from '../ui/controls';
import type { PageDef, PageHost, PageInstance } from '../pages/types';
import { sdeExplorerUrl, type SdeLink } from './links';

/**
 * Interactive figures of the notes (fpe-examples). A figure is declared, not
 * hand-built: its parameters with their defaults, the sidebar controls for
 * them, a model computed from the parameters, and one panel per subfigure
 * that turns the model into a chart spec. The framework builds the page:
 * sidebar, cards with the notes' captions, deep links in the URL, play
 * buttons for time parameters, linked hover, and a link that opens the same
 * process in SDE Explorer.
 */

export type Values = Record<string, number | string | boolean>;

export interface NumberControl<V extends Values> {
  kind?: 'number';
  key: keyof V & string;
  label: string;
  min: number;
  max: number;
  step?: number;
  /** Slider moves in log steps. */
  log?: boolean;
  /** A ▶ button sweeps the value from min to max. */
  play?: boolean;
}

export interface ChoiceControl<V extends Values> {
  kind: 'choice';
  key: keyof V & string;
  label: string;
  options: [string, string][];
  /** A dropdown instead of a row of buttons. */
  select?: boolean;
}

export interface ToggleControl<V extends Values> {
  kind: 'toggle';
  key: keyof V & string;
  label: string;
}

export type Control<V extends Values> = NumberControl<V> | ChoiceControl<V> | ToggleControl<V>;

export interface ControlGroup<V extends Values> {
  title: string;
  controls: Control<V>[];
  hint?: string;
}

export interface Panel<V extends Values, M> {
  /** "(a) …", as in the notes. */
  title: string | ((m: M, v: V) => string);
  /** The notes' caption for this subfigure, plus what is interactive about it (HTML). */
  caption: string;
  wide?: boolean;
  /** Small lin/log style switches in the card header; their keys are parameters too. */
  tools?: ChoiceControl<V>[];
  /** Panels with the same link share a hover cursor (e.g. a common time axis). */
  link?: string;
  /** Pressing or dragging on the plot sets parameters from the x position. */
  pick?(x: number, v: V): Partial<V>;
  /** A hint under the chart about the interaction, e.g. "Drag to move t". */
  interaction?: string;
  spec(m: M, v: V, hover: number | null): ChartSpec;
}

export interface FigureDef<V extends Values, M> {
  id: string;
  /** Menu group: the part of the notes. */
  group: string;
  /** Section title, as in the notes. */
  title: string;
  /** The figure file(s) in fpe-examples/figures. */
  files: string[];
  /** Gardiner sections, e.g. "§5.2.6". */
  gardiner?: string;
  /** One or two sentences: the process and what the figure shows (HTML). */
  intro: string;
  /** The model in the notes' notation (HTML, one line each). */
  equations: string[];
  defaults: V;
  controls: ControlGroup<V>[];
  model(v: V): M;
  panels: Panel<V, M>[];
  /** Key results from the notes, shown under the charts (HTML list items). */
  results?: string[];
  /** The same process in SDE Explorer, where paths and the density run in time. */
  simulate?(v: V): SdeLink | null;
}

const PLAY_SECONDS = 8;

/** A page for the shell's menu from a figure declaration. */
export function figurePage<V extends Values, M>(def: FigureDef<V, M>): PageDef {
  return { id: def.id, title: def.title, group: def.group, mount: (host) => mountFigure(def, host) };
}

function readValues<V extends Values, M>(def: FigureDef<V, M>, q: URLSearchParams): V {
  const v = { ...def.defaults };
  const choices = new Map<string, string[]>();
  for (const g of def.controls) for (const c of g.controls) if (c.kind === 'choice') choices.set(c.key, c.options.map((o) => o[0]));
  for (const p of def.panels) for (const c of p.tools ?? []) choices.set(c.key, c.options.map((o) => o[0]));
  for (const key of Object.keys(v) as (keyof V & string)[]) {
    if (!q.has(key)) continue;
    const raw = q.get(key)!;
    const d = def.defaults[key];
    if (typeof d === 'number') {
      const n = Number(raw);
      if (raw !== '' && Number.isFinite(n)) (v as Values)[key] = n;
    } else if (typeof d === 'boolean') (v as Values)[key] = raw === '1';
    else if (!choices.has(key) || choices.get(key)!.includes(raw)) (v as Values)[key] = raw;
  }
  // Clamp numbers to their sliders.
  for (const g of def.controls)
    for (const c of g.controls)
      if (c.kind === undefined || c.kind === 'number') {
        const x = v[c.key] as number;
        (v as Values)[c.key] = Math.min(c.max, Math.max(c.min, x));
      }
  return v;
}

function writeValues<V extends Values, M>(def: FigureDef<V, M>, v: V): Record<string, string> {
  const out: Record<string, string> = {};
  for (const key of Object.keys(v)) {
    const x = v[key];
    if (x === def.defaults[key]) continue;
    out[key] = typeof x === 'number' ? num(x) : typeof x === 'boolean' ? (x ? '1' : '0') : x;
  }
  return out;
}

function mountFigure<V extends Values, M>(def: FigureDef<V, M>, host: PageHost): PageInstance {
  const v = readValues(def, host.query);
  let model: M = def.model(v);
  let dirtyModel = false;
  const hover = new Map<string, number | null>();
  let frame = 0;
  let persistTimer = 0;
  let playing: { key: string; scale: Scale; start: number; u0: number; button: HTMLButtonElement } | null = null;
  const setters: (() => void)[] = [];

  // ---------------------------------------------------------------- sidebar

  const intro = group(host.controls);
  const blurb = document.createElement('p');
  blurb.className = 'blurb';
  blurb.innerHTML = def.intro;
  const source = document.createElement('p');
  source.className = 'source';
  source.innerHTML = `Notes: <cite>${def.title}</cite> · ${def.files.map((f) => `<code>${f}</code>`).join(', ')}${
    def.gardiner ? ` · <span class="ref">Gardiner ${def.gardiner}</span>` : ''
  }`;
  intro.append(blurb, source);

  for (const g of def.controls) {
    const box = group(host.controls, g.title);
    const toggles: HTMLElement[] = [];
    for (const c of g.controls) {
      if (c.kind === 'toggle') toggles.push(toggleRow(c));
      else box.append(c.kind === 'choice' ? choiceRow(c) : numberRow(c));
    }
    if (toggles.length) {
      const checks = document.createElement('div');
      checks.className = 'checks';
      checks.append(...toggles);
      box.append(checks);
    }
    if (g.hint) box.append(hint(g.hint));
  }

  let simLink: HTMLAnchorElement | null = null;
  if (def.simulate) {
    const box = group(host.controls, 'Simulate');
    simLink = document.createElement('a');
    simLink.className = 'btn link-btn';
    simLink.target = '_blank';
    simLink.rel = 'noopener';
    simLink.textContent = 'Open in SDE Explorer ↗';
    box.append(simLink, hint('Sample paths and the Fokker–Planck solution of this process, run forward in time, with the current parameters.'));
  }

  const reset = button('Reset to the notes’ values', () => {
    stopPlay();
    Object.assign(v, def.defaults);
    for (const s of setters) s();
    changed(true);
  });
  group(host.controls).append(reset);

  function numberRow(c: NumberControl<V>): HTMLElement {
    const row = document.createElement('div');
    row.className = ['param', c.play ? 'play' : '', c.label.length > 3 ? 'long' : ''].join(' ').trim();
    const id = `fig-${def.id}-${c.key}`;
    const label = document.createElement('label');
    label.className = 'param-name';
    label.htmlFor = id;
    label.textContent = c.label;
    const scale = new Scale(c.log ? 'log' : 'linear', c.min, c.max, 0, 1);
    const range = document.createElement('input');
    range.type = 'range';
    range.id = id;
    range.min = '0';
    range.max = '1000';
    range.step = c.log ? '1' : String((1000 * (c.step ?? (c.max - c.min) / 1000)) / (c.max - c.min));
    const box = document.createElement('input');
    box.type = 'number';
    box.step = c.log ? 'any' : String(c.step ?? 'any');
    box.setAttribute('aria-label', `${c.label} value`);
    const sync = () => {
      const x = v[c.key] as number;
      range.value = String(Math.round(scale.fraction(x) * 1000));
      if (document.activeElement !== box) box.value = num(x);
    };
    setters.push(sync);
    sync();
    range.addEventListener('input', () => {
      if (playing?.key === c.key) stopPlay();
      let x = scale.atFraction(Number(range.value) / 1000);
      if (!c.log && c.step) x = Math.round(x / c.step) * c.step;
      set(c.key, Number(x.toPrecision(6)));
      box.value = num(v[c.key] as number);
    });
    box.addEventListener('change', () => {
      const x = Number(box.value);
      if (box.value === '' || !Number.isFinite(x)) return sync();
      set(c.key, Math.min(c.max, Math.max(c.min, x)));
      sync();
    });
    row.append(label, range, box);
    if (c.play) {
      const play = button('▶', () => {
        if (playing?.key === c.key) return stopPlay();
        stopPlay();
        let u0 = scale.fraction(v[c.key] as number);
        if (u0 >= 0.999) u0 = 0;
        playing = { key: c.key, scale, start: performance.now(), u0, button: play };
        play.textContent = '❚❚';
        play.setAttribute('aria-label', `Pause ${c.label}`);
        schedule();
      }, 'btn play-btn');
      play.setAttribute('aria-label', `Play ${c.label}`);
      play.title = `Sweep ${c.label} from ${num(c.min)} to ${num(c.max)}`;
      row.append(play);
    }
    return row;
  }

  function choiceRow(c: ChoiceControl<V>): HTMLElement {
    const wrap = document.createElement('div');
    wrap.className = 'field';
    if (c.select) {
      const f = selectField({ label: c.label, options: c.options, value: v[c.key] as string, onChange: (x) => set(c.key, x) });
      setters.push(() => (f.select.value = v[c.key] as string));
      return f.el;
    }
    const lab = document.createElement('span');
    lab.className = 'field-label';
    lab.textContent = c.label;
    const seg = segmented({ label: c.label, options: c.options, value: v[c.key] as string, onChange: (x) => set(c.key, x) });
    setters.push(() => seg.set(v[c.key] as string));
    wrap.append(lab, seg.el);
    return wrap;
  }

  function toggleRow(c: ToggleControl<V>): HTMLElement {
    const box = checkbox({ label: c.label, checked: v[c.key] as boolean, onChange: (x) => set(c.key, x) });
    setters.push(() => box.set(v[c.key] as boolean));
    return box.el;
  }

  // ---------------------------------------------------------------- main

  const head = document.createElement('header');
  head.className = 'fig-head';
  const h2 = document.createElement('h2');
  h2.className = 'fig-title';
  h2.textContent = def.title;
  const eqs = document.createElement('div');
  eqs.className = 'equations';
  eqs.innerHTML = def.equations.map((e) => `<p class="eq">${e}</p>`).join('');
  head.append(h2, eqs);

  const cards = document.createElement('div');
  cards.className = `cards n${def.panels.length}`;
  const built = def.panels.map((p) => {
    const card = new Card({
      title: typeof p.title === 'string' ? p.title : '',
      label: (typeof p.title === 'string' ? p.title : def.title).replace(/^\([a-z]\)\s*/, ''),
      className: p.wide ? 'wide' : '',
      events: {
        hover: (x) => {
          if (!p.link) return;
          hover.set(p.link, x);
          schedule();
        },
        pick: p.pick
          ? (x) => {
              const upd = p.pick!(x, v);
              stopPlay();
              Object.assign(v, upd);
              for (const s of setters) s();
              changed(true);
            }
          : undefined,
      },
    });
    for (const t of p.tools ?? []) {
      const seg = segmented({ label: t.label, options: t.options, value: v[t.key] as string, onChange: (x) => set(t.key, x), small: true });
      setters.push(() => seg.set(v[t.key] as string));
      card.tools.append(seg.el);
    }
    const cap = document.createElement('p');
    cap.className = 'caption';
    cap.innerHTML = p.caption + (p.interaction ? ` <span class="interaction">${p.interaction}</span>` : '');
    card.foot.append(cap);
    cards.append(card.el);
    return { p, card };
  });

  host.main.append(head, cards);
  if (def.results?.length) {
    const box = document.createElement('details');
    box.className = 'formulas';
    box.open = true;
    box.innerHTML = `<summary>Key results</summary><ul>${def.results.map((r) => `<li>${r}</li>`).join('')}</ul>`;
    host.main.append(box);
  }

  // ---------------------------------------------------------------- updates

  function set(key: string, x: number | string | boolean): void {
    (v as Values)[key] = x;
    changed(true);
  }

  function changed(recompute: boolean): void {
    if (recompute) dirtyModel = true;
    clearTimeout(persistTimer);
    persistTimer = window.setTimeout(() => host.setQuery(writeValues(def, v)), 250);
    schedule();
  }

  function stopPlay(): void {
    if (!playing) return;
    playing.button.textContent = '▶';
    playing.button.setAttribute('aria-label', 'Play');
    playing = null;
  }

  function schedule(): void {
    if (!frame) frame = requestAnimationFrame(tick);
  }

  function tick(now: number): void {
    frame = 0;
    if (playing) {
      const u = playing.u0 + (now - playing.start) / (PLAY_SECONDS * 1000);
      (v as Values)[playing.key] = Number(playing.scale.atFraction(Math.min(1, u)).toPrecision(6));
      for (const s of setters) s();
      dirtyModel = true;
      if (u >= 1) {
        stopPlay();
        changed(false);
      } else schedule();
    }
    if (dirtyModel) {
      model = def.model(v);
      dirtyModel = false;
    }
    draw();
  }

  function draw(): void {
    const theme = host.theme();
    for (const { p, card } of built) {
      if (typeof p.title !== 'string') card.setTitle(p.title(model, v));
      let spec: ChartSpec;
      try {
        spec = p.spec(model, v, p.link ? (hover.get(p.link) ?? null) : null);
      } catch (err) {
        console.error(err);
        spec = { x: { domain: [0, 1], scale: 'linear', label: '' }, y: { scale: 'linear' }, series: [], heading: () => '', note: 'Could not compute this panel.' };
      }
      card.render(spec, theme);
    }
    if (simLink && def.simulate) {
      const link = def.simulate(v);
      simLink.hidden = !link;
      if (link) simLink.href = sdeExplorerUrl(link);
    }
  }

  const observer = new ResizeObserver(() => schedule());
  observer.observe(cards);
  schedule();

  return {
    redraw: schedule,
    destroy: () => {
      cancelAnimationFrame(frame);
      clearTimeout(persistTimer);
      observer.disconnect();
    },
  };
}
