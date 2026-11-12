import { Chart, type ChartEvents, type ChartSpec, type Series } from '../plot/chart';
import { formatValue } from '../plot/scale';
import type { Theme } from '../plot/theme';
import { seriesKey } from '../plot/tooltip';

export interface CardOptions {
  title: string;
  /** Accessible name for the canvas. */
  label: string;
  events?: ChartEvents;
  className?: string;
}

/** A chart in a card: title and tools above, legend, canvas, then a footer for extra controls. */
export class Card {
  readonly el: HTMLElement;
  /** Header slot for per-chart controls (e.g. a lin/log toggle). */
  readonly tools: HTMLElement;
  readonly foot: HTMLElement;
  readonly chart: Chart;
  private readonly title: HTMLElement;
  private readonly legend: HTMLUListElement;
  private legendKey = '';

  constructor(opts: CardOptions) {
    this.el = document.createElement('section');
    this.el.className = `card ${opts.className ?? ''}`.trim();

    const head = document.createElement('div');
    head.className = 'card-head';
    this.title = document.createElement('h3');
    this.title.className = 'card-title';
    this.title.textContent = opts.title;
    this.tools = document.createElement('div');
    this.tools.className = 'card-tools';
    head.append(this.title, this.tools);

    this.legend = document.createElement('ul');
    this.legend.className = 'legend';
    this.legend.setAttribute('aria-label', 'Legend');

    const body = document.createElement('div');
    body.className = 'card-body';
    const canvas = document.createElement('canvas');
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', opts.label);
    body.append(canvas);

    this.foot = document.createElement('div');
    this.foot.className = 'card-foot';

    this.el.append(head, this.legend, body, this.foot);
    this.chart = new Chart(canvas, opts.events);
  }

  setTitle(text: string): void {
    if (this.title.textContent !== text) this.title.textContent = text;
  }

  render(spec: ChartSpec, theme: Theme): void {
    this.chart.set(spec);
    this.updateLegend(spec);
    this.chart.draw(theme);
  }

  /**
   * One entry per series label; none for a single line, which the title
   * already names. A heatmap gets its colour scale, a band a swatch.
   */
  private updateLegend(spec: ChartSpec): void {
    const seen = new Set<string>();
    const items = spec.series.filter((s) => (s.kind === 'heat' || s.legend !== false) && !seen.has(s.label) && seen.add(s.label));
    const sig = (s: Series) =>
      s.kind === 'heat' ? [s.label, s.zMax] : [s.label, s.color, s.kind, s.kind === 'line' ? s.dash : ''];
    const key = spec.note ? '' : JSON.stringify([spec.legendTitle, items.map(sig)]);
    if (key === this.legendKey) return;
    this.legendKey = key;
    this.legend.hidden = spec.note !== undefined || (items.length < 2 && !items.some((s) => s.kind === 'heat'));
    const entries = items.map((s) => {
      const li = document.createElement('li');
      if (s.kind === 'heat') {
        const ramp = document.createElement('i');
        ramp.className = 'key ramp';
        li.append(`0`, ramp, `${formatValue(s.zMax)}  ${s.label}`);
      } else if (s.kind === 'band') {
        li.append(seriesKey(s.color, 'band'), s.label);
      } else li.append(seriesKey(s.color, s.kind === 'dots' ? 'dot' : (s.dash ?? 'solid')), s.label);
      return li;
    });
    if (spec.legendTitle) {
      const li = document.createElement('li');
      li.className = 'legend-title';
      li.textContent = spec.legendTitle;
      entries.unshift(li);
    }
    this.legend.replaceChildren(...entries);
  }
}
