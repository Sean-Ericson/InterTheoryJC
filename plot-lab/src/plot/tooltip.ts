import { roleVar, type Role } from './theme';

export interface TooltipRow {
  label: string;
  value: string;
  color: Role;
  /** Key style, mirroring the mark: a line (solid, dashed, dotted) or a dot. */
  key?: 'solid' | 'dashed' | 'dotted' | 'dot' | 'band';
}

let el: HTMLElement | null = null;

function element(): HTMLElement {
  if (!el) {
    el = document.createElement('div');
    el.className = 'tooltip';
    el.setAttribute('role', 'status');
    el.hidden = true;
    document.body.append(el);
  }
  return el;
}

/** Make a legend or tooltip key: a short stroke in the series colour, or a dot. */
export function seriesKey(color: Role, key: TooltipRow['key'] = 'solid'): HTMLElement {
  const i = document.createElement('i');
  i.className = key === 'dot' ? 'key dot' : `key ${key}`;
  i.style.setProperty('--c', roleVar(color));
  return i;
}

/** One tooltip, every series: values lead, labels follow. */
export function showTooltip(clientX: number, clientY: number, heading: string, rows: TooltipRow[]): void {
  const tip = element();
  const head = document.createElement('div');
  head.className = 'tt-head';
  head.textContent = heading;
  tip.replaceChildren(
    head,
    ...rows.map((r) => {
      const row = document.createElement('div');
      row.className = 'tt-row';
      const val = document.createElement('span');
      val.className = 'tt-val';
      val.textContent = r.value;
      const label = document.createElement('span');
      label.className = 'tt-label';
      label.append(seriesKey(r.color, r.key), r.label);
      row.append(val, label);
      return row;
    }),
  );
  tip.hidden = false;
  const pad = 14;
  const w = tip.offsetWidth;
  const h = tip.offsetHeight;
  let x = clientX + pad;
  let y = clientY + pad;
  if (x + w > window.innerWidth - 8) x = clientX - pad - w;
  if (y + h > window.innerHeight - 8) y = clientY - pad - h;
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${Math.max(8, y)}px`;
}

export function hideTooltip(): void {
  if (el) el.hidden = true;
}
