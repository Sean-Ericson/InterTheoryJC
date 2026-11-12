/** Small builders for sidebar controls. Each returns its element plus setters to push state back in. */

/** Short display of a number for inputs: trims float noise like 0.30000000000000004. */
export function num(v: number): string {
  return String(Number(v.toPrecision(6)));
}

let uid = 0;
function nextId(prefix: string): string {
  return `${prefix}-${++uid}`;
}

export function group(parent: HTMLElement, title?: string): HTMLElement {
  const g = document.createElement('section');
  g.className = 'group';
  if (title) {
    const h = document.createElement('h2');
    h.textContent = title;
    g.append(h);
  }
  parent.append(g);
  return g;
}

export function hint(text: string): HTMLElement {
  const p = document.createElement('p');
  p.className = 'hint';
  p.textContent = text;
  return p;
}

export interface SliderHandle {
  el: HTMLElement;
  set(v: number): void;
  setRange(min: number, max: number, step: number): void;
}

/** Name, range slider and number box in one row. */
export function slider(opts: { label: string; min: number; max: number; step: number; value: number; onInput(v: number): void }): SliderHandle {
  const row = document.createElement('div');
  row.className = 'param';
  const id = nextId('slider');
  const label = document.createElement('label');
  label.htmlFor = id;
  label.className = 'param-name';
  label.textContent = opts.label;
  const range = document.createElement('input');
  range.type = 'range';
  range.id = id;
  const box = document.createElement('input');
  box.type = 'number';
  box.setAttribute('aria-label', `${opts.label} value`);
  let min = opts.min;
  let max = opts.max;

  const setRange = (lo: number, hi: number, step: number) => {
    min = lo;
    max = hi;
    range.min = String(lo);
    range.max = String(hi);
    range.step = String(step);
    box.step = String(step);
  };
  const set = (v: number) => {
    range.value = String(v);
    box.value = num(v);
  };
  setRange(opts.min, opts.max, opts.step);
  set(opts.value);

  range.addEventListener('input', () => {
    const v = Number(range.value);
    box.value = num(v);
    opts.onInput(v);
  });
  box.addEventListener('change', () => {
    const v = Number(box.value);
    if (!Number.isFinite(v)) return set(Number(range.value));
    const c = Math.min(max, Math.max(min, v));
    set(c);
    opts.onInput(c);
  });
  row.append(label, range, box);
  return { el: row, set, setRange };
}

export interface SegmentedHandle<T extends string> {
  el: HTMLElement;
  set(v: T): void;
  setAvailable(v: T, available: boolean, reason: string): void;
}

/** A row of mutually exclusive buttons (radio inputs). */
export function segmented<T extends string>(opts: { label: string; options: [T, string][]; value: T; onChange(v: T): void; small?: boolean }): SegmentedHandle<T> {
  const box = document.createElement('div');
  box.className = opts.small ? 'segmented small' : 'segmented';
  box.setAttribute('role', 'radiogroup');
  box.setAttribute('aria-label', opts.label);
  box.style.setProperty('--n', String(opts.options.length));
  const name = nextId('seg');
  const inputs = new Map<T, HTMLInputElement>();
  for (const [value, text] of opts.options) {
    const label = document.createElement('label');
    const input = document.createElement('input');
    input.type = 'radio';
    input.name = name;
    input.value = value;
    input.checked = value === opts.value;
    input.addEventListener('change', () => {
      if (input.checked) opts.onChange(value);
    });
    const span = document.createElement('span');
    span.textContent = text;
    label.append(input, span);
    box.append(label);
    inputs.set(value, input);
  }
  return {
    el: box,
    set: (v) => {
      for (const [value, input] of inputs) input.checked = value === v;
    },
    setAvailable: (v, available, reason) => {
      const input = inputs.get(v)!;
      input.disabled = !available;
      const label = input.closest('label')!;
      label.classList.toggle('unavailable', !available);
      label.title = available ? '' : reason;
    },
  };
}

export interface CheckboxHandle {
  el: HTMLElement;
  set(v: boolean): void;
  setAvailable(available: boolean, reason: string): void;
}

export function checkbox(opts: { label: string; checked: boolean; onChange(v: boolean): void }): CheckboxHandle {
  const label = document.createElement('label');
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = opts.checked;
  input.addEventListener('change', () => opts.onChange(input.checked));
  label.append(input, ` ${opts.label}`);
  return {
    el: label,
    set: (v) => {
      input.checked = v;
    },
    setAvailable: (available, reason) => {
      input.disabled = !available;
      label.classList.toggle('unavailable', !available);
      label.title = available ? '' : reason;
    },
  };
}

export interface NumberHandle {
  el: HTMLElement;
  input: HTMLInputElement;
  set(v: number): void;
}

/** Labelled number input; `onChange` returns false to reject the value. */
export function numberField(opts: { label: string; value: number; onChange(v: number): boolean }): NumberHandle {
  const label = document.createElement('label');
  label.className = 'field';
  const span = document.createElement('span');
  span.className = 'field-label';
  span.textContent = opts.label;
  const input = document.createElement('input');
  input.type = 'number';
  input.step = 'any';
  let last = opts.value;
  input.value = num(last);
  input.addEventListener('change', () => {
    const v = Number(input.value);
    if (input.value !== '' && Number.isFinite(v) && opts.onChange(v)) last = v;
    else input.value = num(last);
  });
  label.append(span, input);
  return {
    el: label,
    input,
    set: (v) => {
      last = v;
      input.value = num(v);
    },
  };
}

export function selectField(opts: { label: string; options: [string, string][]; value: string; onChange(v: string): void }): { el: HTMLElement; select: HTMLSelectElement } {
  const label = document.createElement('label');
  label.className = 'field';
  const span = document.createElement('span');
  span.className = 'field-label';
  span.textContent = opts.label;
  const select = document.createElement('select');
  select.replaceChildren(...opts.options.map(([v, t]) => new Option(t, v)));
  select.value = opts.value;
  select.addEventListener('change', () => opts.onChange(select.value));
  label.append(span, select);
  return { el: label, select };
}

export function button(text: string, onClick: () => void, className = 'btn'): HTMLButtonElement {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = className;
  b.textContent = text;
  b.addEventListener('click', onClick);
  return b;
}
