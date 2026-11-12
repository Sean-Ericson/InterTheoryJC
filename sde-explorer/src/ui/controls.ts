import { formulaParams, FormulaError, type BC, type Interp } from '../model';
import { CUSTOM_ID, PRESETS, findPreset, type ParamDef } from '../presets';
import type { HeatScale } from '../render/pathsPanel';
import {
  DURATION_OPTIONS,
  GRID_OPTIONS,
  MAX_SHOWN,
  PATH_OPTIONS,
  STEP_OPTIONS,
  applyPreset,
  type Settings,
  type Start,
  type ViewOptions,
} from '../settings';

/** What changed: 'sim' needs a fresh run, 'view' only a redraw, 'playback' the play speed. */
export type ChangeKind = 'sim' | 'view' | 'playback';

export interface ControlHandlers {
  onChange(kind: ChangeKind): void;
}

const BC_LABELS: Record<BC, string> = {
  open: 'Open (natural)',
  reflecting: 'Reflecting',
  absorbing: 'Absorbing',
  periodic: 'Periodic',
};

const VIEW_KEYS: (keyof Omit<ViewOptions, 'heatScale'>)[] = ['heatmap', 'paths', 'histogram', 'fp', 'exact', 'stationary', 'conditioned', 'correlation'];

function $<T extends HTMLElement = HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing #${id}`);
  return el as T;
}

/** Enable or disable a checkbox or radio, greying its label and saying why. */
function enable(input: HTMLInputElement, available: boolean, reason: string): void {
  input.disabled = !available;
  const label = input.closest('label')!;
  label.classList.toggle('unavailable', !available);
  label.title = available ? '' : reason;
}

function options(select: HTMLSelectElement, items: [string, string][]): void {
  select.replaceChildren(...items.map(([value, label]) => new Option(label, value)));
}

/** Short display of a number for inputs: trims float noise like 0.30000000000000004. */
function num(v: number): string {
  return String(Number(v.toPrecision(6)));
}

/** Wires the sidebar inputs to a Settings object. */
export class Controls {
  private formulaTimer = 0;

  constructor(
    private readonly s: Settings,
    private readonly h: ControlHandlers,
  ) {
    options(
      $<HTMLSelectElement>('preset'),
      [...PRESETS.map((p): [string, string] => [p.id, p.name]), [CUSTOM_ID, 'Custom formulas']],
    );
    for (const id of ['bc-left', 'bc-right']) options($<HTMLSelectElement>(id), Object.entries(BC_LABELS) as [string, string][]);
    options($<HTMLSelectElement>('steps'), STEP_OPTIONS.map((n) => [String(n), n.toLocaleString()]));
    options($<HTMLSelectElement>('paths'), PATH_OPTIONS.map((n) => [String(n), n.toLocaleString()]));
    options($<HTMLSelectElement>('grid'), GRID_OPTIONS.map((n) => [String(n), `${n} cells`]));
    options($<HTMLSelectElement>('duration'), DURATION_OPTIONS.map((n) => [String(n), `${n} s`]));
    $<HTMLInputElement>('show').max = String(MAX_SHOWN);
    this.bind();
    this.sync();
  }

  /** Push every setting into the DOM (after loading a preset, say). */
  sync(): void {
    const s = this.s;
    $<HTMLSelectElement>('preset').value = s.presetId;
    const preset = findPreset(s.presetId);
    $('blurb').textContent = preset ? preset.blurb : 'Your own drift and noise. Any symbol other than x and t becomes a parameter.';
    $('ref').textContent = preset ? `Gardiner ${preset.ref}` : '';
    $<HTMLInputElement>('drift').value = s.drift;
    $<HTMLInputElement>('noise').value = s.noise;
    this.setFormulaError(null);
    for (const r of document.querySelectorAll<HTMLInputElement>('input[name="interp"]')) r.checked = r.value === s.interp;
    this.renderParams();
    this.syncStart();
    this.syncInitial();
    $<HTMLInputElement>('lo').value = num(s.domain[0]);
    $<HTMLInputElement>('hi').value = num(s.domain[1]);
    $<HTMLSelectElement>('bc-left').value = s.bc[0];
    $<HTMLSelectElement>('bc-right').value = s.bc[1];
    $<HTMLInputElement>('T').value = num(s.T);
    $<HTMLSelectElement>('steps').value = String(s.steps);
    $<HTMLSelectElement>('paths').value = String(s.nPaths);
    $<HTMLInputElement>('show').value = String(s.nShow);
    $('show-out').textContent = String(s.nShow);
    $<HTMLSelectElement>('grid').value = String(s.nGrid);
    $<HTMLInputElement>('seed').value = String(s.seed);
    $<HTMLSelectElement>('duration').value = String(s.duration);
    for (const key of VIEW_KEYS) $<HTMLInputElement>(`view-${key}`).checked = s.view[key];
    $<HTMLSelectElement>('heat-scale').value = s.view.heatScale;
  }

  /** Show a formula error under the offending input (null clears both). */
  setFormulaError(err: FormulaError | null): void {
    for (const which of ['a', 'b'] as const) {
      const input = $<HTMLInputElement>(which === 'a' ? 'drift' : 'noise');
      const msg = $(which === 'a' ? 'drift-error' : 'noise-error');
      const mine = err?.which === which;
      msg.textContent = mine ? err!.message : '';
      input.setAttribute('aria-invalid', String(mine));
    }
  }

  /** Enable/disable a view checkbox with an explanation for why it's off. */
  setAvailable(key: 'exact' | 'stationary' | 'conditioned', available: boolean, reason: string): void {
    enable($<HTMLInputElement>(`view-${key}`), available, reason);
  }

  /** With absorbing walls the stationary state is the quasi-stationary density. */
  setStationaryKind(kind: 'stationary' | 'qsd'): void {
    $('view-stationary-text').textContent = kind === 'qsd' ? 'Quasi-stationary density' : 'Stationary density';
    $('start-stationary-text').textContent = kind === 'qsd' ? 'Quasi-stationary φ₀' : 'Stationary pₛ';
  }

  /** Starting in the stationary state needs one; without it the start falls back to the Gaussian. */
  setStartAvailable(available: boolean, reason: string): void {
    enable(document.querySelector<HTMLInputElement>('input[name="start"][value="stationary"]')!, available, reason);
    if (!available && this.s.start === 'stationary') {
      this.s.start = 'gaussian';
      this.syncStart();
    }
  }

  private bind(): void {
    const s = this.s;
    const sim = () => this.h.onChange('sim');

    $<HTMLSelectElement>('preset').addEventListener('change', (e) => {
      const id = (e.target as HTMLSelectElement).value;
      const preset = findPreset(id);
      if (preset) applyPreset(s, preset);
      else s.presetId = CUSTOM_ID;
      this.sync();
      sim();
    });

    for (const id of ['drift', 'noise']) {
      $<HTMLInputElement>(id).addEventListener('input', () => {
        clearTimeout(this.formulaTimer);
        this.formulaTimer = window.setTimeout(() => this.formulasEdited(), 350);
      });
    }

    for (const r of document.querySelectorAll<HTMLInputElement>('input[name="interp"]')) {
      r.addEventListener('change', () => {
        if (r.checked) {
          s.interp = r.value as Interp;
          sim();
        }
      });
    }

    for (const r of document.querySelectorAll<HTMLInputElement>('input[name="start"]')) {
      r.addEventListener('change', () => {
        if (r.checked) {
          s.start = r.value as Start;
          this.syncStart();
          sim();
        }
      });
    }

    const x0Range = $<HTMLInputElement>('x0-range');
    const x0Num = $<HTMLInputElement>('x0');
    x0Range.addEventListener('input', () => {
      s.x0 = Number(x0Range.value);
      x0Num.value = num(s.x0);
      sim();
    });
    x0Num.addEventListener('change', () => {
      const v = Number(x0Num.value);
      if (Number.isFinite(v)) s.x0 = Math.min(s.domain[1], Math.max(s.domain[0], v));
      this.syncInitial();
      sim();
    });
    $<HTMLInputElement>('sigma0').addEventListener('change', (e) => {
      const v = Number((e.target as HTMLInputElement).value);
      if (Number.isFinite(v) && v >= 0) s.sigma0 = v;
      this.syncInitial();
      sim();
    });

    for (const [id, idx] of [
      ['lo', 0],
      ['hi', 1],
    ] as const) {
      $<HTMLInputElement>(id).addEventListener('change', (e) => {
        const input = e.target as HTMLInputElement;
        const v = Number(input.value);
        const next: [number, number] = [...s.domain];
        next[idx] = v;
        if (Number.isFinite(v) && next[1] > next[0]) {
          s.domain = next;
          s.x0 = Math.min(next[1], Math.max(next[0], s.x0));
          this.syncInitial();
          sim();
        } else {
          input.value = num(s.domain[idx]);
        }
      });
    }

    for (const [id, idx] of [
      ['bc-left', 0],
      ['bc-right', 1],
    ] as const) {
      $<HTMLSelectElement>(id).addEventListener('change', (e) => {
        const v = (e.target as HTMLSelectElement).value as BC;
        const other = 1 - idx;
        s.bc[idx] = v;
        // Periodic is a property of both ends at once.
        if (v === 'periodic') s.bc[other] = 'periodic';
        else if (s.bc[other] === 'periodic') s.bc[other] = v;
        $<HTMLSelectElement>('bc-left').value = s.bc[0];
        $<HTMLSelectElement>('bc-right').value = s.bc[1];
        sim();
      });
    }

    $<HTMLInputElement>('T').addEventListener('change', (e) => {
      const input = e.target as HTMLInputElement;
      const v = Number(input.value);
      if (Number.isFinite(v) && v > 0) {
        s.T = v;
        sim();
      } else input.value = num(s.T);
    });
    const selectNum = (id: string, set: (v: number) => void) =>
      $<HTMLSelectElement>(id).addEventListener('change', (e) => {
        set(Number((e.target as HTMLSelectElement).value));
        sim();
      });
    selectNum('steps', (v) => (s.steps = v));
    selectNum('paths', (v) => (s.nPaths = v));
    selectNum('grid', (v) => (s.nGrid = v));

    const show = $<HTMLInputElement>('show');
    show.addEventListener('input', () => {
      s.nShow = Number(show.value);
      $('show-out').textContent = show.value;
      this.h.onChange('view');
    });

    $<HTMLInputElement>('seed').addEventListener('change', (e) => {
      const v = Math.floor(Number((e.target as HTMLInputElement).value));
      if (Number.isFinite(v)) s.seed = v;
      sim();
    });
    $('resample').addEventListener('click', () => {
      s.seed = Math.floor(Math.random() * 1e6);
      $<HTMLInputElement>('seed').value = String(s.seed);
      sim();
    });

    $<HTMLSelectElement>('duration').addEventListener('change', (e) => {
      s.duration = Number((e.target as HTMLSelectElement).value);
      this.h.onChange('playback');
    });

    for (const key of VIEW_KEYS) {
      $<HTMLInputElement>(`view-${key}`).addEventListener('change', (e) => {
        s.view[key] = (e.target as HTMLInputElement).checked;
        this.h.onChange('view');
      });
    }
    $<HTMLSelectElement>('heat-scale').addEventListener('change', (e) => {
      s.view.heatScale = (e.target as HTMLSelectElement).value as HeatScale;
      this.h.onChange('view');
    });
  }

  /** Validate edited formulas; on success switch to Custom and rebuild the parameter list. */
  private formulasEdited(): void {
    const s = this.s;
    const drift = $<HTMLInputElement>('drift').value;
    const noise = $<HTMLInputElement>('noise').value;
    let names: string[];
    try {
      names = formulaParams(drift, noise).params;
    } catch (err) {
      if (err instanceof FormulaError) this.setFormulaError(err);
      return;
    }
    this.setFormulaError(null);
    if (drift === s.drift && noise === s.noise) return;
    s.drift = drift;
    s.noise = noise;
    s.presetId = CUSTOM_ID;
    const old = new Map(s.params.map((p) => [p.name, p]));
    s.params = names.map((name): ParamDef => old.get(name) ?? { name, value: 1, min: -5, max: 5, step: 0.01 });
    $<HTMLSelectElement>('preset').value = CUSTOM_ID;
    $('blurb').textContent = 'Your own drift and noise. Any symbol other than x and t becomes a parameter.';
    $('ref').textContent = '';
    this.renderParams();
    this.h.onChange('sim');
  }

  private renderParams(): void {
    const box = $('params');
    if (this.s.params.length === 0) {
      const p = document.createElement('p');
      p.className = 'hint';
      p.textContent = 'No parameters. Use any symbol besides x and t in a formula to add one.';
      box.replaceChildren(p);
      return;
    }
    box.replaceChildren(...this.s.params.map((p) => this.paramRow(p)));
  }

  private paramRow(p: ParamDef): HTMLElement {
    const row = document.createElement('div');
    row.className = 'param';
    const id = `param-${p.name}`;
    const label = document.createElement('label');
    label.htmlFor = id;
    label.className = 'param-name';
    label.textContent = p.name;
    const range = document.createElement('input');
    range.type = 'range';
    range.id = id;
    range.min = String(p.min);
    range.max = String(p.max);
    range.step = String(p.step);
    range.value = String(p.value);
    range.setAttribute('aria-label', `${p.name} slider`);
    const box = document.createElement('input');
    box.type = 'number';
    box.step = String(p.step);
    box.value = num(p.value);
    box.setAttribute('aria-label', `${p.name} value`);

    range.addEventListener('input', () => {
      p.value = Number(range.value);
      box.value = num(p.value);
      this.h.onChange('sim');
    });
    box.addEventListener('change', () => {
      const v = Number(box.value);
      if (!Number.isFinite(v)) {
        box.value = num(p.value);
        return;
      }
      p.value = v;
      // Typed values outside the slider's range stretch the range.
      if (v < p.min) p.min = v;
      if (v > p.max) p.max = v;
      range.min = String(p.min);
      range.max = String(p.max);
      range.value = String(v);
      this.h.onChange('sim');
    });
    row.append(label, range, box);
    return row;
  }

  /** Start radios; x₀ and σ₀ only matter for the Gaussian start. */
  private syncStart(): void {
    const s = this.s;
    for (const r of document.querySelectorAll<HTMLInputElement>('input[name="start"]')) r.checked = r.value === s.start;
    for (const id of ['x0-range', 'x0', 'sigma0']) $<HTMLInputElement>(id).disabled = s.start === 'stationary';
  }

  private syncInitial(): void {
    const s = this.s;
    const range = $<HTMLInputElement>('x0-range');
    range.min = String(s.domain[0]);
    range.max = String(s.domain[1]);
    range.step = String((s.domain[1] - s.domain[0]) / 400);
    range.value = String(s.x0);
    $<HTMLInputElement>('x0').value = num(s.x0);
    $<HTMLInputElement>('sigma0').value = num(s.sigma0);
  }
}
