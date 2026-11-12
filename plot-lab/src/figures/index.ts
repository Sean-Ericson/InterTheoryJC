import { intervalPage } from '../pages/interval/page';
import type { PageDef } from '../pages/types';
import { doubleWell, lncos, lncosh, physical, radial, solvable } from './later';
import { PART1 } from './part1';
import { PART2 } from './part2';
import { PART3 } from './part3';
import { PART6 } from './part6';
import { reductions } from './reductions';
import { TOOLKIT } from './toolkit';

/**
 * Every figure of the notes (fpe-examples), in the notes' order. Each page
 * is one section: its figure(s) with the parameters the notes fixed made
 * adjustable. Free diffusion with absorbing walls is the hand-built page with
 * the exit-time simulation, placed after the reflecting walls as in the notes.
 */
const [hlReflecting, hlAbsorbing, reflecting, ...restOfPart1] = PART1;

export const FIGURES: PageDef[] = [
  physical,
  ...TOOLKIT,
  hlReflecting,
  hlAbsorbing,
  reflecting,
  intervalPage,
  ...restOfPart1,
  ...PART2,
  ...PART3,
  solvable,
  radial,
  lncosh,
  lncos,
  doubleWell,
  ...PART6,
  reductions,
];
