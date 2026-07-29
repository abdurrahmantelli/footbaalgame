// ===========================================================================
// physics.js - 3D Ball Trajectory (Magnus Effect + Drag + Gravity)
// JS port of core/generate_data.py + demo/test_simulation.py
// ===========================================================================
export const GRAVITY = 9.81;
export const AIR_DENSITY = 1.2;
export const BALL_MASS = 0.43;
export const BALL_RADIUS = 0.11;
export const BALL_AREA = Math.PI * (BALL_RADIUS ** 2);
export const DRAG_COEFF = 0.25;
export const MAGNUS_COEFF = 0.15;
export const GOAL_WIDTH = 7.32;
export const GOAL_HEIGHT = 2.44;
export const PENALTY_DISTANCE = 11.0;

export class Ball {
  constructor() { this.reset(); }
  reset() {
    this.x = 0.0; this.y = BALL_RADIUS; this.z = 0.0;
    this.vx = 0.0; this.vy = 0.0; this.vz = 0.0;
    this.spin = 0.0; this.active = false; this.trajectory = [];
  }
  launch(vx0, vy0, vz0, spin) {
    this.vx = vx0; this.vy = vy0; this.vz = vz0;
    this.spin = spin; this.active = true; this.trajectory = [];
  }
  update(dt = 0.005) {
    if (!this.active) return;
    this.trajectory.push({ x: this.x, y: this.y, z: this.z });
    const vMag = Math.sqrt(this.vx ** 2 + this.vy ** 2 + this.vz ** 2);
    if (vMag === 0) { this.active = false; return; }
    const fDragX = -0.5 * AIR_DENSITY * DRAG_COEFF * BALL_AREA * vMag * this.vx;
    const fDragY = -0.5 * AIR_DENSITY * DRAG_COEFF * BALL_AREA * vMag * this.vy;
    const fDragZ = -0.5 * AIR_DENSITY * DRAG_COEFF * BALL_AREA * vMag * this.vz;
    const fMagX = MAGNUS_COEFF * AIR_DENSITY * BALL_AREA * BALL_RADIUS * this.spin * this.vz;
    const fMagZ = -MAGNUS_COEFF * AIR_DENSITY * BALL_AREA * BALL_RADIUS * this.spin * this.vx;
    const ax = (fDragX + fMagX) / BALL_MASS;
    const ay = -GRAVITY + fDragY / BALL_MASS;
    const az = (fDragZ + fMagZ) / BALL_MASS;
    this.vx += ax * dt; this.vy += ay * dt; this.vz += az * dt;
    this.x += this.vx * dt; this.y += this.vy * dt; this.z += this.vz * dt;
    if (this.y < BALL_RADIUS && this.vy < 0) { this.y = BALL_RADIUS; this.vy = -this.vy * 0.4; }
    if (this.z >= PENALTY_DISTANCE) {
      if (!this.saved && this.z < PENALTY_DISTANCE + 0.4) {
        this.vx *= 0.8; this.vy = Math.min(this.vy, 0) - 1.5; this.vz *= 0.1;
      } else {
        this.active = false;
      }
    }
  }
}

export function getGoalZone(x, y) {
  if (x < -GOAL_WIDTH / 2 || x > GOAL_WIDTH / 2) return -1;
  if (y < 0 || y > GOAL_HEIGHT) return -1;
  let col;
  if (x < -GOAL_WIDTH / 6) col = 0; else if (x > GOAL_WIDTH / 6) col = 2; else col = 1;
  let row;
  if (y > (2 * GOAL_HEIGHT / 3)) row = 0; else if (y < (GOAL_HEIGHT / 3)) row = 2; else row = 1;
  return row * 3 + col;
}

export const ZONE_LABELS = [
  "Sol Üst", "Orta Üst", "Sağ Üst",
  "Sol Orta", "Tam Orta", "Sağ Orta",
  "Sol Alt", "Orta Alt", "Sağ Alt",
];

export function zoneToTargetXY(zoneIndex) {
  const col = zoneIndex % 3;
  const row = Math.floor(zoneIndex / 3);
  const targetX = (col - 1) * (GOAL_WIDTH / 3.0);
  const targetY = GOAL_HEIGHT - (row + 0.5) * (GOAL_HEIGHT / 3.0);
  return { x: targetX, y: targetY };
}
