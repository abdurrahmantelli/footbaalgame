// ===========================================================================
// scene.js - Three.js WebGL 3D Stadium & Goal Scene (Arm Mali GPU optimized)
// Low-poly geometry, antialias limited to 2x DPR, shadows off for Mali perf.
// ===========================================================================
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js";
import { GOAL_WIDTH, GOAL_HEIGHT, PENALTY_DISTANCE, BALL_RADIUS } from "./physics.js";

export class Stadium {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0f172a);
    this.scene.fog = new THREE.Fog(0x0f172a, 18, 45);
    this.camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 200);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setSize(innerWidth, innerHeight);
    this.renderer.shadowMap.enabled = false;
    container.appendChild(this.renderer.domElement);
    this._lights(); this._pitch(); this._goal(); this._ball(); this._keeper();
    addEventListener("resize", () => this.onResize());
  }
  _lights() {
    const h = new THREE.HemisphereLight(0x88aaff, 0x1e293b, 0.85);
    this.scene.add(h);
    const d = new THREE.DirectionalLight(0xffffff, 0.75);
    d.position.set(6, 12, 4);
    this.scene.add(d);
  }
  _pitch() {
    const g = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({ color: 0x1f3d22, roughness: 1 }));
    g.rotation.x = -Math.PI / 2;
    this.scene.add(g);
    const lm = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const line = (w, h, x, z) => { const m = new THREE.Mesh(new THREE.BoxGeometry(w, 0.02, h), lm); m.position.set(x, 0.02, z); this.scene.add(m); };
    line(GOAL_WIDTH + 6, 0.12, 0, PENALTY_DISTANCE);
    const spot = new THREE.Mesh(new THREE.CircleGeometry(0.25, 16), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    spot.rotation.x = -Math.PI / 2; spot.position.set(0, 0.03, 0);
    this.scene.add(spot);
  }
  _goal() {
    const pm = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.4 });
    const r = 0.06;
    const lp = new THREE.Mesh(new THREE.CylinderGeometry(r, r, GOAL_HEIGHT, 8), pm);
    lp.position.set(-GOAL_WIDTH / 2, GOAL_HEIGHT / 2, PENALTY_DISTANCE);
    this.scene.add(lp);
    const rp = lp.clone(); rp.position.x = GOAL_WIDTH / 2; this.scene.add(rp);
    const bar = new THREE.Mesh(new THREE.CylinderGeometry(r, r, GOAL_WIDTH + r * 2, 8), pm);
    bar.rotation.z = Math.PI / 2; bar.position.set(0, GOAL_HEIGHT, PENALTY_DISTANCE);
    this.scene.add(bar);
    this.zoneMeshes = [];
    const zw = GOAL_WIDTH / 3, zh = GOAL_HEIGHT / 3;
    for (let i = 0; i < 9; i++) {
      const col = i % 3, row = Math.floor(i / 3);
      const zx = -GOAL_WIDTH / 2 + (col + 0.5) * zw, zy = (2.5 - row) * zh;
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(zw, zh), new THREE.MeshBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.18, side: THREE.DoubleSide }));
      mesh.position.set(zx, zy, PENALTY_DISTANCE - 0.02);
      this.scene.add(mesh);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(zw, zh)), new THREE.LineBasicMaterial({ color: 0x64748b, transparent: true, opacity: 0.6 }));
      edges.position.copy(mesh.position);
      this.scene.add(edges);
      this.zoneMeshes.push({ mesh, index: i });
    }
    const net = new THREE.Mesh(new THREE.PlaneGeometry(GOAL_WIDTH, GOAL_HEIGHT, 12, 6), new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.12, side: THREE.DoubleSide }));
    net.position.set(0, GOAL_HEIGHT / 2, PENALTY_DISTANCE + 0.05);
    this.scene.add(net);
  }
  _ball() {
    this.ballMesh = new THREE.Mesh(new THREE.SphereGeometry(BALL_RADIUS, 16, 16), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5 }));
    this.ballMesh.position.set(0, BALL_RADIUS, 0);
    this.scene.add(this.ballMesh);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3000 * 3), 3));
    g.setDrawRange(0, 0);
    this.trajLine = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x60a5fa, transparent: true, opacity: 0.8 }));
    this.scene.add(this.trajLine);
  }
  _keeper() {
    this.keeper = new THREE.Group();
    const jm = new THREE.MeshStandardMaterial({ color: 0xeab308, roughness: 0.7 });
    const pm = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.7 });
    const sm = new THREE.MeshStandardMaterial({ color: 0xfdba74, roughness: 0.8 });
    const gm = new THREE.MeshStandardMaterial({ color: 0xf97316, roughness: 0.6 });
    const torso = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.7, 0.3), jm); torso.position.y = 1.15; this.keeper.add(torso);
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.18, 12, 12), sm); head.position.y = 1.72; this.keeper.add(head);
    const ll = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.8, 0.2), pm); ll.position.set(-0.13, 0.4, 0); this.keeper.add(ll);
    const lr = ll.clone(); lr.position.x = 0.13; this.keeper.add(lr);
    this.armL = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.7, 0.16), jm); this.armL.position.set(-0.33, 1.25, 0); this.keeper.add(this.armL);
    this.armR = this.armL.clone(); this.armR.position.x = 0.33; this.keeper.add(this.armR);
    this.gloveL = new THREE.Mesh(new THREE.SphereGeometry(0.1, 10, 10), gm); this.gloveL.position.set(-0.33, 0.9, 0); this.keeper.add(this.gloveL);
    this.gloveR = this.gloveL.clone(); this.gloveR.position.x = 0.33; this.keeper.add(this.gloveR);
    this.keeper.position.set(0, 1.05, PENALTY_DISTANCE - 0.2);
    this.scene.add(this.keeper);
  }
  updateBall(b) {
    this.ballMesh.position.set(b.x, b.y, b.z);
    const pos = this.trajLine.geometry.attributes.position.array;
    const n = Math.min(b.trajectory.length, 1000);
    for (let i = 0; i < n; i++) { const p = b.trajectory[i]; pos[i * 3] = p.x; pos[i * 3 + 1] = p.y; pos[i * 3 + 2] = p.z; }
    this.trajLine.geometry.setDrawRange(0, n);
    this.trajLine.geometry.attributes.position.needsUpdate = true;
  }
  updateKeeper(kx, ky, diving, dir) {
    this.keeper.position.x = kx; this.keeper.position.y = Math.max(ky, BALL_RADIUS);
    this.keeper.position.z = PENALTY_DISTANCE - 0.2;
    
    // Kalecinin uçuşunu gerçekçi göstermek için gövdeyi atladığı yöne eğeriz.
    if (diving && dir !== 0) {
      this.keeper.rotation.z = -dir * 1.1;
      this.keeper.rotation.y = -dir * 0.25;
    } else {
      this.keeper.rotation.z = 0;
      this.keeper.rotation.y = 0;
    }
    
    const stretch = diving ? 0.5 : 0;
    this.armL.position.x = -0.33 - (dir < 0 ? stretch : 0);
    this.armR.position.x = 0.33 + (dir > 0 ? stretch : 0);
    this.gloveL.position.x = this.armL.position.x;
    this.gloveR.position.x = this.armR.position.x;
  }
  highlightZone(z) {
    for (const m of this.zoneMeshes) {
      if (m.index === z) { m.mesh.material.color.setHex(0x3b82f6); m.mesh.material.opacity = 0.45; }
      else { m.mesh.material.color.setHex(0x334155); m.mesh.material.opacity = 0.18; }
    }
  }
  updateCamera(b) {
    const camDist = 2.2, camHeight = 2 + b.y * 0.3, camLag = b.x * 0.25;
    this.camera.position.set(camLag + b.x * 0.08, camHeight, b.z - camDist);
    this.camera.lookAt(b.x * 0.35, 1 + b.y * 0.2, PENALTY_DISTANCE);
  }
  getGoalTargetCoords(screenX, screenY) {
    const mx = (screenX / window.innerWidth) * 2 - 1;
    const my = -(screenY / window.innerHeight) * 2 + 1;
    const vec = new THREE.Vector3(mx, my, 0.5);
    vec.unproject(this.camera);
    const dir = vec.sub(this.camera.position).normalize();
    const distance = (PENALTY_DISTANCE - this.camera.position.z) / dir.z;
    const targetX = this.camera.position.x + dir.x * distance;
    const targetY = this.camera.position.y + dir.y * distance;
    return { x: targetX, y: targetY };
  }
  onResize() {
    this.camera.aspect = innerWidth / innerHeight; this.camera.updateProjectionMatrix();
    this.renderer.setSize(innerWidth, innerHeight);
  }
  render() { this.renderer.render(this.scene, this.camera); }
}
