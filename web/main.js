// Real-time hand tracking in the browser: MediaPipe Tasks Vision (webcam) -> Three.js 3D scene.
// Client-side only (static-hostable). Mirrors the look/controls of the local viser app.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import GUI from 'lil-gui';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { OneEuroFilter } from './one_euro.js';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

// Standard MediaPipe 21-keypoint hand topology (joint index pairs = bones).
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const SLOTS = ['Right', 'Left'];
const HAND_COLORS = { Right: [0.31, 0.67, 1.0], Left: [1.0, 0.55, 0.31] };
const DEFAULT_COLOR = [0.7, 0.7, 0.7];
const NJ = 21;

// Normalized intrinsics of the MacBook Air M2 FaceTime HD camera (from our calibration),
// stored as fractions so they scale to whatever resolution the browser delivers.
// Square-ish pixels => f ≈ 0.74·width (matches the ~0.75·width rule of thumb); cx/W, cy/H.
const CALIB = { fxN: 950.6523 / 1280, fyN: 949.5324 / 1280, cxN: 648.1649 / 1280, cyN: 365.2777 / 720 };

function median(a) {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y), m = s.length >> 1;
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

const params = {
  smoothing: true, minCutoff: 3.0, beta: 5.0,
  cameraFeed: true, selfieMirror: true, fov: 42,
  planeDistance: 2.0, jointSize: 0.004, boneWidth: 7.0,
  resetView: () => resetView(),
};

const statusEl = document.getElementById('status');
const video = document.getElementById('cam');

// ---------- Three.js scene ----------
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x101114);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(params.fov, window.innerWidth / window.innerHeight, 0.01, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

function resetView() {
  // Reproduce the real camera: viewport AT the optical center looking down -z.
  // From here the metric skeleton overlays the feed; orbit away to see the 3D.
  camera.position.set(0, 0, 0);
  controls.target.set(0, 0, -1);
  camera.fov = params.fov;
  camera.updateProjectionMatrix();
  controls.update();
}
resetView();

// Joints: one instanced sphere mesh, two hands x 21 joints.
const jointMesh = new THREE.InstancedMesh(
  new THREE.SphereGeometry(1, 12, 8), new THREE.MeshBasicMaterial(), SLOTS.length * NJ);
jointMesh.frustumCulled = false;
scene.add(jointMesh);
const dummy = new THREE.Object3D();
const tmpColor = new THREE.Color();

// Bones: instanced cylinders (regular meshes => render reliably; width via the slider).
const NB = HAND_CONNECTIONS.length;
const boneMesh = new THREE.InstancedMesh(
  new THREE.CylinderGeometry(1, 1, 1, 8), new THREE.MeshBasicMaterial(), SLOTS.length * NB);
boneMesh.frustumCulled = false;
scene.add(boneMesh);
const Y_AXIS = new THREE.Vector3(0, 1, 0);
const vA = new THREE.Vector3(), vB = new THREE.Vector3(), vMid = new THREE.Vector3(), vDir = new THREE.Vector3();
const quat = new THREE.Quaternion();

// Webcam feed plane + intrinsics (set once video dimensions are known).
let videoTexture = null, plane = null;
let vidW = 1280, vidH = 720;
let fx = 0, fy = 0, cx = 0, cy = 0;

function makePlane() {
  plane = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    new THREE.MeshBasicMaterial({ map: videoTexture, toneMapped: false }));
  scene.add(plane);
}

function updatePlane() {
  if (!plane || !fx) return;
  plane.visible = params.cameraFeed;
  const d = params.planeDistance;
  const rw = d * vidW / fx, rh = d * vidH / fy;   // fill the calibrated frustum at depth d
  let px = (vidW / 2 - cx) * d / fx;               // principal-point offset
  const py = -(vidH / 2 - cy) * d / fy;
  if (params.selfieMirror) px = -px;
  plane.scale.set((params.selfieMirror ? -1 : 1) * rw, rh, 1);
  plane.position.set(px, py, -d);
}

// ---------- GUI ----------
const gui = new GUI({ title: 'hand tracking' });
const sm = gui.addFolder('Smoothing (One-Euro)');
sm.add(params, 'smoothing').name('enabled');
sm.add(params, 'minCutoff', 0.1, 5, 0.05).name('min cutoff');
sm.add(params, 'beta', 0, 5, 0.05).name('beta');
gui.add(params, 'cameraFeed').name('camera feed');
gui.add(params, 'selfieMirror').name('selfie mirror');
const fovCtrl = gui.add(params, 'fov', 5, 90, 1).name('field of view').onChange((v) => {
  camera.fov = v; camera.updateProjectionMatrix();
});
const ap = gui.addFolder('Appearance');
ap.add(params, 'planeDistance', 0.2, 3, 0.05).name('plane distance');
ap.add(params, 'jointSize', 0.001, 0.012, 0.0005).name('joint size');
ap.add(params, 'boneWidth', 1, 10, 0.5).name('bone width');
gui.add(params, 'resetView').name('reset view');

// ---------- hand state ----------
const filters = new Map(); // label -> OneEuroFilter

function getFilter(label) {
  let f = filters.get(label);
  if (!f) { f = new OneEuroFilter(params.minCutoff, params.beta); filters.set(label, f); }
  f.minCutoff = params.minCutoff; f.beta = params.beta;
  return f;
}

function updateHands(res, t) {
  const slotJoints = {};
  if (res && res.worldLandmarks && res.worldLandmarks.length) {
    for (let i = 0; i < res.worldLandmarks.length; i++) {
      const world = res.worldLandmarks[i];
      const image = res.landmarks[i];
      let label = res.handednesses?.[i]?.[0]?.categoryName;
      if (!SLOTS.includes(label)) label = SLOTS[i % SLOTS.length];

      // Metric placement: estimate wrist depth from known hand size vs. pixel size,
      // back-project the wrist, and offset the (flipped) articulation to sit there.
      const ratios = [];
      for (const [a, b] of HAND_CONNECTIONS) {
        const md = Math.hypot(world[a].x - world[b].x, world[a].y - world[b].y, world[a].z - world[b].z);
        const pd = Math.hypot((image[a].x - image[b].x) * vidW, (image[a].y - image[b].y) * vidH);
        if (pd > 1) ratios.push(fx * md / pd);
      }
      const Z = ratios.length ? median(ratios) : 0.5;
      const u0 = image[0].x * vidW, v0 = image[0].y * vidH;
      const tx = (u0 - cx) * Z / fx, ty = -(v0 - cy) * Z / fy, tz = -Z;  // target wrist (three frame)
      const offx = tx - world[0].x, offy = ty + world[0].y, offz = tz + world[0].z;
      let j = new Float32Array(63);
      for (let k = 0; k < NJ; k++) {
        j[3 * k] = world[k].x + offx;
        j[3 * k + 1] = -world[k].y + offy;
        j[3 * k + 2] = -world[k].z + offz;
      }
      if (params.smoothing) j = getFilter(label).filter(j, t);
      if (params.selfieMirror) for (let k = 0; k < NJ; k++) j[3 * k] = -j[3 * k];
      slotJoints[label] = j;
    }
  }
  if (!params.smoothing) filters.clear();

  // Update instanced joints (spheres) and bones (cylinders).
  const boneRadius = params.boneWidth * 0.0005;
  SLOTS.forEach((label, si) => {
    const jb = si * NJ, bb = si * NB;
    const j = slotJoints[label];
    const col = HAND_COLORS[label] || DEFAULT_COLOR;
    tmpColor.setRGB(col[0], col[1], col[2]);

    for (let k = 0; k < NJ; k++) {
      dummy.quaternion.identity();
      if (j) { dummy.position.set(j[3 * k], j[3 * k + 1], j[3 * k + 2]); dummy.scale.setScalar(params.jointSize); }
      else { dummy.position.set(0, 0, 0); dummy.scale.setScalar(0); }
      dummy.updateMatrix();
      jointMesh.setMatrixAt(jb + k, dummy.matrix);
      jointMesh.setColorAt(jb + k, tmpColor);
    }

    for (let c = 0; c < NB; c++) {
      if (j) {
        const a = HAND_CONNECTIONS[c][0], b = HAND_CONNECTIONS[c][1];
        vA.set(j[3 * a], j[3 * a + 1], j[3 * a + 2]);
        vB.set(j[3 * b], j[3 * b + 1], j[3 * b + 2]);
        vDir.subVectors(vB, vA);
        const len = vDir.length() || 1e-6;
        vMid.addVectors(vA, vB).multiplyScalar(0.5);
        quat.setFromUnitVectors(Y_AXIS, vDir.normalize());
        dummy.position.copy(vMid); dummy.quaternion.copy(quat); dummy.scale.set(boneRadius, len, boneRadius);
      } else {
        dummy.position.set(0, 0, 0); dummy.quaternion.identity(); dummy.scale.setScalar(0);
      }
      dummy.updateMatrix();
      boneMesh.setMatrixAt(bb + c, dummy.matrix);
      boneMesh.setColorAt(bb + c, tmpColor);
    }
  });
  jointMesh.instanceMatrix.needsUpdate = true;
  boneMesh.instanceMatrix.needsUpdate = true;
  if (jointMesh.instanceColor) jointMesh.instanceColor.needsUpdate = true;
  if (boneMesh.instanceColor) boneMesh.instanceColor.needsUpdate = true;
}

// ---------- main loop ----------
let handLandmarker = null;
let lastVideoTime = -1;

function loop() {
  requestAnimationFrame(loop);
  controls.update();
  updatePlane();

  if (handLandmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const res = handLandmarker.detectForVideo(video, performance.now());
    updateHands(res, performance.now() / 1000);
    if (videoTexture) videoTexture.needsUpdate = true;
  }
  renderer.render(scene, camera);
}

async function init() {
  statusEl.textContent = 'Requesting camera…';
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
  video.srcObject = stream;
  await video.play();
  vidW = video.videoWidth || 1280;
  vidH = video.videoHeight || 720;
  fx = CALIB.fxN * vidW; fy = CALIB.fyN * vidW;
  cx = CALIB.cxN * vidW; cy = CALIB.cyN * vidH;
  // Default FOV = the calibrated camera's FOV, so the skeleton overlays the feed dead-on.
  const calibFov = THREE.MathUtils.radToDeg(2 * Math.atan((vidH / 2) / fy));
  fovCtrl.setValue(Math.round(calibFov));
  resetView();
  videoTexture = new THREE.VideoTexture(video);
  videoTexture.colorSpace = THREE.SRGBColorSpace;
  makePlane();

  statusEl.textContent = 'Loading hand model…';
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numHands: 2,
  });
  statusEl.style.display = 'none';
  loop();
}

// Camera permission prompts are most reliable after a user gesture.
function start() {
  window.removeEventListener('click', start);
  init().catch((e) => {
    statusEl.style.display = '';
    statusEl.textContent = 'Error: ' + (e.message || e);
    console.error(e);
  });
}
window.addEventListener('click', start);

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
});
