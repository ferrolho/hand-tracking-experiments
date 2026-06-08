// Real-time hand tracking in the browser: MediaPipe Tasks Vision (webcam) -> Three.js 3D scene.
// Client-side only (static-hostable). Mirrors the look/controls of the local viser app.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
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
const PLANE_W = 1.6;

const params = {
  smoothing: true, minCutoff: 1.0, beta: 0.5,
  cameraFeed: true, selfieMirror: true, fov: 35,
  planeDistance: 1.0, jointSize: 0.01, boneWidth: 3.0,
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
  camera.position.set(0, 0, 1.3);
  controls.target.set(0, 0, -0.4);
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

// Bones: fat lines (so the width slider actually has an effect).
const bonesGeo = new LineSegmentsGeometry();
const bonesMat = new LineMaterial({ vertexColors: true, linewidth: params.boneWidth });
bonesMat.resolution.set(window.innerWidth, window.innerHeight);
const bones = new LineSegments2(bonesGeo, bonesMat);
bones.frustumCulled = false;
bones.visible = false;
scene.add(bones);

// Webcam feed plane (created once video dimensions are known).
let videoTexture = null, plane = null, planeAspect = 16 / 9;
let spreadX = PLANE_W * 0.45, spreadY = (PLANE_W / planeAspect) * 0.45;

function makePlane() {
  plane = new THREE.Mesh(
    new THREE.PlaneGeometry(PLANE_W, PLANE_W),
    new THREE.MeshBasicMaterial({ map: videoTexture, toneMapped: false }));
  scene.add(plane);
}

function updatePlane() {
  if (!plane) return;
  plane.visible = params.cameraFeed;
  const h = PLANE_W / planeAspect;
  plane.scale.set((params.selfieMirror ? -1 : 1) * 1, h / PLANE_W, 1);
  plane.position.set(0, 0, -params.planeDistance);
}

// ---------- GUI ----------
const gui = new GUI({ title: 'hand tracking' });
const sm = gui.addFolder('Smoothing (One-Euro)');
sm.add(params, 'smoothing').name('enabled');
sm.add(params, 'minCutoff', 0.1, 5, 0.05).name('min cutoff');
sm.add(params, 'beta', 0, 5, 0.05).name('beta');
gui.add(params, 'cameraFeed').name('camera feed');
gui.add(params, 'selfieMirror').name('selfie mirror');
gui.add(params, 'fov', 5, 90, 1).name('field of view').onChange((v) => {
  camera.fov = v; camera.updateProjectionMatrix();
});
const ap = gui.addFolder('Appearance');
ap.add(params, 'planeDistance', 0.2, 3, 0.05).name('plane distance');
ap.add(params, 'jointSize', 0.002, 0.03, 0.001).name('joint size');
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

      // Articulation (metric, flip y/z to viser-like frame) + global offset from image position.
      const wrist = image[0];
      const ox = (wrist.x - 0.5) * spreadX;
      const oy = -(wrist.y - 0.5) * spreadY;
      let j = new Float32Array(63);
      for (let k = 0; k < NJ; k++) {
        j[3 * k] = world[k].x + ox;
        j[3 * k + 1] = -world[k].y + oy;
        j[3 * k + 2] = -world[k].z;
      }
      if (params.smoothing) j = getFilter(label).filter(j, t);
      if (params.selfieMirror) for (let k = 0; k < NJ; k++) j[3 * k] = -j[3 * k];
      slotJoints[label] = j;
    }
  }
  if (!params.smoothing) filters.clear();

  // Update instanced joints + accumulate bone segments.
  const segPos = [], segCol = [];
  SLOTS.forEach((label, si) => {
    const base = si * NJ;
    const j = slotJoints[label];
    const col = HAND_COLORS[label] || DEFAULT_COLOR;
    for (let k = 0; k < NJ; k++) {
      if (j) { dummy.position.set(j[3 * k], j[3 * k + 1], j[3 * k + 2]); dummy.scale.setScalar(params.jointSize); }
      else { dummy.position.set(0, 0, 0); dummy.scale.setScalar(0); }
      dummy.updateMatrix();
      jointMesh.setMatrixAt(base + k, dummy.matrix);
      jointMesh.setColorAt(base + k, tmpColor.setRGB(col[0], col[1], col[2]));
    }
    if (j) {
      for (const [a, b] of HAND_CONNECTIONS) {
        segPos.push(j[3 * a], j[3 * a + 1], j[3 * a + 2], j[3 * b], j[3 * b + 1], j[3 * b + 2]);
        segCol.push(col[0], col[1], col[2], col[0], col[1], col[2]);
      }
    }
  });
  jointMesh.instanceMatrix.needsUpdate = true;
  if (jointMesh.instanceColor) jointMesh.instanceColor.needsUpdate = true;

  if (segPos.length) {
    bonesGeo.setPositions(segPos);
    bonesGeo.setColors(segCol);
    bones.visible = true;
  } else {
    bones.visible = false;
  }
}

// ---------- main loop ----------
let handLandmarker = null;
let lastVideoTime = -1;

function loop() {
  requestAnimationFrame(loop);
  controls.update();
  updatePlane();
  bonesMat.linewidth = params.boneWidth;

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
  planeAspect = (video.videoWidth / video.videoHeight) || 16 / 9;
  spreadX = PLANE_W * 0.45;
  spreadY = (PLANE_W / planeAspect) * 0.45;
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
  bonesMat.resolution.set(window.innerWidth, window.innerHeight);
});
