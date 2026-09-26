import loadMujoco from "@mujoco/mujoco";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { asyncBufferFromUrl, parquetReadObjects } from "hyparquet";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import "./style.css";
import { boundPlaybackArena } from "./model-xml.js";

const DEMO_MODEL = `<mujoco model="recorded-state-demo">
  <option timestep="0.02"/>
  <visual><headlight ambient=".35 .35 .35" diffuse=".7 .7 .7"/></visual>
  <worldbody>
    <geom type="plane" size="4 4 .1" rgba=".16 .18 .22 1"/>
    <body name="cart" pos="0 0 .2">
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="y" type="slide" axis="0 1 0"/>
      <joint name="yaw" type="hinge" axis="0 0 1"/>
      <geom type="box" size=".28 .18 .12" rgba=".18 .55 .95 1"/>
      <body pos=".28 0 .08"><geom type="sphere" size=".08" rgba="1 .55 .12 1"/></body>
    </body>
  </worldbody>
</mujoco>`;

const requestSimilar = document.querySelector("#request-similar");
const recordingSource = new URLSearchParams(location.search).get("src");
if (recordingSource) requestSimilar.href = `/vis/?src=${encodeURIComponent(recordingSource)}`;

function demoEpisode() {
  const fps = 30;
  const qpos = [];
  for (let i = 0; i < 360; i += 1) {
    const t = i / fps;
    qpos.push([1.25 * Math.sin(t * .7), .7 * Math.sin(t * 1.4), t * .65]);
  }
  return { title: "Built-in state-only demo", fps, qpos };
}

async function loadRecording() {
  const source = new URLSearchParams(location.search).get("src");
  if (!source) return { modelXml: DEMO_MODEL, episode: demoEpisode() };
  const manifestUrl = new URL(source, location.href);
  const manifest = await fetch(manifestUrl, { credentials: "same-origin", cache: "no-store" }).then(checkResponse).then(r => r.json());
  const resolve = value => new URL(value, manifestUrl).href;
  const [modelXml, episode] = await Promise.all([
    fetch(resolve(manifest.model), { credentials: "same-origin", cache: "no-store" }).then(checkResponse).then(r => r.text()),
    fetch(resolve(manifest.episode), { credentials: "same-origin", cache: "no-store" }).then(checkResponse).then(r => r.json()),
  ]);
  const assetPaths = manifest.assets || [];
  const assets = [];
  let cacheHits = 0;
  for (let offset = 0; offset < assetPaths.length; offset += 8) {
    const batch = assetPaths.slice(offset, offset + 8);
    const loaded = await Promise.all(batch.map(async path => {
      const result = await fetchCachedAsset(resolve(path), path);
      if (result.cached) cacheHits += 1;
      return { name: path, data: new Uint8Array(await result.response.arrayBuffer()) };
    }));
    assets.push(...loaded);
    status.textContent = `Loading model assets ${Math.min(offset + batch.length, assetPaths.length)} / ${assetPaths.length} · ${cacheHits} cached`;
  }
  episode.title ??= manifest.title;
  return {
    modelXml,
    episode,
    assets,
    lerobot: manifest.lerobot ? {
      parquet: resolve(manifest.lerobot.parquet),
      info: resolve(manifest.lerobot.info),
    } : null,
    metadata: manifest.metadata || null,
  };
}

async function fetchCachedAsset(url, assetPath) {
  if (!("caches" in window)) {
    return { response: await fetch(url, { credentials: "same-origin" }).then(checkResponse), cached: false };
  }
  const cache = await caches.open("vla-mujoco-assets-v1");
  // Packaged asset basenames start with a content hash. Using that stable name
  // lets different rollouts share one cached Panda/LIBERO scene asset.
  const filename = assetPath.split("/").pop();
  const cacheKey = new URL(`/vis/.asset-cache/${encodeURIComponent(filename)}`, location.origin).href;
  const cached = await cache.match(cacheKey);
  if (cached) return { response: cached, cached: true };
  const response = await fetch(url, { credentials: "same-origin" }).then(checkResponse);
  await cache.put(cacheKey, response.clone());
  return { response, cached: false };
}

function checkResponse(response) {
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}: ${response.url}`);
  return response;
}

class RolloutViewer {
  constructor(mujoco, container, modelXml, episode, assets = []) {
    this.mujoco = mujoco;
    this.container = container;
    this.episode = episode;
    const vfs = new mujoco.MjVFS();
    for (const asset of assets) vfs.addBuffer(asset.name, asset.data);
    this.model = mujoco.MjModel.from_xml_string(boundPlaybackArena(modelXml), vfs);
    vfs.delete();
    if (!this.model) throw new Error("MuJoCo could not load the model XML");
    this.data = new mujoco.MjData(this.model);
    this.option = new mujoco.MjvOption();
    this.perturb = new mujoco.MjvPerturb();
    this.cameraSpec = new mujoco.MjvCamera();
    this.mjScene = new mujoco.MjvScene(this.model, 16384);
    if (!Array.isArray(episode.qpos) || !episode.qpos.length) throw new Error("Episode has no qpos frames");
    if (episode.qpos[0].length !== this.model.nq) throw new Error(`qpos width ${episode.qpos[0].length} does not match model nq ${this.model.nq}`);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x090b0f);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    container.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(45, 1, .01, 1000);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(2.2, -2.2, 1.6);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0, .5);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 2.2));
    const key = new THREE.DirectionalLight(0xffffff, 2.4);
    key.position.set(-3, -4, 6);
    key.castShadow = true;
    this.scene.add(key);
    this.meshes = [];
    this.geometryCache = new Map();
    this.textureCache = new Map();
    for (const [points, color] of [[episode.referenceRoot, 0x38bdf8], [episode.actualRoot, 0xfbbf24]]) {
      if (points?.length) {
        const geometry = new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(p[0],p[1],.025)));
        this.scene.add(new THREE.Line(geometry,new THREE.LineBasicMaterial({color})));
      }
    }
    if (episode.qpos[0]?.length >= 7) {
      const p = episode.qpos[0];
      this.controls.target.set(p[0],p[1],.5);
      this.camera.position.set(p[0]+2.2,p[1]-2.2,1.6);
    }
    this.frame = 0;
    this.playing = false;
    this.lastTick = performance.now();
    this.accumulator = 0;
    this.onFrame = null;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.setFrame(0);
    this.useModelCamera("agentview");
  }

  useModelCamera(name) {
    const cameraId = this.mujoco.mj_name2id(this.model, this.mujoco.mjtObj.mjOBJ_CAMERA.value, name);
    if (cameraId < 0) return;
    const position = this.data.cam_xpos.slice(3 * cameraId, 3 * cameraId + 3);
    const matrix = this.data.cam_xmat.slice(9 * cameraId, 9 * cameraId + 9);
    this.camera.position.set(...position);
    this.camera.up.set(matrix[1], matrix[4], matrix[7]);
    this.controls.target.set(position[0] - 1.5 * matrix[2], position[1] - 1.5 * matrix[5], position[2] - 1.5 * matrix[8]);
    this.camera.fov = Number(this.model.cam_fovy[cameraId]) || 45;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  isCollisionProxy(geom) {
    if (geom.objtype !== this.mujoco.mjtObj.mjOBJ_GEOM.value || geom.objid < 0) return false;
    if (Number(this.model.geom_group[geom.objid]) !== 0) return false;
    const name = this.mujoco.mj_id2name(this.model, this.mujoco.mjtObj.mjOBJ_GEOM.value, geom.objid) || "";
    return geom.rgba[3] < .99 || /(?:collision|_col(?:$|_))/.test(name);
  }

  geometry(geom) {
    const modelGeom = geom.objtype === this.mujoco.mjtObj.mjOBJ_GEOM.value && geom.objid >= 0;
    const dataId = modelGeom ? Number(this.model.geom_dataid[geom.objid]) : Number(geom.dataid);
    const key = JSON.stringify([geom.type, [...geom.size], dataId]);
    if (this.geometryCache.has(key)) return this.geometryCache.get(key);
    const kind = this.mujoco.mjtGeom;
    let shape;
    if (geom.type === kind.mjGEOM_PLANE.value) shape = new THREE.PlaneGeometry(200, 200);
    else if (geom.type === kind.mjGEOM_SPHERE.value) shape = new THREE.SphereGeometry(geom.size[0], 24, 16);
    else if (geom.type === kind.mjGEOM_BOX.value) shape = new THREE.BoxGeometry(2 * geom.size[0], 2 * geom.size[1], 2 * geom.size[2]);
    else if (geom.type === kind.mjGEOM_CYLINDER.value) { shape = new THREE.CylinderGeometry(geom.size[0], geom.size[0], 2 * geom.size[2], 24); shape.rotateX(Math.PI / 2); }
    else if (geom.type === kind.mjGEOM_CAPSULE.value) { shape = new THREE.CapsuleGeometry(geom.size[0], 2 * geom.size[2], 8, 16); shape.rotateX(Math.PI / 2); }
    else if (geom.type === kind.mjGEOM_ELLIPSOID.value) { shape = new THREE.SphereGeometry(1, 24, 16); shape.scale(...geom.size); }
    else if (geom.type === kind.mjGEOM_MESH.value) {
      const meshId = dataId;
      const vertexAddress = this.model.mesh_vertadr[meshId];
      const faceAddress = this.model.mesh_faceadr[meshId];
      const faceCount = this.model.mesh_facenum[meshId];
      const normalAddress = this.model.mesh_normaladr[meshId];
      const textureAddress = this.model.mesh_texcoordadr[meshId];
      const positions = new Float32Array(faceCount * 9);
      const normals = new Float32Array(faceCount * 9);
      const uvs = textureAddress >= 0 ? new Float32Array(faceCount * 6) : null;
      for (let face = 0; face < faceCount; face += 1) {
        for (let corner = 0; corner < 3; corner += 1) {
          const faceOffset = 3 * (faceAddress + face) + corner;
          const vertex = vertexAddress + this.model.mesh_face[faceOffset];
          const normal = normalAddress + this.model.mesh_facenormal[faceOffset];
          positions.set(this.model.mesh_vert.slice(3 * vertex, 3 * vertex + 3), 9 * face + 3 * corner);
          normals.set(this.model.mesh_normal.slice(3 * normal, 3 * normal + 3), 9 * face + 3 * corner);
          if (uvs) {
            const texture = textureAddress + this.model.mesh_facetexcoord[faceOffset];
            uvs[6 * face + 2 * corner] = this.model.mesh_texcoord[2 * texture];
            uvs[6 * face + 2 * corner + 1] = 1 - this.model.mesh_texcoord[2 * texture + 1];
          }
        }
      }
      shape = new THREE.BufferGeometry();
      shape.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      shape.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
      if (uvs) shape.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
    }
    else shape = new THREE.BufferGeometry();
    this.geometryCache.set(key, shape);
    return shape;
  }

  texture(geom) {
    if (geom.texid < 0) return null;
    const repeatX = Math.max(Number(geom.texrepeat[0]) || 1, 1);
    const repeatY = Math.max(Number(geom.texrepeat[1]) || 1, 1);
    const key = `${geom.texid}:${repeatX}:${repeatY}`;
    if (this.textureCache.has(key)) return this.textureCache.get(key);
    const width = Number(this.model.tex_width[geom.texid]);
    const height = Number(this.model.tex_height[geom.texid]);
    const channels = Number(this.model.tex_nchannel[geom.texid]);
    const address = Number(this.model.tex_adr[geom.texid]);
    const source = this.model.tex_data.slice(address, address + width * height * channels);
    const pixels = new Uint8Array(width * height * 4);
    for (let pixel = 0; pixel < width * height; pixel += 1) {
      const sourceOffset = pixel * channels;
      const targetOffset = pixel * 4;
      if (channels === 1) {
        pixels[targetOffset] = source[sourceOffset];
        pixels[targetOffset + 1] = source[sourceOffset];
        pixels[targetOffset + 2] = source[sourceOffset];
      } else {
        pixels[targetOffset] = source[sourceOffset];
        pixels[targetOffset + 1] = source[sourceOffset + 1];
        pixels[targetOffset + 2] = source[sourceOffset + 2];
      }
      pixels[targetOffset + 3] = channels === 4 ? source[sourceOffset + 3] : 255;
    }
    const texture = new THREE.DataTexture(pixels, width, height, THREE.RGBAFormat, THREE.UnsignedByteType);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(repeatX, repeatY);
    texture.needsUpdate = true;
    this.textureCache.set(key, texture);
    return texture;
  }

  setFrame(index) {
    this.frame = Math.max(0, Math.min(Math.round(index), this.episode.qpos.length - 1));
    this.data.qpos.set(this.episode.qpos[this.frame]);
    if (this.episode.qvel?.[this.frame]) this.data.qvel.set(this.episode.qvel[this.frame]);
    if (this.episode.ctrl?.[this.frame]) this.data.ctrl.set(this.episode.ctrl[this.frame]);
    this.data.time = this.episode.time?.[this.frame] ?? this.frame / (this.episode.fps || 30);
    this.mujoco.mj_forward(this.model, this.data);
    timeline.value = this.frame;
    step.value = `${this.frame + 1} / ${this.episode.qpos.length}`;
    this.onFrame?.(this.frame);
  }

  updateScene() {
    this.mujoco.mjv_updateScene(this.model, this.data, this.option, this.perturb, this.cameraSpec, this.mujoco.mjtCatBit.mjCAT_ALL.value, this.mjScene);
    const geoms = this.mjScene.geoms;
    const modelGeomType = this.mujoco.mjtObj.mjOBJ_GEOM.value;
    const seen = new Set();
    for (let i = 0; i < geoms.size(); i += 1) {
      const geom = geoms.get(i);
      const geomId = Number(geom.objid);
      if (geom.objtype !== modelGeomType || geomId < 0 || geomId >= this.model.ngeom) {
        geom.delete();
        continue;
      }
      let mesh = this.meshes[geomId];
      // Match LIBERO's competition camera output: render the visual geometry,
      // not robosuite's group-0 collision proxies.
      if (this.isCollisionProxy(geom)) {
        if (mesh) mesh.visible = false;
        geom.delete();
        continue;
      }
      if (!mesh) {
        const material = new THREE.MeshPhongMaterial({
          color: new THREE.Color(geom.rgba[0], geom.rgba[1], geom.rgba[2]),
          map: this.texture(geom),
          opacity: geom.rgba[3],
          transparent: geom.rgba[3] < 1,
          shininess: Math.max(0, Math.min(100, geom.shininess * 100)),
          specular: new THREE.Color().setScalar(Math.max(0, Math.min(1, geom.specular))),
        });
        mesh = new THREE.Mesh(this.geometry(geom), material);
        mesh.userData.mujocoType = geom.type;
        mesh.castShadow = mesh.receiveShadow = true;
        mesh.matrixAutoUpdate = false;
        this.meshes[geomId] = mesh;
        this.scene.add(mesh);
      }
      seen.add(geomId);
      mesh.visible = true;
      const positionOffset = 3 * geomId;
      const matrixOffset = 9 * geomId;
      const position = this.data.geom_xpos;
      const matrix = this.data.geom_xmat;
      mesh.matrix.set(
        matrix[matrixOffset], matrix[matrixOffset + 1], matrix[matrixOffset + 2], position[positionOffset],
        matrix[matrixOffset + 3], matrix[matrixOffset + 4], matrix[matrixOffset + 5], position[positionOffset + 1],
        matrix[matrixOffset + 6], matrix[matrixOffset + 7], matrix[matrixOffset + 8], position[positionOffset + 2],
        0, 0, 0, 1,
      );
      mesh.matrixWorldNeedsUpdate = true;
      geom.delete();
    }
    for (let geomId = 0; geomId < this.meshes.length; geomId += 1) {
      if (this.meshes[geomId] && !seen.has(geomId)) this.meshes[geomId].visible = false;
    }
    geoms.delete();
  }

  resize() {
    const { width, height } = this.container.getBoundingClientRect();
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / Math.max(height, 1);
    this.camera.updateProjectionMatrix();
  }

  run() {
    const draw = now => {
      const elapsed = Math.min((now - this.lastTick) / 1000, .25);
      this.lastTick = now;
      if (this.playing) {
        this.accumulator += elapsed * Number(speed.value);
        const period = 1 / (this.episode.fps || 30);
        while (this.accumulator >= period) {
          this.accumulator -= period;
          if (this.frame >= this.episode.qpos.length - 1) this.playing = false;
          else this.setFrame(this.frame + 1);
        }
        play.textContent = this.playing ? "Pause" : "Play";
      }
      this.controls.update();
      this.updateScene();
      this.renderer.render(this.scene, this.camera);
      this.animation = requestAnimationFrame(draw);
    };
    this.resize();
    this.animation = requestAnimationFrame(draw);
  }
}

const viewerElement = document.querySelector("#viewer");
const status = document.querySelector("#status");
const title = document.querySelector("#title");
const play = document.querySelector("#play");
const restart = document.querySelector("#restart");
const timeline = document.querySelector("#timeline");
const step = document.querySelector("#step");
const speed = document.querySelector("#speed");
const metricsElement = document.querySelector("#metrics");
const metricsStatus = document.querySelector("#metrics-status");
const metricsFrame = document.querySelector("#metrics-frame");
const chartsElement = document.querySelector("#charts");
const metadataContent = document.querySelector("#metadata-content");

function renderMetadata(metadata) {
  if (!metadata) return;
  metadataContent.replaceChildren();
  for (const [section, value] of Object.entries(metadata)) {
    const details = document.createElement("details");
    details.open = section !== "special_config";
    const summary = document.createElement("summary");
    summary.textContent = section.replaceAll("_", " ");
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(value, null, 2);
    details.append(summary, pre);
    metadataContent.appendChild(details);
  }
}

const CHART_GROUPS = [
  { title: "End-effector velocity", columns: ["ee_velocity"], unit: "m/s" },
  { title: "End-effector acceleration", columns: ["ee_acceleration"], unit: "m/s²" },
  { title: "End-effector jerk", columns: ["ee_jerk"], unit: "m/s³" },
  { title: "Keypoint distances", columns: ["keypoint_distance"], unit: "m" },
];
const ACTION_NAMES = ["ΔX", "ΔY", "ΔZ", "ΔRoll", "ΔPitch", "ΔYaw", "Gripper"];
const ACTION_PARTITIONS = [
  { label: "executed", column: "executed_action_chunk", color: "#3b82f6" },
  { label: "inpainted", column: "inpainted_action_chunk", color: "#facc15" },
  { label: "discarded", column: "discarded_action_chunk", color: "#7c8594" },
];
const COLORS = ["#60a5fa", "#f59e0b", "#34d399", "#f472b6", "#a78bfa", "#fb7185", "#22d3ee", "#facc15"];
const PLOT_HEIGHT = 300;

function chunkBoundaryPlugin(boundaries) {
  return {
    hooks: {
      draw: [plot => {
        const { ctx, bbox } = plot;
        ctx.save();
        ctx.strokeStyle = "#596170";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 5]);
        for (const boundary of boundaries) {
          const x = Math.round(plot.valToPos(boundary, "x", true)) + .5;
          if (x <= bbox.left || x >= bbox.left + bbox.width) continue;
          ctx.beginPath();
          ctx.moveTo(x, bbox.top);
          ctx.lineTo(x, bbox.top + bbox.height);
          ctx.stroke();
        }
        ctx.restore();
      }],
    },
  };
}

function vector(value) {
  if (value == null) return [];
  if (Array.isArray(value)) return value.flat(Infinity).map(Number);
  if (ArrayBuffer.isView(value)) return Array.from(value, Number);
  return [Number(value)];
}

async function startMetrics(lerobot, player) {
  if (!lerobot) {
    const actual = player.episode.actualRoot;
    const reference = player.episode.referenceRoot;
    if (!actual?.length || !reference?.length) {
      metricsStatus.textContent = 'No root trajectory telemetry is available.';
      return;
    }
    const count = Math.min(actual.length, reference.length), fps = player.episode.fps || 50;
    const times = Array.from({length:count}, (_,i) => i/fps);
    const xy = times.map((_,i) => Math.hypot(actual[i][0]-reference[i][0],actual[i][1]-reference[i][1]));
    const speed = points => times.map((_,i) => i ? Math.hypot(points[i][0]-points[i-1][0],points[i][1]-points[i-1][1])*fps : 0);
    const plots = [];
    for (const [name, labels, values] of [['Root XY error (m)', ['error'], [xy]], ['Horizontal speed (m/s)', ['actual','reference'], [speed(actual),speed(reference)]]]) {
      const card = document.createElement('article'); card.className = 'chart-card';
      const heading = document.createElement('h3'); heading.textContent = name;
      const plotElement = document.createElement('div'); card.append(heading,plotElement); chartsElement.append(card);
      plots.push(new uPlot({width:Math.max(300,card.clientWidth-24),height:250,
        scales:{x:{time:false}},axes:[{stroke:'#9aa4b2'},{stroke:'#9aa4b2'}],
        series:[{label:'seconds'},...labels.map((label,i)=>({label,stroke:['#fbbf24','#38bdf8'][i]}))]},
        [times,...values],plotElement));
    }
    player.onFrame = frame => {
      const index = Math.min(frame,count-1);
      metricsFrame.textContent = (index/fps).toFixed(2)+' s · XY error '+xy[index].toFixed(3)+' m';
      for (const plot of plots) plot.setCursor({left:plot.valToPos(index/fps,'x'),top:0});
    };
    metricsStatus.textContent = 'Recorded root motion; speeds use adjacent 50 Hz samples.';
    player.onFrame(0);
    return;
  }
  metricsStatus.textContent = "Reading LeRobot telemetry…";
  const info = await fetch(lerobot.info, { credentials: "same-origin" }).then(checkResponse).then(r => r.json());
  const available = info.features || {};
  const predictionColumns = Object.keys(available).filter(name => /prediction|predicted|head|probability|completion/.test(name));
  const groups = [...CHART_GROUPS, ...(predictionColumns.length ? [{ title: "Prediction heads", columns: predictionColumns, unit: "value" }] : [])]
    .map(group => ({ ...group, columns: group.columns.filter(name => available[name]) }))
    .filter(group => group.columns.length);
  if (!groups.length) throw new Error("This LeRobot episode has no chartable telemetry columns");
  const actionPartitions = ACTION_PARTITIONS.filter(partition => available[partition.column]);
  const columns = [
    "timestamp",
    ...new Set([
      ...groups.flatMap(group => group.columns),
      ...actionPartitions.map(partition => partition.column),
      ...(actionPartitions.length ? ["replan_index", "action_chunk_index"] : []),
    ]),
  ];
  const file = await asyncBufferFromUrl({ url: lerobot.parquet, requestInit: { credentials: "same-origin" } });
  const rows = await parquetReadObjects({ file, columns });
  const times = rows.map((row, index) => Number(row.timestamp ?? index / (player.episode.fps || 30)));
  const plots = [];
  for (const group of groups) {
    const series = [];
    for (const column of group.columns) {
      const width = Math.max(...rows.slice(0, 8).map(row => vector(row[column]).length), 1);
      const names = available[column]?.names;
      for (let component = 0; component < width; component += 1) {
        series.push({
          label: width === 1 ? column : `${column}.${names?.[component] ?? component}`,
          values: rows.map(row => vector(row[column])[component] ?? null),
        });
      }
    }
    const visibleSeries = series.filter(item => item.values.some(value => Number.isFinite(value)));
    if (!visibleSeries.length) continue;
    const card = document.createElement("article");
    card.className = "chart-card";
    card.innerHTML = `<h3>${group.title}<small>${group.unit}</small></h3><div class="plot"></div>`;
    chartsElement.appendChild(card);
    const plot = new uPlot({
      width: card.clientWidth - 24,
      height: PLOT_HEIGHT,
      legend: { show: true, live: true },
      cursor: { drag: { x: true, y: false, setScale: false } },
      scales: { x: { time: false } },
      axes: [
        { label: "time (s)", stroke: "#9aa4b2", grid: { stroke: "#252b35", width: 1 } },
        { label: group.unit, size: 58, stroke: "#9aa4b2", grid: { stroke: "#252b35", width: 1 } },
      ],
      series: [{ label: "time" }, ...visibleSeries.map((item, i) => ({ label: item.label, stroke: COLORS[i % COLORS.length], width: 1.5 }))],
      hooks: { setCursor: [plotInstance => {
        if (plotInstance.cursor.idx == null || plotInstance.cursor._playbackSync) return;
        player.playing = false;
        player.setFrame(plotInstance.cursor.idx);
        play.textContent = "Play";
      }] },
    }, [times, ...visibleSeries.map(item => item.values)], card.querySelector(".plot"));
    plots.push(plot);
  }
  const actionPlots = [];
  if (actionPartitions.length) {
    const fps = player.episode.fps || 30;
    const partitionOffsets = [];
    let horizon = 0;
    for (const partition of actionPartitions) {
      partitionOffsets.push(horizon);
      horizon += Number(available[partition.column].shape?.[0] || 0);
    }
    const replanRows = rows.filter((row, index) =>
      Number(row.action_chunk_index ?? (index === 0 ? 0 : 1)) === 0);
    const chunkBoundaries = replanRows.slice(1).map(row => Number(row.timestamp));
    const actionTimes = Array.from({ length: rows.length + horizon - 1 }, (_, index) => index / fps);
    for (let action = 0; action < ACTION_NAMES.length; action += 1) {
      const chunkSeries = [];
      const chunkValues = [];
      for (let replan = 0; replan < replanRows.length; replan += 1) {
        const row = replanRows[replan];
        const startFrame = Math.round(Number(row.timestamp || 0) * fps);
        const fullChunk = actionPartitions.flatMap(partition => vector(row[partition.column]));
        actionPartitions.forEach((partition, partitionIndex) => {
          const shape = available[partition.column].shape || [];
          const length = Number(shape[0]);
          const width = Number(shape[1] || ACTION_NAMES.length);
          const startOffset = partitionOffsets[partitionIndex];
          const lineStart = partitionIndex === 0 ? startOffset : startOffset - 1;
          const values = Array(actionTimes.length).fill(null);
          for (let offset = lineStart; offset < startOffset + length; offset += 1) {
            values[startFrame + offset] = fullChunk[offset * width + action] ?? null;
          }
          chunkValues.push(values);
          chunkSeries.push({
            label: `${partition.label} · chunk ${replan}`,
            stroke: partition.color,
            width: 1.6,
            points: { show: true, size: 3.5, fill: partition.color },
          });
        });
      }
      const card = document.createElement("article");
      card.className = "chart-card";
      card.innerHTML = `<h3><span>Action · ${ACTION_NAMES[action]}</span><span class="partition-key"><i class="executed"></i>executed <i class="inpainted"></i>inpainted <i class="discarded"></i>discarded</span></h3><div class="plot"></div>`;
      chartsElement.appendChild(card);
      const plot = new uPlot({
        width: card.clientWidth - 24,
        height: PLOT_HEIGHT,
        plugins: [chunkBoundaryPlugin(chunkBoundaries)],
        legend: { show: false },
        cursor: { drag: { x: true, y: false, setScale: false } },
        scales: { x: { time: false } },
        axes: [
          { label: "time (s)", stroke: "#9aa4b2", grid: { stroke: "#252b35", width: 1 } },
          { label: "command", size: 58, stroke: "#9aa4b2", grid: { stroke: "#252b35", width: 1 } },
        ],
        series: [{ label: "time" }, ...chunkSeries],
        hooks: { setCursor: [plotInstance => {
          if (plotInstance.cursor.idx == null || plotInstance.cursor._playbackSync) return;
          player.playing = false;
          player.setFrame(Math.min(plotInstance.cursor.idx, rows.length - 1));
          play.textContent = "Play";
        }] },
      }, [actionTimes, ...chunkValues], card.querySelector(".plot"));
      actionPlots.push(plot);
    }
  }
  metricsStatus.remove();
  const syncCharts = frame => {
    metricsFrame.textContent = `frame ${frame + 1} · ${times[frame]?.toFixed(2) ?? "—"} s`;
    for (const plot of plots) {
      plot.cursor._playbackSync = true;
      plot.setCursor({ left: plot.valToPos(times[frame], "x"), top: -10 }, false);
      plot.cursor._playbackSync = false;
    }
    for (const plot of actionPlots) {
      plot.cursor._playbackSync = true;
      plot.setCursor({ left: plot.valToPos(times[frame], "x"), top: -10 }, false);
      plot.cursor._playbackSync = false;
    }
  };
  player.onFrame = syncCharts;
  syncCharts(player.frame);
  const observer = new ResizeObserver(entries => {
    const width = Math.floor(entries[0].contentRect.width);
    for (const plot of plots) plot.setSize({ width: Math.max(300, width - 2), height: PLOT_HEIGHT });
    for (const plot of actionPlots) plot.setSize({ width: Math.max(300, width - 2), height: PLOT_HEIGHT });
  });
  observer.observe(chartsElement);
}

try {
  const [{ modelXml, episode, assets, lerobot, metadata }, mujoco] = await Promise.all([loadRecording(), loadMujoco()]);
  const player = new RolloutViewer(mujoco, viewerElement, modelXml, episode, assets);
  window.k1Player = player;
  title.textContent = episode.title || "MuJoCo rollout";
  status.textContent = `${player.model.nq} DoF · ${episode.qpos.length} states · state-only playback`;
  timeline.max = episode.qpos.length - 1;
  timeline.disabled = play.disabled = restart.disabled = false;
  timeline.addEventListener("input", () => { player.playing = false; player.setFrame(Number(timeline.value)); });
  play.addEventListener("click", () => { if (player.frame === episode.qpos.length - 1) player.setFrame(0); player.playing = !player.playing; play.textContent = player.playing ? "Pause" : "Play"; });
  restart.addEventListener("click", () => { player.playing = false; player.setFrame(0); play.textContent = "Play"; });
  window.addEventListener("keydown", event => { if (event.code === "Space" && event.target === document.body) { event.preventDefault(); play.click(); } });
  player.run();
  renderMetadata(metadata);
  startMetrics(lerobot, player).catch(error => {
    console.error(error);
    metricsStatus.textContent = `LeRobot telemetry failed: ${error.message}`;
    metricsStatus.classList.add("error");
  });
} catch (error) {
  console.error(error);
  status.textContent = error.message;
  status.classList.add("error");
}
