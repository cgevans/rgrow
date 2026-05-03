// Quick smoke test: load the Node-target wasm build, run Sim end-to-end.
//
// Run: node rgrow-wasm/smoke_test.cjs
//
// This is a one-shot script for local verification, not part of the
// shipped artefacts. The Node target is built separately with
//   wasm-pack build rgrow-wasm --out-dir pkg-node --target nodejs

const fs = require("fs");
const path = require("path");
const { Sim } = require("./pkg-node/rgrow_wasm.js");

const yaml = fs.readFileSync(
  path.join(__dirname, "web/examples/sierpinski.yaml"),
  "utf8",
);

const sim = Sim.fromYaml(yaml);
console.log("model:", sim.modelName());
console.log("canvas:", sim.canvasSize());
console.log("frame@8px:", sim.frameSize(8));

const before = Number(sim.totalEvents());
const stepResult = sim.stepForEvents(10000n);
console.log("step result:", stepResult);
const after = Number(sim.totalEvents());
console.log(`events advanced by ${after - before}`);

const bytes = sim.renderAlloc(8, true);
console.log(`rendered ${bytes.length} bytes`);
console.log("first pixel RGBA:", Array.from(bytes.slice(0, 4)));

// Hash a few stats so we can spot drift if this is rerun.
let nonzero = 0;
for (let i = 3; i < bytes.length; i += 4) {
  if (bytes[i] !== 0) nonzero++;
}
console.log(`opaque (non-transparent) pixels: ${nonzero}/${bytes.length / 4}`);

const params = sim.parameters();
console.log("parameters:", params);

// Set a parameter and confirm it sticks.
const orig = params.find((p) => p.name === "g_se").current_value;
sim.setParameter("g_se", orig - 0.5);
const updated = sim.parameters().find((p) => p.name === "g_se");
console.log(`g_se: ${orig} -> ${updated.current_value} (after setParameter)`);
