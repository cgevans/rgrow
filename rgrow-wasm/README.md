# rgrow-wasm

WebAssembly bindings for the [rgrow](../) tile-assembly simulator, with a
small static HTML/JS front-end. Runs the same simulation core as the
desktop GUI, in a browser tab, with no server.

## Try it

After CI publishes, the interface is available at
<https://cge.codeberg.page/rgrow/app/>. To run locally:

```sh
# Build, serve, and open in your default browser:
just wasm-serve

# Or, manually:
wasm-pack build rgrow-wasm --out-dir web/pkg --target web --release
python -m http.server 8000 -d rgrow-wasm/web
# then open http://localhost:8000/

# Headless smoke test under Node:
just wasm-test
```

## What's in here

- `src/lib.rs` — the `Sim` JS class. Owns a `(SystemEnum, StateEnum)`
  pair and exposes step / render / parameter methods to JS.
- `web/index.html`, `web/app.js`, `web/style.css` — the UI. Parameter
  controls are generated dynamically from `sim.parameters()` so any
  model that lists tunable parameters works without UI changes.
- `web/examples/*.{yaml,json}` — bundled examples so the demo runs on first
  visit with no upload.

## Architecture

The desktop GUI is implemented as **two processes** communicating over a
Unix socket plus shared memory: one runs the simulator and renders
RGBA frames into an `mmap`'d region, the other (an iced GUI) reads
those frames and shows them. In the browser we collapse that into a
single wasm module:

| Desktop                   | Browser                          |
|---------------------------|----------------------------------|
| simulator subprocess      | wasm `Sim` instance              |
| GUI (iced) subprocess     | DOM + `<canvas>`                 |
| Unix socket / named pipe  | direct method calls              |
| `mmap` shared frame       | `Uint8ClampedArray` + putImageData |
| iced `image::viewer`      | `<canvas>` with `image-rendering: pixelated` |

Rendering goes through the same `painter` module as the desktop path
(see `rgrow::painter::render_frame_dyn`), so the two views can't drift.

## Build details

The wasm crate sits as a third workspace member alongside `rgrow-rust`
and `rgrow-python`. It depends on `rgrow` with `default-features =
false` (no `gui`, no `parallel`, no `python`). Two `getrandom` versions
are explicitly forced into their JS-backed feature so `rand` works in
the browser.

`wasm-pack build` produces a `pkg/` directory containing the JS shim,
the `.wasm` binary, and TypeScript types. We build directly into
`web/pkg/` so the `web/` directory is the deployable artifact.

For a quick smoke test under Node:

```sh
wasm-pack build rgrow-wasm --out-dir pkg-node --target nodejs --dev
node rgrow-wasm/smoke_test.cjs
```

## Non-goals (for now)

- Parallel FFS / committor analysis in the browser.
- WebGL / WebGPU acceleration. Software RGBA → `<canvas>` is the
  contract.
- Web Worker / `SharedArrayBuffer` threading. Main-thread only;
  per-frame step budget is bounded so the UI stays responsive.
- A tileset editor. Paste, file upload, or built-in examples only.
