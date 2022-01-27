/// <reference types='@webgpu/types' />
/// 2021 - Ignacio E. Losiggio
/// Originally based on this example: https://austin-eng.com/webgpu-samples/samples/helloTriangle
/// Original idea: https://flam3.com/flame_draves.pdf
/// Also a nice writeup: https://iquilezles.org/www/articles/ifsfractals/ifsfractals.htm
///
/// Stages:
///   1. Generate random points
///   2. Play the chaos game for some time
///   3. Gather the maximum value
///   4. Plot on the log-density display

// Import colourmaps generated from: https://github.com/tritoke/libcmap
import cmaps from './colourmaps.js'

class Config {
  buffer = new ArrayBuffer(24)

  _origin = new Float32Array(this.buffer, 0, 2)
  get x()      { return this._origin[0] }
  set x(value) { this._origin[0] = value }
  get y()      { return this._origin[1] }
  set y(value) { this._origin[1] = value }

  _dimensions = new Uint32Array(this.buffer, 8, 2)
  get width()       { return this._dimensions[0] }
  set width(value)  { this._dimensions[0] = value }
  get height()      { return this._dimensions[1] }
  set height(value) { this._dimensions[1] = value }

  _frame = new Uint32Array(this.buffer, 16, 1)
  get frame()      { return this._frame[0] }
  set frame(value) { this._frame[0] = value }

  _zoom = new Float32Array(this.buffer, 20, 1)
  get zoom()      { return this._zoom[0] }
  set zoom(value) { this._zoom[0] = value }
}

class StructWithFlexibleArrayElement {
  static get SIZE() { return this.BASE_SIZE + this.Element.SIZE * this.MAX_ELEMENTS }
  buffer = new ArrayBuffer(this.constructor.BASE_SIZE + this.constructor.Element.SIZE * this.constructor.MAX_ELEMENTS)
  constructor() {
    const proto = Object.getPrototypeOf(this)
    if (!proto.hasOwnProperty('length'))
      this.length = 0
  }

  add(props) {
    if (this.length === this.constructor.MAX_ELEMENTS)
      throw new Error(`${this.constructor.Element.name} limit exceeded!`)
    const view = new DataView(this.buffer, this.constructor.BASE_SIZE + this.constructor.Element.SIZE * this.length, this.constructor.Element.SIZE)
    const element = this[this.length] = new this.constructor.Element(view)
    Object.assign(element, props)
    this.length++
    return element
  }

  findIndex(obj) {
    for (let i = 0; i < this.length; i++)
      if (this[i] === obj)
        return i
  }

  remove(from, to) {
    if (typeof from === 'object')
      from = this.findIndex(from)
    if (typeof to === 'object')
      to = this.findIndex(to) + 1
    else if (to === undefined)
      to = from + 1

    if (from === undefined || to === undefined)
      throw new Error('Start and/or end objects were not found')

    const byte_view = new Uint8Array(this.buffer)
    byte_view.copyWithin(
      this.constructor.BASE_SIZE + this.constructor.Element.SIZE * from,
      this.constructor.BASE_SIZE + this.constructor.Element.SIZE * to,
      this.constructor.BASE_SIZE + this.constructor.Element.SIZE * this.length
    )
    Array.prototype.splice.call(this, from, to - from)
    for (let i = from; i < this.length; i++)
      this[i].view = new DataView(this.buffer, this.constructor.BASE_SIZE + this.constructor.Element.SIZE * i)
  }
}

const VARIATION_ID_TO_STR_ENTRIES = [
  [ 0, 'linear'],
  [ 1, 'sinusoidal'],
  [13, 'julia'],
  [27, 'eyefish']
];
const VARIATION_ID_TO_STR = new Map(VARIATION_ID_TO_STR_ENTRIES)
const STR_TO_VARIATION_ID = new Map(VARIATION_ID_TO_STR_ENTRIES.map(([a, b]) => [b, a]))
class XForm {
  static SIZE = 32
  constructor(view) { this.view = view }

  get variation() {
    const id = this.view.getUint32(0, true)
    const result = VARIATION_ID_TO_STR.get(id)
    if (result === undefined) throw new Error(`Unknown variation id ${id}`)
    return result
  }

  set variation(value) {
    const id = STR_TO_VARIATION_ID.get(value)
    if (id === undefined) throw new Error(`Unknown id for variation string '${value}'`)
    this.view.setUint32(0, id, true)
  }

  get color() { return this.view.getFloat32(4, true) }
  set color(value) { return this.view.setFloat32(4, value, true) }

  get a()      { return this.view.getFloat32( 8, true) }
  get b()      { return this.view.getFloat32(12, true) }
  get c()      { return this.view.getFloat32(16, true) }
  get d()      { return this.view.getFloat32(20, true) }
  get e()      { return this.view.getFloat32(24, true) }
  get f()      { return this.view.getFloat32(28, true) }
  set a(value) { this.view.setFloat32( 8, value, true) }
  set b(value) { this.view.setFloat32(12, value, true) }
  set c(value) { this.view.setFloat32(16, value, true) }
  set d(value) { this.view.setFloat32(20, value, true) }
  set e(value) { this.view.setFloat32(24, value, true) }
  set f(value) { this.view.setFloat32(28, value, true) }
}

class Fractal extends StructWithFlexibleArrayElement {
  static BASE_SIZE = 4
  static MAX_ELEMENTS = 128
  static Element = XForm

  _length = new Uint32Array(this.buffer, 0, 1)
  get length()      { return this._length[0] }
  set length(value) { return this._length[0] = value }
}

class Colors extends StructWithFlexibleArrayElement {
  static BASE_SIZE = 0
  static MAX_ELEMENTS = 256
  static Element = class Color {
    static SIZE = 16
    constructor(view) { this.view = view }
    get r() { return this.view.getFloat32( 0, true) }
    get g() { return this.view.getFloat32( 4, true) }
    get b() { return this.view.getFloat32( 8, true) }
    get a() { return this.view.getFloat32(12, true) }
    set r(value) { this.view.setFloat32( 0, value, true) }
    set g(value) { this.view.setFloat32( 4, value, true) }
    set b(value) { this.view.setFloat32( 8, value, true) }
    set a(value) { this.view.setFloat32(12, value, true) }
  }
}
class Circles extends StructWithFlexibleArrayElement {
  static BASE_SIZE = 0
  static MAX_ELEMENTS = 128
  static Element = class Circle {
    static SIZE = 16
    constructor(view) { this.view = view }
    get x() { return this.view.getFloat32(0, true) }
    get y() { return this.view.getFloat32(4, true) }
    get r() { return this.view.getFloat32(8, true) }
    set x(value) { this.view.setFloat32(0, value, true) }
    set y(value) { this.view.setFloat32(4, value, true) }
    set r(value) { this.view.setFloat32(8, value, true) }

    contains(point) {
      return (this.x - point.x) ** 2 + (this.y - point.y) ** 2 < (this.r / flam3.config.zoom) ** 2
    }
  }
}
class Lines extends StructWithFlexibleArrayElement {
  static BASE_SIZE = 0
  static MAX_ELEMENTS = 128
  static Element = class Line {
    static SIZE = 24
    constructor(view) { this.view = view }
    get from_x() { return this.view.getFloat32( 0, true) }
    get from_y() { return this.view.getFloat32( 4, true) }
    get to_x()   { return this.view.getFloat32( 8, true) }
    get to_y()   { return this.view.getFloat32(12, true) }
    get width()  { return this.view.getFloat32(16, true) }
    set from_x(value) { this.view.setFloat32( 0, value, true) }
    set from_y(value) { this.view.setFloat32( 4, value, true) }
    set to_x(value)   { this.view.setFloat32( 8, value, true) }
    set to_y(value)   { this.view.setFloat32(12, value, true) }
    set width(value)  { this.view.setFloat32(16, value, true) }

    squared_distance_to(point) {
      const pa_x = point.x   - this.from_x
      const pa_y = point.y   - this.from_y
      const ba_x = this.to_x - this.from_x
      const ba_y = this.to_y - this.from_y
      const unclamped_h = (pa_x * ba_x + pa_y * ba_y)
                        / (ba_x **2 + ba_y ** 2)
      const h = clamp(unclamped_h, 0, 1)
      return (pa_x - ba_x * h) ** 2 + (pa_y - ba_y * h) ** 2
    }

    contains(point) {
      return this.squared_distance_to(point) < (this.width * 4 / flam3.config.zoom) ** 2
    }
  }
}
class Primitives extends StructWithFlexibleArrayElement {
  static BASE_SIZE = 4
  static MAX_ELEMENTS = 256
  static Element = class Primitive {
    static SIZE = 4
    constructor(view) { this._view = view }

    get view() { return this._view }
    set view(value) {
      this._view = value
      this.index = this.shape.view.byteOffset / this.shape.constructor.SIZE
    }

    get kind()  { return this.view.getUint16(0, true) }
    get index() { return this.view.getUint16(2, true) }
    set kind(value)  { this.view.setUint16(0, value, true) }
    set index(value) { this.view.setUint16(2, value, true) }
  }

  constructor() {
    super()
    this.colors = new Colors
    this.lines = new Lines
    this.circles = new Circles
  }

  add(props) {
    const { kind, color, shape } = props
    const collection = kind === 0 ? this.lines : this.circles
    return super.add({
      kind,
      index: collection.length,
      color: this.colors.add(color),
      shape: collection.add(shape)
    })
  }

  _length = new Uint32Array(this.buffer, 0, 1)
  get length()      { return this._length[0] }
  set length(value) { this._length[0] = value }
}

class CMap extends StructWithFlexibleArrayElement {
  static BASE_SIZE = 4
  static MAX_ELEMENTS = 1024
  static Element = class Color {
    static SIZE = 4
    constructor(view) { this._view = view }

    get view() { return this._view }
    set view(value) {
      this._view = value
    }

    get r() { return this.view.getUint8(0, true) }
    get g() { return this.view.getUint8(1, true) }
    get b() { return this.view.getUint8(2, true) }
    set r(value) { this.view.setUint8(0, value, true) }
    set g(value) { this.view.setUint8(1, value, true) }
    set b(value) { this.view.setUint8(2, value, true) }
  }

  copyFrom(cmapUint8Array) {
    if ((cmapUint8Array.length & 0x3) !== 0)
      throw new Error('Length should be multiple of four')
    const newLength = cmapUint8Array.length / 4
    if (this.length > newLength)
      this.length = newLength
    while (this.length < newLength) this.add()
    const srcArray = new Uint32Array(cmapUint8Array.buffer)
    const dstArray = new Uint32Array(this.buffer, this.constructor.BASE_SIZE);
    srcArray.forEach((v, i) => dstArray[i] = v)
  }

  _length = new Float32Array(this.buffer, 0, 1)
  get length()      { return this._length[0] }
  set length(value) { this._length[0] = value }
}

const common_code = `
[[block]] struct Stage1Histogram {
  max: atomic<u32>;
  padding1: u32; padding2: u32; padding3: u32;
  data: array<atomic<u32>>;
};

[[block]] struct Stage2Histogram {
  max: atomic<u32>;
  data: array<vec4<u32>>;
};

[[block]] struct FragmentHistogram {
  max: u32;
  data: array<vec4<u32>>;
};

[[block]] struct CanvasConfiguration {
  origin: vec2<f32>;
  dimensions: vec2<u32>;
  frame: u32;
  zoom: f32;
};

// FIXME: Use a mat3x3
struct AffineTransform {
  a: f32;
  b: f32;
  c: f32;
  d: f32;
  e: f32;
  f: f32;
};

[[block]] struct CMap {
  len: f32;
  colors: array<u32>;
};

struct XForm {
  variation_id: u32;
  color: f32;
  transform: AffineTransform;
};

[[block]] struct Fractal {
  size: u32;
  xforms: array<XForm>;
};

[[group(0), binding(0)]] var<storage, read_write> stage1_histogram: Stage1Histogram;
[[group(0), binding(0)]] var<storage, read_write> stage2_histogram: Stage2Histogram;
[[group(0), binding(0)]] var<storage, read> fragment_histogram: FragmentHistogram;
[[group(0), binding(1)]] var<storage, read> fractal: Fractal;
[[group(0), binding(2)]] var<uniform> config: CanvasConfiguration;
[[group(0), binding(3)]] var<storage, read> cmap: CMap;

// Adapted from: https://drafts.csswg.org/css-color-4/#color-conversion-code
fn gam_sRGB(RGB: vec3<f32>) -> vec3<f32> {
  // convert an array of linear-light sRGB values in the range 0.0-1.0
  // to gamma corrected form
  // https://en.wikipedia.org/wiki/SRGB
  // Extended transfer function:
  // For negative values, linear portion extends on reflection
  // of axis, then uses reflected pow below that
  let sign_per_channel = sign(RGB);
  let abs_RGB = abs(RGB);
  let non_linear_mask = abs_RGB > vec3<f32>(0.0031308);
  let non_linear_RGB = sign_per_channel * (1.055 * pow(RGB, vec3<f32>(1./2.4)) - 0.055);
  let linear_RGB = 12.92 * RGB;
  return select(linear_RGB, non_linear_RGB, non_linear_mask);
}

// From: https://nullprogram.com/blog/2018/07/31/
fn hash(v: u32) -> u32 {
    var x: u32 = v;
    x = x ^ (x >> 17u);
    x = x * 0xED5AD4BBu;
    x = x ^ (x >> 11u);
    x = x * 0xAC4C1B51u;
    x = x ^ (x >> 1u);
    x = x * 0x31848BABu;
    x = x ^ (x >> 14u);
    return x;
}

var<private> random_state: u32;

fn seed(v: u32) { random_state = v; }

fn random() -> u32 {
  random_state = hash(random_state);
  return random_state;
}

fn frandom() -> f32 { return f32(random()) / 0xFFFFFFFF.0; }

fn apply_transform(p: vec2<f32>, transform: AffineTransform) -> vec2<f32> {
  return vec2<f32>(
    transform.a * p.x + transform.b * p.y + transform.c,
    transform.d * p.x + transform.e * p.y + transform.f
  );
}

let PI = 3.1415926535897932384626433;

let LINEAR_FN = 0u;
fn linear(p: vec2<f32>) -> vec2<f32> {
  return p;
}

let SINUSOIDAL_FN = 1u;
fn sinusoidal(p: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(sin(p.x), sin(p.y));
}

let JULIA_FN = 13u;
fn julia(p: vec2<f32>) -> vec2<f32> {
  let phi_over_two = atan2(p.x, p.y) / 2.0;
  let omega = f32((random() & 1u) == 0u) * PI;

  return sqrt(length(p)) * vec2<f32>(
    cos(phi_over_two + omega),
    sin(phi_over_two + omega)
  );
}

let EYEFISH_FN = 27u;
fn eyefish(p: vec2<f32>) -> vec2<f32> {
  let v = 2.0 / (length(p) + 1.0);
  return v * p;
}

fn apply_fn(variation_id: u32, p: vec2<f32>) -> vec2<f32> {
  switch (variation_id) {
    case  0u: { return linear(p);     }
    case  1u: { return sinusoidal(p); }
    case 13u: { return julia(p); }
    case 27u: { return eyefish(p); }
    default: {}
  }
  // Dumb and unreachable
  return vec2<f32>(0.0, 0.0);
}

fn apply_xform(xform: XForm, p: vec2<f32>) -> vec2<f32> {
  return apply_fn(xform.variation_id, apply_transform(p, xform.transform));
}

fn next(p: vec3<f32>) -> vec3<f32> {
  let i = random() % fractal.size;
  let xform = fractal.xforms[i];
  let next_p = apply_xform(xform, p.xy);
  let next_c = (p.z + xform.color) / 2.0;
  return vec3<f32>(next_p, next_c);
}

fn plot(v: vec3<f32>) {
  let p = (v.xy - config.origin) * config.zoom;
  if (-1. <= p.x && p.x < 1. && -1. <= p.y && p.y < 1.) {
    let ipoint = vec2<u32>(
      u32((p.x + 1.) / 2. * f32(config.dimensions.x)),
      u32((p.y + 1.) / 2. * f32(config.dimensions.y))
    );
    let offset = 4u * (ipoint.y * config.dimensions.x + ipoint.x);

    let color = cmap.colors[u32(round(v.z * (cmap.len - 1.0)))];
    let r = (color >>  0u) & 0xFFu;
    let g = (color >>  8u) & 0xFFu;
    let b = (color >> 16u) & 0xFFu;
    atomicAdd(&stage1_histogram.data[offset + 0u], r);
    atomicAdd(&stage1_histogram.data[offset + 1u], g);
    atomicAdd(&stage1_histogram.data[offset + 2u], b);
    atomicAdd(&stage1_histogram.data[offset + 3u], 1u);
  }
}
`

const histogram_max_wgsl = `
${common_code}
[[stage(compute), workgroup_size(1)]]
fn histogram_max(
  [[builtin(global_invocation_id)]] invocation: vec3<u32>,
  [[builtin(num_workgroups)]] invocation_size: vec3<u32>
) {
  // We are only using 1D invocations for now so...
  let CANVAS_SIZE = config.dimensions.x * config.dimensions.y;
  let BLOCK_SIZE = (CANVAS_SIZE + invocation_size.x - 1u) / invocation_size.x;
  let ITERATION_SIZE = min(BLOCK_SIZE * invocation.x + 1u, CANVAS_SIZE);
  var invocation_max: u32 = 0x0u;

  for (var i = BLOCK_SIZE * invocation.x; i < ITERATION_SIZE; i = i + 1u) {
    invocation_max = max(invocation_max, stage2_histogram.data[i].a);
  }

  atomicMax(&stage2_histogram.max, invocation_max);
}
`

const add_points_wgsl = `
${common_code}
// FIXME: Tune the workgroup size
[[stage(compute), workgroup_size(1)]]
fn add_points(
  [[builtin(global_invocation_id)]] invocation: vec3<u32>
) {
  seed(hash(config.frame) ^ hash(invocation.x));
  var point = vec3<f32>(
    frandom() * 2. - 1.,
    frandom() * 2. - 1.,
    frandom()
  );

  for (var i = 0; i < 20; i = i + 1) { point = next(point); }
  for (var i = 20; i < 100; i = i + 1) {
    point = next(point);
    plot(point);
  }
}
`

const vertex_wgsl =`
[[stage(vertex)]]
fn vertex_main([[builtin(vertex_index)]] VertexIndex : u32) -> [[builtin(position)]] vec4<f32> {
  var pos = array<vec2<f32>, 4>(  
    // Upper Triangle
    vec2<f32>( 1.,  1.),
    vec2<f32>(-1.,  1.),
    vec2<f32>( 1., -1.),
    vec2<f32>(-1., -1.)
  );

  return vec4<f32>(pos[VertexIndex], 0.0, 1.0);
}
`

const histogram_fragment_wgsl = `
${common_code}
[[stage(fragment)]]
fn fragment_main([[builtin(position)]] pos: vec4<f32>) -> [[location(0)]] vec4<f32> {
  let point = vec2<u32>(
    u32(pos.x),
    u32(pos.y)
  );
  let i = point.y * config.dimensions.y + point.x;
  let values = vec4<f32>(fragment_histogram.data[i]);
  let log_max_a = log(f32(fragment_histogram.max));
  let color = values.rgb / 255.0 / values.a;
  let alpha = log(values.a) / log_max_a;
  return vec4<f32>(gam_sRGB(color.rgb) * alpha, 1.0);
}
`

const gui_fragment_wgsl = `
${common_code}

let LINE_KIND = 0u;
struct LinePrimitive {
  from: vec2<f32>;
  to: vec2<f32>;
  width: f32;
};

let CIRCLE_KIND = 1u;
struct CirclePrimitive {
  center: vec2<f32>;
  radius: f32;
};

[[block]] struct Primitives {
  length: u32;
  // The primitives are packed:
  //  * lower 16 bits denote the kind
  //  * higher 16 bits denote the index
  data: array<u32>;
};
[[block]] struct Colors  { data: array<vec4<f32>>; };
[[block]] struct Lines   { data: array<LinePrimitive>; };
[[block]] struct Circles { data: array<CirclePrimitive>; };

[[group(1), binding(0)]] var<storage, read> primitives: Primitives;
[[group(1), binding(1)]] var<storage, read> colors: Colors;
[[group(1), binding(2)]] var<storage, read> lines: Lines;
[[group(1), binding(3)]] var<storage, read> circles: Circles;

[[stage(fragment)]]
fn fragment_main([[builtin(position)]] screen_pos: vec4<f32>) -> [[location(0)]] vec4<f32> {
  let dimensions = vec2<f32>(config.dimensions);
  let normal_pos = (screen_pos.xy / dimensions * 2.0 - vec2<f32>(1.0)) / config.zoom + config.origin;
  var result = vec4<f32>(0.0); // Black (Transparent)

  for (var i = 0u; i < primitives.length; i = i + 1u) {
    let primitive = primitives.data[i];
    let primitive_type = primitive & 0x0000FFFFu;
    let primitive_index = primitive >> 16u;
    let color = colors.data[i];
    switch (primitive_type) {
      // src: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
      case 0u /* LINE_KIND */: {
        let line = lines.data[primitive_index];
        let pa = normal_pos - line.from;
        let ba = line.to - line.from;
        let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        let dist = length(pa - ba * h);
        if (dist < line.width / config.zoom / 2.0) {
          result = color;
        }
      }
      case 1u /* CIRCLE_KIND */: {
        let circle = circles.data[primitive_index];
        if (length(normal_pos - circle.center) < circle.radius / config.zoom) {
          result = color;
        }
      }
      default: {}
    }
  }
  return result;
}
`

function project(line, point) {
  const delta_x = line.from_x - line.to_x
  const delta_y = line.from_y - line.to_y
  const squared_length = (delta_x**2 + delta_y**2)
  const k = ((point.x - line.to_x) * delta_x + (point.y - line.to_y) * delta_y) / squared_length
  return {
    x: delta_x * k + line.to_x,
    y: delta_y * k + line.to_y
  }
}

function intersect(l1, l_2) {
  const l1_delta_x = l1.from_x - l1.to_x
  const l1_delta_y = l1.from_y - l1.to_y
  const l2_delta_x = l_2.from_x - l_2.to_x
  const l2_delta_y = l_2.from_y - l_2.to_y
  const l1_k = l1.from_x * l1.to_y - l1.from_y * l1.to_x
  const l2_k = l_2.from_x * l_2.to_y - l_2.from_y * l_2.to_x
  const d = l1_delta_x * l2_delta_y - l1_delta_y * l2_delta_x
  return  {
    x: (l1_k * l2_delta_x - l2_k * l1_delta_x) / d,
    y: (l1_k * l2_delta_y - l2_k * l1_delta_y) / d
  }
}

function clamp(v, min, max) {
  return Math.min(max, Math.max(min, v))
}

const init = async (canvas, starts_running = true) => {
  if (navigator.gpu === undefined) {
    console.error('WebGPU is not supported (or not enabled)')
    document.getElementById('webgpu-not-supported-error').style = ''
    return
  }
  const adapter = await navigator.gpu.requestAdapter()
  if (adapter === null) {
    console.error('No WebGPU device is available')
    document.getElementById('webgpu-no-device-error').style = ''
  }
  const device = await adapter.requestDevice()

  const context = canvas.getContext('webgpu')
  const format = context.getPreferredFormat(adapter)

  context.configure({
    device,
    format: format,
    size: [canvas.width, canvas.height]
  })

  const add_points_module = device.createShaderModule({
      label: 'FLAM3 > Module > Add Points',
      code: add_points_wgsl
  })
  const histogram_max_module = device.createShaderModule({
      label: 'FLAM3 > Module > Hisogram Max',
      code: histogram_max_wgsl
  })
  const vertex_module = device.createShaderModule({
      label: 'FLAM3 > Module > Vertex',
      code: vertex_wgsl
  })
  const histogram_fragment_module = device.createShaderModule({
      label: 'FLAM3 > Module > Histogram Fragment',
      code: histogram_fragment_wgsl
  })
  const gui_fragment_module = device.createShaderModule({
      label: 'FLAM3 > Module > Histogram Fragment',
      code: gui_fragment_wgsl
  })

  const fractalBindGroupLayout = device.createBindGroupLayout({
      label: 'FLAM3 > Bind Group Layout > Fractal',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
          buffer: { type: 'storage' }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' }
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' }
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' }
        }
      ]
  })

  const guiBindGroupLayout = device.createBindGroupLayout({
      label: 'FLAM3 > Bind Group Layout > GUI',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: 'storage' }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: 'storage' }
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: 'storage' }
        },
        {
          binding: 3,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: 'storage' }
        }
      ]
  })

  const fractalPipelineLayout = device.createPipelineLayout({
    label: 'FLAM3 > Pipeline Layout > Fractal',
    bindGroupLayouts: [fractalBindGroupLayout]
  })
  const guiPipelineLayout = device.createPipelineLayout({
    label: 'FLAM3 > Pipeline Layout > GUI',
    bindGroupLayouts: [fractalBindGroupLayout, guiBindGroupLayout]
  })

  const HISTOGRAM_BUFFER_SIZE = 4 + 3 * 4 + 4 * 4 * 900 * 900
  const histogramBuffer = device.createBuffer({
    label: 'FLAM3 > Buffer > Histogram',
    size: HISTOGRAM_BUFFER_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  })
  const cleanHistogramBuffer = device.createBuffer({
    label: 'FLAM3 > Buffer > Clean Histogram',
    size: HISTOGRAM_BUFFER_SIZE,
    usage: GPUBufferUsage.COPY_SRC
  })
  const fractalBuffer = device.createBuffer({
    label: 'FLAM3 > Buffer > Fractal',
    size: Fractal.SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  })
  const configBuffer = device.createBuffer({
    label: 'FLAM3 > Buffer > Configuration',
    size: 24,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  })
  const cmapBuffer = device.createBuffer({
    label: 'FLAM3 > Buffer > CMap',
    size: CMap.SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  })

  const fractalBindGroup = device.createBindGroup({
    label: 'FLAM3 > Group Binding > Fractal',
    layout: fractalBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          label: 'FLAM3 > Group Binding > Fractal > Histogram',
          buffer: histogramBuffer
        }
      },
      {
        binding: 1,
        resource: {
          label: 'FLAM3 > Group Binding > Fractal > Fractal',
          buffer: fractalBuffer
        }
      },
      {
        binding: 2,
        resource: {
          label: 'FLAM3 > Group Binding > Fractal > Configuration',
          buffer: configBuffer
        }
      },
      {
        binding: 3,
        resource: {
          label: 'FLAM3 > Group Binding > Fractal > CMap',
          buffer: cmapBuffer
        }
      }
    ]
  })

  const primitivesBuffer = device.createBuffer({
    label: 'FLAM3 > Buffer > GUI > Primitives',
    size: Primitives.SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  })
  const colorsBuffer = device.createBuffer({
    label: 'FLAM3 > Buffer > GUI > Colors',
    size: Colors.SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  })
  const linesBuffer = device.createBuffer({
    label: 'FLAM3 > Buffer > GUI > Lines',
    size: Lines.SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  })
  const circlesBuffer = device.createBuffer({
    label: 'FLAM3 > Buffer > GUI > Circles',
    size: Circles.SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  })

  const guiBindGroup = device.createBindGroup({
    label: 'FLAM3 > Group Binding > GUI',
    layout: guiBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          label: 'FLAM3 > Group Binding > GUI > Primitives',
          buffer: primitivesBuffer
        }
      },
      {
        binding: 1,
        resource: {
          label: 'FLAM3 > Group Binding > GUI > Colors',
          buffer: colorsBuffer
        }
      },
      {
        binding: 2,
        resource: {
          label: 'FLAM3 > Group Binding > GUI > Lines',
          buffer: linesBuffer
        }
      },
      {
        binding: 3,
        resource: {
          label: 'FLAM3 > Group Binding > GUI > Circles',
          buffer: circlesBuffer
        }
      },
    ]
  })

  const addPointsPipeline = await device.createComputePipelineAsync({
      label: 'FLAM3 > Pipeline > Add points',
      layout: fractalPipelineLayout,
      compute: {
          module: add_points_module,
          entryPoint: 'add_points'
      },
  })

  const histogramMaxPipeline = await device.createComputePipelineAsync({
      label: 'FLAM3 > Pipeline > Histogram Max',
      layout: fractalPipelineLayout,
      compute: {
          module: histogram_max_module,
          entryPoint: 'histogram_max'
      },
  })

  const renderPipeline = await device.createRenderPipelineAsync({
    label: 'FLAM3 > Pipeline > Render',
    layout: fractalPipelineLayout,
    vertex: {
      layout: fractalPipelineLayout,
      module: vertex_module,
      entryPoint: 'vertex_main',
    },
    fragment: {
      layout: fractalPipelineLayout,
      module: histogram_fragment_module,
      entryPoint: 'fragment_main',
      targets: [{ format }]
    },
    primitive: {
      topology: 'triangle-strip',
      stripIndexFormat: 'uint32'
    }
  })

  const guiPipeline = await device.createRenderPipelineAsync({
    label: 'FLAM3 > Pipeline > GUI',
    layout: guiPipelineLayout,
    vertex: {
      layout: guiPipelineLayout,
      module: vertex_module,
      entryPoint: 'vertex_main',
    },
    fragment: {
      layout: guiPipelineLayout,
      module: gui_fragment_module,
      entryPoint: 'fragment_main',
      targets: [{
        format,
        blend: {
          color: {
            srcFactor: 'src-alpha',
            dstFactor: 'one-minus-src-alpha'
          },
          alpha: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha'
          }
        }
      }]
    },
    primitive: {
      topology: 'triangle-strip',
      stripIndexFormat: 'uint32'
    }
  })

  const config = window.config = new Config
  config.width = canvas.width
  config.height = canvas.height
  config.zoom = 1

  const fractal = new Fractal
  //fractal.add({ variation: 'sinusoidal', color: 0, a: 0.5, b:  0.0, c:  0.5, d:  0.0, e: 0.5, f:  0.5 })
  //fractal.add({ variation: 'sinusoidal', color: 0, a: 0.5, b:  0.0, c: -0.5, d:  0.0, e: 0.5, f:  0.5 })
  //fractal.add({ variation: 'sinusoidal', color: 0, a: 0.5, b:  0.0, c:  0.0, d:  0.0, e: 0.5, f: -0.5 })
  //fractal.add({ variation: 'linear',     color: 0, a: 0.0, b:  0.8, c:  0.0, d:  0.6, e: 0.0, f:  0.0 })
  //fractal.add({ variation: 'linear',     color: 0, a: 0.0, b: -0.8, c:  0.0, d: -0.6, e: 0.0, f:  0.0 })

  //fractal.add({ variation: 'eyefish', color: 0, a:  0.321636, b: -0.204179, c: -0.633718, d:  0.204179, e:  0.321637, f:  1.140693 })
  //fractal.add({ variation: 'eyefish', color: 0, a:  0.715673, b: -0.418864, c:  0.576108, d:  0.418864, e:  0.715673, f:  0.455125 })
  //fractal.add({ variation: 'eyefish', color: 1, a: -0.212317, b:  0.536045, c:  0.53578,  d: -0.536045, e: -0.212317, f: -0.743179 })
  //fractal.add({ variation: 'linear',  color: 1, a:  0.7,      b:  0.0,      c:  0.0,      d:  0.0,      e:  0.7,      f:  0.0      })
  fractal.add({ variation: 'linear', color: 0, a:  0.5, b: 0, c:    0, d: 0, e:  0.5, f: -0.5 })
  fractal.add({ variation: 'linear', color: 0, a:  0.5, b: 0, c: -0.5, d: 0, e:  0.5, f:  0.5 })
  fractal.add({ variation: 'linear', color: 1, a:  0.5, b: 0, c:  0.5, d: 0, e:  0.5, f:  0.5 })
  fractal.add({ variation: 'linear', color: 0, a: -2,   b: 0, c:    0, d: 0, e: -2,   f:    0 })

  const cmap = new CMap

  const primitives = new Primitives

  class XFormEditor {
    constructor(xform, primitives, xform_list, color) {
      const elem = document.createElement('xform-editor')
      elem.setAttribute('variation', xform.variation)
      elem.setAttribute('a', xform.a.toFixed(2))
      elem.setAttribute('b', xform.b.toFixed(2))
      elem.setAttribute('c', xform.c.toFixed(2))
      elem.setAttribute('d', xform.d.toFixed(2))
      elem.setAttribute('e', xform.e.toFixed(2))
      elem.setAttribute('f', xform.f.toFixed(2))
      elem.setAttribute('color', xform.color)
      elem.shadowRoot.querySelector('select').onchange = ev => {
        this.xform.variation = ev.currentTarget.value
        flam3.clear()
      }
      elem.shadowRoot.querySelector('input[name="color"]').oninput = ev => {
        this.xform.color = Number.parseFloat(ev.currentTarget.value)
        flam3.clear()
      }
      elem.shadowRoot.querySelector('button').onclick = ev => {
        this.remove(fractal, primitives)
      }
      xform_list.appendChild(elem)
      this.editor = elem

      const base_color = color
      const translucent_color = { ...base_color, a: base_color.a * 0.3 }
      const transparent_black = { r: 0.0, b: 0.0, g: 0.0, a: 0.0 }

      this.xform = xform
      this.current_drag_data = undefined
      this.currently_dragging = undefined

      this.line_00_10 = primitives.add({
        kind: 0,
        color: base_color,
        shape: {
          width: 0.01,
          from_x: xform.c,           from_y: xform.f,
          to_x:   xform.a + xform.c, to_y:   xform.d + xform.f
        }
      })
      this.line_00_01 = primitives.add({
        kind: 0,
        color: base_color,
        shape: {
          width: 0.01,
          from_x: xform.c,           from_y: xform.f,
          to_x:   xform.b + xform.c, to_y:   xform.e + xform.f
        }
      })
      this.line_10_01 = primitives.add({
        kind: 0,
        color: base_color,
        shape: {
          width: 0.01,
          from_x: xform.a + xform.c, from_y: xform.d + xform.f,
          to_x:   xform.b + xform.c, to_y:   xform.e + xform.f
        }
      })
      this.ring_00 = primitives.add({
        kind: 1,
        color: base_color,
        shape: {
          x: xform.c, y: xform.f,
          r: 0.06
        }
      })
      this.hole_00 = primitives.add({
        kind: 1,
        color: translucent_color,
        shape: {
          x: xform.c, y: xform.f,
          r: 0.05
        }
      })
      this.ring_10 = primitives.add({
        kind: 1,
        color: base_color,
        shape: {
          x: xform.a + xform.c, y: xform.d + xform.f,
          r: 0.04
        }
      })
      this.hole_10 = primitives.add({
        kind: 1,
        color: transparent_black,
        shape: {
          x: xform.a + xform.c, y: xform.d + xform.f,
          r: 0.03
        }
      })
      this.ring_01 = primitives.add({
        kind: 1,
        color: base_color,
        shape: {
          x: xform.b + xform.c, y: xform.e + xform.f,
          r: 0.04
        }
      })
      this.hole_01 = primitives.add({
        kind: 1,
        color: transparent_black,
        shape: {
          x: xform.b + xform.c, y: xform.e + xform.f,
          r: 0.03
        }
      })
    }

    pointer_down(point) {
      const draggable_elements = [this.ring_10, this.ring_01, this.ring_00, this.line_10_01, this.line_00_01, this.line_00_10]
      this.currently_dragging = draggable_elements.find(elem => elem.shape.contains(point))
      if (this.currently_dragging === this.line_10_01) {
        const p10_x = this.line_10_01.shape.from_x
        const p10_y = this.line_10_01.shape.from_y
        const p01_x = this.line_10_01.shape.to_x
        const p01_y = this.line_10_01.shape.to_y
        const p00_x = this.ring_00.shape.x
        const p00_y = this.ring_00.shape.y
        this.current_drag_data = {
          // Cache the triangle shape so we can avoid divide-by-zero issues :)
          line_00_01: { from_x: p00_x, from_y: p00_y, to_x: p01_x, to_y: p01_y },
          line_00_10: { from_x: p00_x, from_y: p00_y, to_x: p10_x, to_y: p10_y },
          line_10_01: { from_x: p10_x, from_y: p10_y, to_x: p01_x, to_y: p01_y },
        }
      }
      return this.currently_dragging !== undefined
    }
    pointer_up() {
      this.currently_dragging = undefined
      this.current_drag_data = undefined
    }
    pointer_move(point) {
      if (this.currently_dragging === undefined) return false
      // Translate the whole thing
      if (this.currently_dragging === this.ring_00) {
        this.c = point.x
        this.f = point.y
      }
      // Translate the (0, 1) point
      if (this.currently_dragging === this.ring_01) {
        this.b = point.x - this.xform.c
        this.e = point.y - this.xform.f
      }
      // Translate the (1, 0) point
      if (this.currently_dragging === this.ring_10) {
        this.a = point.x - this.xform.c
        this.d = point.y - this.xform.f
      }
      // Scale the triangle
      if (this.currently_dragging === this.line_10_01) {
        //
        // 10'----A------------P-01'
        //  \     ^            ^ /
        //   \    |            |/
        //    \   |            |
        //     \  |           /|
        //      \ |          / |
        //       \|         /  |
        //       10--------01--P'
        //         \      /
        //          \    /
        //           \  /
        //            00
        const P_prime = project(this.current_drag_data.line_10_01, point)
        const P_to_P_prime = {
          x: point.x - P_prime.x,
          y: point.y - P_prime.y
        }
        const A_to_P = {
          from_x: this.current_drag_data.line_10_01.from_x + P_to_P_prime.x,
          from_y: this.current_drag_data.line_10_01.from_y + P_to_P_prime.y,
          to_x:   point.x,
          to_y:   point.y
        }
        const p10_prime = intersect(this.current_drag_data.line_00_10, A_to_P)
        const p01_prime = intersect(this.current_drag_data.line_00_01, A_to_P)

        if (isFinite(p10_prime.x) && isFinite(p10_prime.y) && isFinite(p01_prime.x) && isFinite(p01_prime.y)) {
          this.a = p10_prime.x - this.xform.c
          this.b = p01_prime.x - this.xform.c
          this.d = p10_prime.y - this.xform.f
          this.e = p01_prime.y - this.xform.f
        }
      }
      if (this.currently_dragging === this.line_00_01) {
        const p00_A_x  = point.x - this.ring_00.shape.x
        const p00_A_y  = point.y - this.ring_00.shape.y
        const p00_01_x = this.ring_00.shape.x - this.ring_01.shape.x
        const p00_01_y = this.ring_00.shape.y - this.ring_01.shape.y
        const p00_10_x = this.ring_00.shape.x - this.ring_10.shape.x
        const p00_10_y = this.ring_00.shape.y - this.ring_10.shape.y
        const squared_length_p00_A  = p00_A_x  ** 2 + p00_A_y  ** 2
        const squared_length_p00_01 = p00_01_x ** 2 + p00_01_y ** 2
        const dot_product = p00_A_x * p00_01_x + p00_A_y * p00_01_y
        const cos_alpha = clamp(dot_product / Math.sqrt(squared_length_p00_A * squared_length_p00_01), -1, 1)
        const sign = Math.sign(p00_01_x * p00_A_y - p00_A_x * p00_01_y)
        const alpha = Math.acos(cos_alpha)
        const sin_alpha = Math.sin(alpha) * sign
        this.a = cos_alpha * p00_10_x - sin_alpha * p00_10_y
        this.b = cos_alpha * p00_01_x - sin_alpha * p00_01_y
        this.d = sin_alpha * p00_10_x + cos_alpha * p00_10_y
        this.e = sin_alpha * p00_01_x + cos_alpha * p00_01_y
      }
      if (this.currently_dragging === this.line_00_10) {
        const p00_A_x  = point.x - this.ring_00.shape.x
        const p00_A_y  = point.y - this.ring_00.shape.y
        const p00_01_x = this.ring_00.shape.x - this.ring_01.shape.x
        const p00_01_y = this.ring_00.shape.y - this.ring_01.shape.y
        const p00_10_x = this.ring_00.shape.x - this.ring_10.shape.x
        const p00_10_y = this.ring_00.shape.y - this.ring_10.shape.y
        const squared_length_p00_A  = p00_A_x  ** 2 + p00_A_y  ** 2
        const squared_length_p00_10 = p00_10_x ** 2 + p00_10_y ** 2
        const dot_product = p00_A_x * p00_10_x + p00_A_y * p00_10_y
        const cos_alpha = clamp(dot_product / Math.sqrt(squared_length_p00_A * squared_length_p00_10), -1, 1)
        const sign = Math.sign(p00_10_x * p00_A_y - p00_A_x * p00_10_y)
        const alpha = Math.acos(cos_alpha)
        const sin_alpha = Math.sin(alpha) * sign
        this.a = cos_alpha * p00_10_x - sin_alpha * p00_10_y
        this.b = cos_alpha * p00_01_x - sin_alpha * p00_01_y
        this.d = sin_alpha * p00_10_x + cos_alpha * p00_10_y
        this.e = sin_alpha * p00_01_x + cos_alpha * p00_01_y
      }
      flam3.clear()
      return true
    }

    set variation(value) {
      this.xform.variation = value
      this.editor.setAttribute('variation', value)
    }
    set a(value) {
      this.xform.a = value
      this.ring_10.shape.x = this.hole_10.shape.x = this.line_00_10.shape.to_x = this.line_10_01.shape.from_x = this.xform.a + this.xform.c
      this.editor.setAttribute('a', value.toFixed(2))
    }
    set b(value) {
      this.xform.b = value
      this.editor.setAttribute('b', value.toFixed(2))
      this.ring_01.shape.x = this.hole_01.shape.x = this.line_00_01.shape.to_x = this.line_10_01.shape.to_x   = this.xform.b + this.xform.c
    }
    set c(value) {
      this.xform.c = value
      this.editor.setAttribute('c', value.toFixed(2))
      const delta_x = value - this.ring_00.shape.x
      const lines = [this.line_00_01, this.line_00_10, this.line_10_01]
      for (const line of lines) {
        line.shape.from_x += delta_x
        line.shape.to_x   += delta_x
      }
      const circles = [this.ring_01, this.hole_01, this.ring_10, this.hole_10, this.ring_00, this.hole_00]
      for (const circle of circles) {
        circle.shape.x += delta_x
      }
    }
    set d(value) {
      this.xform.d = value
      this.editor.setAttribute('d', value.toFixed(2))
      this.ring_10.shape.y = this.hole_10.shape.y = this.line_00_10.shape.to_y = this.line_10_01.shape.from_y = this.xform.d + this.xform.f
    }
    set e(value) {
      this.xform.e = value
      this.editor.setAttribute('e', value.toFixed(2))
      this.ring_01.shape.y = this.hole_01.shape.y = this.line_00_01.shape.to_y = this.line_10_01.shape.to_y   = this.xform.e + this.xform.f
    }
    set f(value) {
      this.xform.f = value
      this.editor.setAttribute('f', value.toFixed(2))
      const delta_y = value - this.ring_00.shape.y
      const lines = [this.line_00_01, this.line_00_10, this.line_10_01]
      for (const line of lines) {
        line.shape.from_y += delta_y
        line.shape.to_y   += delta_y
      }
      const circles = [this.ring_01, this.hole_01, this.ring_10, this.hole_10, this.ring_00, this.hole_00]
      for (const circle of circles) {
        circle.shape.y += delta_y
      }
    }

    remove(fractal, primitives) {
      primitives.colors.remove(this.line_00_10.color, this.hole_01.color)
      primitives.lines.remove(this.line_00_10.shape, this.line_10_01.shape)
      primitives.circles.remove(this.ring_00.shape, this.hole_01.shape)
      primitives.remove(this.line_00_10, this.hole_01)
      fractal.remove(this.xform)
      this.editor.remove()
      gui.splice(gui.findIndex(v => v === this), 1)
      flam3.clear()
    }
  }

  const xform_list = document.getElementById('xforms')
  const gui = []
  for (let i = 0; i < fractal.length; i++)
    gui.push(new XFormEditor(fractal[i], primitives, xform_list, { r: 1.0, g: 0.4, b: 0.1, a: 1.0 }))
  gui.reverse()
  document.getElementById('add-xform').onclick = () => {
    const xform = fractal.add({ variation: 'linear', a: 1, b: 0, c: 0, d: 0, e: 1, f: 0 })
    gui.splice(0, 0, new XFormEditor(xform, primitives, xform_list, { r: 1.0, g: 0.4, b: 0.1, a: 1.0 }))
  }

  let running = starts_running
  function frame() {
    const commandBuffers = []
    let num_passes = 0
    function with_encoder(action) {
      const commandEncoder = device.createCommandEncoder()
      action(commandEncoder)
      num_passes++
      commandBuffers.push(commandEncoder.finish())
    }

    if (should_clear_histogram) {
      with_encoder(commandEncoder => {
        commandEncoder.copyBufferToBuffer(cleanHistogramBuffer, 0, histogramBuffer, 0, HISTOGRAM_BUFFER_SIZE)
      })
      should_clear_histogram = false
    }
    ++config.frame
    device.queue.writeBuffer(configBuffer,  0, config.buffer,  0)
    device.queue.writeBuffer(fractalBuffer, 0, fractal.buffer, 0)

    // Add some points to the histogram
    with_encoder(commandEncoder => {
      const passEncoder = commandEncoder.beginComputePass({
        label: 'FLAM3 > Pass > Add points'
      })
      passEncoder.setBindGroup(0, fractalBindGroup)
      passEncoder.setPipeline(addPointsPipeline)
      passEncoder.dispatch(30000)
      passEncoder.endPass()
    })

    // Find the max of the histogram
    with_encoder(commandEncoder => {
      const passEncoder = commandEncoder.beginComputePass({
        label: 'FLAM3 > Pass > Histogram Max'
      })
      passEncoder.setBindGroup(0, fractalBindGroup)
      passEncoder.setPipeline(histogramMaxPipeline)
      passEncoder.dispatch(1000)
      passEncoder.endPass()
    })

    // Render the histogram
    with_encoder(commandEncoder => {
      const passEncoder = commandEncoder.beginRenderPass({
        label: 'FLAM3 > Pass > Render',
        colorAttachments: [{
          view: context.getCurrentTexture().createView(),
          loadValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          storeOp: 'store'
        }]
      })
      passEncoder.setBindGroup(0, fractalBindGroup)
      passEncoder.setPipeline(renderPipeline)
      passEncoder.draw(4)
      passEncoder.endPass()
    })


    if (flam3.gui) {
      device.queue.writeBuffer(primitivesBuffer, 0, primitives.buffer,         0)
      device.queue.writeBuffer(colorsBuffer,     0, primitives.colors.buffer,  0)
      device.queue.writeBuffer(linesBuffer,      0, primitives.lines.buffer,   0)
      device.queue.writeBuffer(circlesBuffer,    0, primitives.circles.buffer, 0)

      // Render the GUI
      with_encoder(commandEncoder => {
        const passEncoder = commandEncoder.beginRenderPass({
          label: 'FLAM3 > Pass > GUI',
          colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadValue: 'load',
            storeOp: 'store'
          }]
        })
        passEncoder.setBindGroup(0, fractalBindGroup)
        passEncoder.setBindGroup(1, guiBindGroup)
        passEncoder.setPipeline(guiPipeline)
        passEncoder.draw(4)
        passEncoder.endPass()
      })
    }

    device.queue.submit(commandBuffers)
    if (running) requestAnimationFrame(frame)
  }

  if (running) requestAnimationFrame(frame)

  let should_clear_histogram = false
  const flam3 = {
    gui: true,
    fractal,
    config,
    get isRunning() { return running },
    set cmap(value) {
      cmap.copyFrom(cmaps[value])
      device.queue.writeBuffer(cmapBuffer, 0, cmap.buffer)
      flam3.clear()
    },
    stop()  { running = false },
    start() { running = true; frame() },
    step()  { frame() },
    clear() {
      // FIXME: Clear the canvas
      should_clear_histogram = true
    }
  }
  // Set default cmap
  flam3.cmap = 'gnuplot'

  // BEGIN UI
  canvas.onwheel = ev => {
    ev.preventDefault()
    flam3.config.zoom *= ev.deltaY < 0 ? 1.1 : 0.9
    flam3.clear()
  }
  const default_controls = {
    pointer_down() { return true },
    pointer_up() { return true },
    pointer_move(_point, ev) {
      const cursor_delta_x = -ev.movementX / canvas.width  * 2
      const cursor_delta_y = -ev.movementY / canvas.height * 2
      flam3.config.x += cursor_delta_x / config.zoom
      flam3.config.y += cursor_delta_y / config.zoom
      flam3.clear()
      return true
    }
  }
  gui.push(default_controls)
  function to_normalized_point(ev) {
    return {
      x: (ev.offsetX / canvas.width  * 2 - 1) / config.zoom + flam3.config.x,
      y: (ev.offsetY / canvas.height * 2 - 1) / config.zoom + flam3.config.y
    }
  }
  canvas.onpointerdown = ev => {
    const normalized_point = to_normalized_point(ev)
    if (!flam3.gui)
      default_controls.pointer_down(normalized_point, ev)
    else
      gui.find(gui_element => gui_element.pointer_down(normalized_point, ev))

    canvas.onpointermove = ev => {
      const normalized_point = to_normalized_point(ev)
      if (!flam3.gui)
        default_controls.pointer_move(normalized_point, ev)
      else
        gui.find(gui_element => gui_element.pointer_move(normalized_point, ev))
    }
    canvas.setPointerCapture(ev.pointerId)
  }
  canvas.onpointerup = ev => {
    const normalized_point = to_normalized_point(ev)
    if (!flam3.gui)
      default_controls.pointer_up(normalized_point, ev)
    else
      gui.find(gui_element => gui_element.pointer_up(normalized_point, ev))
    canvas.onpointermove = null
    canvas.releasePointerCapture(ev.pointerId)
  }

  return flam3
};

window.document.body.onload = async () => {
  window.flam3 = await init(document.getElementById('output'))

  const cmap_selection = document.getElementById('flam3-cmap')
  for (const cmap_name of Object.keys(cmaps)) {
    const option = document.createElement('option')
    option.textContent = cmap_name
    cmap_selection.appendChild(option)
  }
  cmap_selection.value = 'gnuplot'
  cmap_selection.onchange = ev => {
    window.flam3.cmap = ev.currentTarget.value
  }
}
