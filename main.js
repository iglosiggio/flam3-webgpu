/// <reference types='@webgpu/types' />
/// 2021 - Ignacio E. Losiggio
/// Originally based on this example: https://austin-eng.com/webgpu-samples/samples/helloTriangle
/// Original idea: https://flam3.com/flame_draves.pdf
///
/// Stages:
///   1. Generate random points
///   2. Play the chaos game for some time
///   3. Gather the maximum value
///   4. Plot on the log-density display

const common_code = `
[[block]] struct Stage1Histogram {
  max: atomic<u32>;
  data: array<atomic<u32>>;
};

[[block]] struct Stage2Histogram {
  max: atomic<u32>;
  data: array<u32>;
};

[[block]] struct FragmentHistogram {
  max: u32;
  data: array<u32>;
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

struct Variation {
  fn_id: u32;
  transform: AffineTransform;
};

[[block]] struct Fractal {
  size: u32;
  variations: array<Variation>;
};

[[group(0), binding(0)]] var<storage, read_write> stage1_histogram: Stage1Histogram;
[[group(0), binding(0)]] var<storage, read_write> stage2_histogram: Stage2Histogram;
[[group(0), binding(0)]] var<storage, read> fragment_histogram: FragmentHistogram;
[[group(0), binding(1)]] var<storage, read> fractal: Fractal;
[[group(0), binding(2)]] var<uniform> config: CanvasConfiguration;

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

let LINEAR_FN = 0u;
fn linear(p: vec2<f32>) -> vec2<f32> {
  return p;
}

let SINUSOIDAL_FN = 1u;
fn sinusoidal(p: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(sin(p.x), sin(p.y));
}

let EYEFISH_FN = 27u;
fn eyefish(p: vec2<f32>) -> vec2<f32> {
  let v = 2.0 / (length(p) + 1.0);
  return v * p;
}

fn apply_fn(fn_id: u32, p: vec2<f32>) -> vec2<f32> {
  switch (fn_id) {
    case  0u: { return linear(p);     }
    case  1u: { return sinusoidal(p); }
    case 27u: { return eyefish(p); }
    default: {}
  }
  // Dumb and unreachable
  return vec2<f32>(0.0, 0.0);
}

fn apply_variation(variation: Variation, p: vec2<f32>) -> vec2<f32> {
  return apply_fn(variation.fn_id, apply_transform(p, variation.transform));
}

fn next(p: vec2<f32>) -> vec2<f32> {
  let i = random() % fractal.size;
  return apply_variation(fractal.variations[i], p);
}

fn plot(v: vec2<f32>) {
  let p = (v - config.origin) * config.zoom;
  if (-1. <= p.x && p.x < 1. && -1. <= p.y && p.y < 1.) {
    let ipoint = vec2<u32>(
      u32((p.x + 1.) / 2. * f32(config.dimensions.x)),
      u32((p.y + 1.) / 2. * f32(config.dimensions.y))
    );
    let offset = ipoint.y * config.dimensions.x + ipoint.x;
    atomicAdd(&stage1_histogram.data[offset], 1u);
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
    invocation_max = max(invocation_max, stage2_histogram.data[i]);
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
  var point = vec2<f32>(
    frandom() * 2. - 1.,
    frandom() * 2. - 1.,
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
  //return vec4<f32>(1.0, abs(sin(f32(fragment_histogram.data[i]) / 40000.0)), 0.0, 1.0);
  let result = f32(fragment_histogram.data[i]);
  let logresult = log(result)/log(f32(fragment_histogram.max));
  return vec4<f32>(logresult, logresult, logresult, 1.0);
}
`

const gui_fragment_wgsl = `
${common_code}

let LINE_KIND = 0u;
struct LinePrimitive {
  width: f32;
  from: vec2<f32>;
  to: vec2<f32>;
};

let CIRCLE_KIND = 1u;
struct CirclePrimitive {
  center: vec2<f32>;
  radius: f32;
};

// Stores the primitive kind
// The primitive is packed:
//  * The lower 16 bits denote the type
//  * The higher 16 bits denote the index
var<private> primitives: array<u32, 4> = array<u32, 4>(
  0x00000001u,
  0x00010001u,
  0x00000000u,
  0x00010000u,
);
var<private> colors: array<vec4<f32>, 4> = array<vec4<f32>, 4>(
  vec4<f32>(1.0, 1.0, 1.0, 1.0), // White
  vec4<f32>(1.0, 0.0, 0.0, 0.5), // Red (Transparent)
  vec4<f32>(0.0, 1.0, 0.0, 1.0), // Blue
  vec4<f32>(0.0, 1.0, 1.0, 1.0), // Yellow
);
var<private> circles: array<CirclePrimitive, 2> = array<CirclePrimitive, 2>(
  CirclePrimitive(vec2<f32>(0.0, 0.0), 0.4),
  CirclePrimitive(vec2<f32>(0.4, -0.2), 0.2),
);
var<private> lines: array<LinePrimitive, 2> = array<LinePrimitive, 2>(
  LinePrimitive(0.05, vec2<f32>(-1.0, -1.0), vec2<f32>(1.0,  1.0)),
  LinePrimitive(0.10, vec2<f32>(-0.5,  0.8), vec2<f32>(0.6, -0.4)),
);

[[stage(fragment)]]
fn fragment_main([[builtin(position)]] screen_pos: vec4<f32>) -> [[location(0)]] vec4<f32> {
  let dimensions = vec2<f32>(config.dimensions);
  let normal_pos = (screen_pos.xy / dimensions * -2.0 + vec2<f32>(1.0)) / config.zoom - config.origin;
  var result = vec4<f32>(0.0); // Black (Transparent)

  for (var i = 0u; i < 4u; i = i + 1u) {
    let primitive = primitives[i];
    let primitive_type = primitive & 0x0000FFFFu;
    let primitive_index = primitive >> 16u;
    let color = colors[i];
    switch (primitive_type) {
      // src: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
      case 0u /* LINE_KIND */: {
        // FIXME: Improve the names of the variables
        let line = lines[primitive_index];
        let line_length: f32 = length(line.to - line.from);
        let d: vec2<f32> = (line.to - line.from) / line_length;
        var q: vec2<f32> = normal_pos - (line.from + line.to) * 0.5;
        q = mat2x2<f32>(d.x, -d.y, d.y, d.x) * q;
        q = abs(q) - vec2<f32>(line_length, line.width) * 0.5;
        if (length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) < 0.0) {
          result = color;
        }
      }
      case 1u /* CIRCLE_KIND */: {
        let circle = circles[primitive_index];
        if (length(normal_pos - circle.center) < circle.radius) {
          result = color;
        }
      }
      default: {}
    }
  }
  return result;
}
`

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

  const devicePixelRatio = window.devicePixelRatio || 1
  const presentationSize = [
    canvas.clientWidth * devicePixelRatio,
    canvas.clientHeight * devicePixelRatio
  ]
  const format = context.getPreferredFormat(adapter)

  context.configure({
    device,
    format: format,
    size: presentationSize
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

  const bindGroupLayout = device.createBindGroupLayout({
      label: 'FLAM3 > Bind Group Layout',
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
        }
      ]
  })

  const layout = device.createPipelineLayout({
    label: 'FLAM3 > Pipeline Layout',
    bindGroupLayouts: [bindGroupLayout]
  })

  const HISTOGRAM_BUFFER_SIZE = 4 + 4 * 900 * 900
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
    size: 4 + 28 * 128,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  })
  const configBuffer = device.createBuffer({
    label: 'FLAM3 > Buffer > Configuration',
    size: 24,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  })
  //const timestampsBuffer = device.createBuffer({
  //  label: 'FLAM3 > Buffer > Timestamps',
  //  size: 8 * 2 * 4,
  //  usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_DST
  //})

  const bindGroup = device.createBindGroup({
    label: 'FLAM3 > Group Binding',
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          label: 'FLAM3 > Binding > Histogram',
          buffer: histogramBuffer
        }
      },
      {
        binding: 1,
        resource: {
          label: 'FLAM3 > Binding > Fractal',
          buffer: fractalBuffer
        }
      },
      {
        binding: 2,
        resource: {
          label: 'FLAM3 > Binding > Configuration',
          buffer: configBuffer
        }
      }
    ]
  })

  const addPointsPipeline = await device.createComputePipelineAsync({
      label: 'FLAM3 > Pipeline > Add points',
      layout,
      compute: {
          module: add_points_module,
          entryPoint: 'add_points'
      },
  })

  const histogramMaxPipeline = await device.createComputePipelineAsync({
      label: 'FLAM3 > Pipeline > Histogram Max',
      layout,
      compute: {
          module: histogram_max_module,
          entryPoint: 'histogram_max'
      },
  })

  const renderPipeline = await device.createRenderPipelineAsync({
    label: 'FLAM3 > Pipeline > Render',
    layout,
    vertex: {
      layout,
      module: vertex_module,
      entryPoint: 'vertex_main',
    },
    fragment: {
      layout,
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
    layout,
    vertex: {
      layout,
      module: vertex_module,
      entryPoint: 'vertex_main',
    },
    fragment: {
      layout,
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

  //const timestampQuerySet = device.createQuerySet({
  //  label: 'FLAM3 > QuerySet > Timings',
  //  type: 'timestamp',
  //  count: 16,
  //})

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
  const config = window.config = new Config
  config.width = canvas.width
  config.height = canvas.height
  config.zoom = 1

  const VARIATION_SIZE = 28
  const fn_id_to_str_entries = [
    [ 0, 'linear'],
    [ 1, 'sinusoidal'],
    [27, 'eyefish']
  ];
  const fn_id_to_str = new Map(fn_id_to_str_entries)
  const str_to_fn_id = new Map(fn_id_to_str_entries.map(([a, b]) => [b, a]))
  class Variation {
    constructor(buffer, offset, element) {
      this._fn_id = new Uint32Array(buffer, offset, 1)
      this._affine_transform = new Float32Array(buffer, offset + 4, 6)

      if (element)
        [
          this._fn_id_element,
          this._a_element,
          this._b_element,
          this._c_element,
          this._d_element,
          this._e_element,
          this._f_element
        ] = element.querySelectorAll('[slot]')
    }

    get fn_id() {
      const id = this._fn_id[0]
      const result = fn_id_to_str.get(id)
      if (result === undefined) throw new Error(`Unknown fn_id '${id}'`)
      return result
    }

    set fn_id(value) {
      const id = str_to_fn_id.get(value)
      if (id === undefined) throw new Error(`Unknown fn_id string '${value}'`)
      if (this._fn_id_element) this._fn_id_element.textContent = value
      this._fn_id[0] = id
    }

    get a()      { return this._affine_transform[0] }
    get b()      { return this._affine_transform[1] }
    get c()      { return this._affine_transform[2] }
    get d()      { return this._affine_transform[3] }
    get e()      { return this._affine_transform[4] }
    get f()      { return this._affine_transform[5] }
    set a(value) {
      this._affine_transform[0] = value
      if (this._a_element) this._a_element.textContent = value
    }
    set b(value) {
      this._affine_transform[1] = value
      if (this._b_element) this._b_element.textContent = value
    }
    set c(value) {
      this._affine_transform[2] = value
      if (this._c_element) this._c_element.textContent = value
    }
    set d(value) {
      this._affine_transform[3] = value
      if (this._d_element) this._d_element.textContent = value
    }
    set e(value) {
      this._affine_transform[4] = value
      if (this._e_element) this._e_element.textContent = value
    }
    set f(value) {
      this._affine_transform[5] = value
      if (this._f_element) this._f_element.textContent = value
    }
  }

  const MAX_VARIATIONS = 128
  class Fractal {
    buffer = new ArrayBuffer(4 + VARIATION_SIZE * MAX_VARIATIONS)

    _length = new Uint32Array(this.buffer, 0, 1)
    get length()      { return this._length[0] }
    set length(value) { return this._length[0] = value }

    at(idx) {
      if (idx < 0 || this.length <= idx)
        throw new Error('Index out of bounds!')
      return this[idx]
    }

    add(fn_id, a, b, c, d, e, f) {
      if (this.length === MAX_VARIATIONS)
        throw new Error('Variations limit exceeded!')
      const editor = add_variation_editor()
      const variation = this[this.length] = new Variation(this.buffer, 4 + VARIATION_SIZE * this.length, editor)
      this.length++
      if (fn_id !== undefined) variation.fn_id = fn_id
      if (a     !== undefined) variation.a     = a
      if (b     !== undefined) variation.b     = b
      if (c     !== undefined) variation.c     = c
      if (d     !== undefined) variation.d     = d
      if (e     !== undefined) variation.e     = e
      if (f     !== undefined) variation.f     = f
      return variation
    }
  }
  const fractal = new Fractal
  //fractal.add('sinusoidal', 0.5,  0.0,  0.5,  0.0, 0.5,  0.5)
  //fractal.add('sinusoidal', 0.5,  0.0, -0.5,  0.0, 0.5,  0.5)
  //fractal.add('sinusoidal', 0.5,  0.0,  0.0,  0.0, 0.5, -0.5)
  //fractal.add('linear',     0.0,  0.8,  0.0,  0.6, 0.0,  0.0)
  //fractal.add('linear',     0.0, -0.8,  0.0, -0.6, 0.0,  0.0)
  fractal.add('eyefish',  0.321636, -0.204179, -0.633718,  0.204179,   0.321637,  1.140693)
  fractal.add('eyefish',  0.715673, -0.418864, 0.576108,  0.418864,  0.715673,  0.455125)
  fractal.add('eyefish', -0.212317,  0.536045, 0.53578,  -0.536045, -0.212317, -0.743179)
  fractal.add('linear', 0.7,  0.0, 0.0,  0.0, 0.7, 0.0)

  let running = starts_running
  function frame() {
    const commandBuffers = []
    let num_passes = 0
    function with_encoder(action) {
      const commandEncoder = device.createCommandEncoder()
      //commandEncoder.writeTimestamp(timestampQuerySet, 2 * num_passes)
      action(commandEncoder)
      //commandEncoder.writeTimestamp(timestampQuerySet, 2 * num_passes + 1)
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
    device.queue.writeBuffer(configBuffer, 0, config.buffer, 0)
    device.queue.writeBuffer(fractalBuffer, 0, fractal.buffer, 0)

    // Add some points to the histogram
    with_encoder(commandEncoder => {
      const passEncoder = commandEncoder.beginComputePass({
        label: 'FLAM3 > Pass > Add points'
      })
      passEncoder.setBindGroup(0, bindGroup)
      passEncoder.setPipeline(addPointsPipeline)
      passEncoder.dispatch(20000)
      passEncoder.endPass()
    })

    // Find the max of the histogram
    with_encoder(commandEncoder => {
      const passEncoder = commandEncoder.beginComputePass({
        label: 'FLAM3 > Pass > Histogram Max'
      })
      passEncoder.setBindGroup(0, bindGroup)
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
      passEncoder.setBindGroup(0, bindGroup)
      passEncoder.setPipeline(renderPipeline)
      passEncoder.draw(4)
      passEncoder.endPass()
    })

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
      passEncoder.setBindGroup(0, bindGroup)
      passEncoder.setPipeline(guiPipeline)
      passEncoder.draw(4)
      passEncoder.endPass()
    })

    // Resolve the timestamps
    //{
    //  const commandEncoder = device.createCommandEncoder()
    //  commandEncoder.resolveQuerySet(timestampQuerySet, 0, 2 * num_passes, timestampsBuffer, 0)
    //  commandBuffers.push(commandEncoder.finish())
    //}

    device.queue.submit(commandBuffers)
    if (running) requestAnimationFrame(frame)
  }

  if (running) requestAnimationFrame(frame)

  let should_clear_histogram = false
  const flam3 = {
    fractal,
    config,
    get isRunning() { return running },
    stop()  { running = false },
    start() { running = true; frame() },
    step()  { frame() },
    clear() {
      // FIXME: Clear the canvas
      should_clear_histogram = true
    }
  }

  // BEGIN UI
  canvas.onwheel = ev => {
    ev.preventDefault()
    flam3.config.zoom *= ev.deltaY < 0 ? 1.1 : 0.9
    flam3.clear()
  }
  canvas.onpointerdown = ev => {
    canvas.onpointermove = ev => {
      const cursor_delta_x = -ev.movementX / canvas.width
      const cursor_delta_y = -ev.movementY / canvas.height
      flam3.config.x += cursor_delta_x / config.zoom
      flam3.config.y += cursor_delta_y / config.zoom
      flam3.clear()
    }
    canvas.setPointerCapture(ev.pointerId)
  }
  canvas.onpointerup = ev => {
    canvas.onpointermove = null
    canvas.releasePointerCapture(ev.pointerId)
  }

  return flam3
};

window.document.body.onload = async () => {
  window.flam3 = await init(document.getElementById('output'))
}
