/// <reference types="@webgpu/types" />
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

fn apply_fn(fn_id: u32, p: vec2<f32>) -> vec2<f32> {
  switch (fn_id) {
    case 0u: { return linear(p);     }
    case 1u: { return sinusoidal(p); }
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

fn plot(p: vec2<f32>) {
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

const render_wgsl = `
${common_code}
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

const init = async canvas => {
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
  const render_module = device.createShaderModule({
      label: 'FLAM3 > Module > Render',
      code: render_wgsl
  })

  const bindGroupLayout = device.createBindGroupLayout({
      label: 'FLAM3 > Bind Group Layout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
          buffer: { type: "storage" }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" }
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" }
        }
      ]
  })

  const layout = device.createPipelineLayout({
    label: 'FLAM3 > Pipeline Layout',
    bindGroupLayouts: [bindGroupLayout]
  })

  const histogramBuffer = device.createBuffer({
    label: 'FLAM3 > Buffer > Histogram',
    size: 4 + 4 * 900 * 900,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
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
      module: render_module,
      entryPoint: 'vertex_main',
    },
    fragment: {
      layout,
      module: render_module,
      entryPoint: 'fragment_main',
      targets: [{ format }]
    },
    primitive: {
      topology: 'triangle-strip',
      stripIndexFormat: 'uint32'
    }
  })

  class Config {
    buffer = new ArrayBuffer(24)

    _origin = new Float32Array(this.buffer, 0, 2)
    get x()      { return this._origin[0] }
    set x(value) { return this._origin[0] = value }
    get y()      { return this._origin[1] }
    set y(value) { return this._origin[1] = value }

    _dimensions = new Uint32Array(this.buffer, 8, 2)
    get width()       { return this._dimensions[0] }
    set width(value)  { return this._dimensions[0] = value }
    get height()      { return this._dimensions[1] }
    set height(value) { return this._dimensions[1] = value }

    _frame = new Uint32Array(this.buffer, 16, 1)
    get frame()      { return this._frame[0] }
    set frame(value) { return this._frame[0] = value }
  }
  const config = window.config = new Config
  config.width = canvas.width
  config.height = canvas.height

  const VARIATION_SIZE = 28
  const fn_id_to_str = new Map([
    [0, 'linear'],
    [1, 'sinusoidal']
  ])
  const str_to_fn_id = new Map([
    ['linear', 0],
    ['sinusoidal', 1]
  ])
  class Variation {
    constructor(buffer, offset) {
      this._fn_id = new Uint32Array(buffer, offset, 1)
      this._affine_transform = new Float32Array(buffer, offset + 4, 6)
    }

    get fn_id() {
      const id = this._fn_id[0]
      const result = fn_id_to_str.get(id)
      if (result === undefined)
        throw new Error(`Unknown fn_id "${id}"`)
    }

    set fn_id(value) {
      const id = str_to_fn_id.get(value)
      if (id === undefined)
        throw new Error(`Unknown fn_id string "${value}"`)
      this._fn_id[0] = id
    }

    get a()      { return this._affine_transform[0] }
    set a(value) { return this._affine_transform[0] = value }
    get b()      { return this._affine_transform[1] }
    set b(value) { return this._affine_transform[1] = value }
    get c()      { return this._affine_transform[2] }
    set c(value) { return this._affine_transform[2] = value }
    get d()      { return this._affine_transform[3] }
    set d(value) { return this._affine_transform[3] = value }
    get e()      { return this._affine_transform[4] }
    set e(value) { return this._affine_transform[4] = value }
    get f()      { return this._affine_transform[5] }
    set f(value) { return this._affine_transform[5] = value }
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
      return new Variation(this.buffer, 4 + VARIATION_SIZE * idx)
    }

    add(fn_id, a, b, c, d, e, f) {
      if (this.length === MAX_VARIATIONS)
        throw new Error('Variations limit exceeded!')
      const variation = new Variation(this.buffer, 4 + VARIATION_SIZE * this.length++)
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
  fractal.add('sinusoidal', 0.5,  0.0,  0.5,  0.0, 0.5,  0.5)
  fractal.add('sinusoidal', 0.5,  0.0, -0.5,  0.0, 0.5,  0.5)
  fractal.add('sinusoidal', 0.5,  0.0,  0.0,  0.0, 0.5, -0.5)
  fractal.add('linear',     0.0,  0.8,  0.0,  0.6, 0.0,  0.0)
  fractal.add('linear',     0.0, -0.8,  0.0, -0.6, 0.0,  0.0)

  function frame() {
    // Copy current configuration
    ++config.frame
    device.queue.writeBuffer(configBuffer, 0, config.buffer, 0)
    device.queue.writeBuffer(fractalBuffer, 0, fractal.buffer, 0)

    const commandBuffers = []

    // Add some points to the histogram
    {
      const commandEncoder = device.createCommandEncoder()
      const passEncoder = commandEncoder.beginComputePass({
        label: 'FLAM3 > Pass > Add points'
      })
      passEncoder.setBindGroup(0, bindGroup)
      passEncoder.setPipeline(addPointsPipeline)
      passEncoder.dispatch(10000)
      passEncoder.endPass()
      commandBuffers.push(commandEncoder.finish())
    }

    // Find the max of the histogram
    {
      const commandEncoder = device.createCommandEncoder()
      const passEncoder = commandEncoder.beginComputePass({
        label: 'FLAM3 > Pass > Histogram Max'
      })
      passEncoder.setBindGroup(0, bindGroup)
      passEncoder.setPipeline(histogramMaxPipeline)
      passEncoder.dispatch(1000)
      passEncoder.endPass()
      commandBuffers.push(commandEncoder.finish())
    }

    // Render the histogram
    {
      const commandEncoder = device.createCommandEncoder()
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
      commandBuffers.push(commandEncoder.finish())
    }

    device.queue.submit(commandBuffers)
    requestAnimationFrame(frame)
  }

  requestAnimationFrame(frame)
};

window.document.body.onload = () => init(document.getElementById('output'))
