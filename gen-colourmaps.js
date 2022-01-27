const argv = process.argv
if (argv.length != 3) {
    console.error(`Usage: ${argv[1]} path/to/folder/with/colormaps`)
    process.exit(1)
}

const fs = require('fs')
const path = require('path')
const dir = argv[2]
const files = fs.readdirSync(dir)
const cmaps_entries = files.map(filename => [filename.slice(0, -5), read_cmap(path.join(dir, filename))]);

function read_cmap(filename) {
    const contents = fs.readFileSync(filename, 'utf-8')
    const lines = contents.split('\n')
    const int = Number.parseInt
    const colors = lines
      .filter(line => line[0] === '#')
      .map(line => line.length === 4
                   ? [int(line[1], 16) << 4, int(line[2], 16) << 4, int(line[3], 16) << 4, 0]
                   : [int(line.slice(1, 3), 16), int(line.slice(3, 5), 16), int(line.slice(5, 7), 16), 0])
    return colors
}

function encode(cmap) {
    const bytes = cmap.reduce((a, b) => a.concat(b), [])
    return Buffer.from(bytes).toString('base64')
}

console.log('// THIS FILE WAS AUTOGENERATED, SEE gen-colourmaps.js FOR MORE INFO')
console.log('const p = v => new Uint8Array(Array.prototype.map.call(atob(v), v => v.codePointAt(0)))')
console.log('export const cmaps = {')
for (const [cmap_name, cmap] of cmaps_entries) {
    console.log(`  '${cmap_name}': p('${encode(cmap)}'),`)
}
console.log('}')
console.log('export default cmaps')