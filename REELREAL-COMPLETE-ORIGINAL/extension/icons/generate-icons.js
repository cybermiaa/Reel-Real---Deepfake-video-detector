/* ============================================================================
   ICON GENERATOR — build-time helper, NOT part of the shipped extension
   ----------------------------------------------------------------------------
   Chrome only accepts raster images for extension icons (no SVG), so this
   writes the three required PNGs from scratch using Node's built-in zlib.
   No dependencies, no design tool.

       node extension/icons/generate-icons.js

   Design: forest-green rounded square, orange diagonal slash — the "/" from the
   REEL/REAL wordmark. Swap in your own PNGs any time; only the filenames and
   the sizes referenced in manifest.json matter.
   ========================================================================== */

const zlib = require('zlib');
const fs = require('fs');
const path = require('path');

const FOREST = [0x14, 0x4D, 0x37];
const ORANGE = [0xEF, 0x63, 0x37];

function crc32(buf) {
  let c, table = [];
  for (let n = 0; n < 256; n++) {
    c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  let crc = 0xFFFFFFFF;
  for (const b of buf) crc = table[(crc ^ b) & 0xFF] ^ (crc >>> 8);
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, crc]);
}

function makePNG(size) {
  const radius = size * 0.22;
  // Slash geometry, as a fraction of the icon box.
  const slashHalf = size * 0.075;
  const rows = [];

  for (let y = 0; y < size; y++) {
    // Each PNG scanline is prefixed with a filter-type byte; 0 = none.
    const row = [0];
    for (let x = 0; x < size; x++) {
      // Rounded-corner test: only the corner quadrants get a radius check.
      const cx = x < radius ? radius : (x > size - radius ? size - radius : x);
      const cy = y < radius ? radius : (y > size - radius ? size - radius : y);
      const inside = Math.hypot(x - cx, y - cy) <= radius;

      if (!inside) { row.push(0, 0, 0, 0); continue; }

      // The slash runs bottom-left to top-right: x + y ≈ size.
      const onSlash = Math.abs((x + y) - size) < slashHalf * 2 &&
                      y > size * 0.18 && y < size * 0.82;

      const [r, g, b] = onSlash ? ORANGE : FOREST;
      row.push(r, g, b, 255);
    }
    rows.push(Buffer.from(row));
  }

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8;   // bit depth
  ihdr[9] = 6;   // colour type 6 = RGBA
  ihdr[10] = 0;  // deflate
  ihdr[11] = 0;  // adaptive filtering
  ihdr[12] = 0;  // no interlace

  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]), // PNG magic
    chunk('IHDR', ihdr),
    chunk('IDAT', zlib.deflateSync(Buffer.concat(rows))),
    chunk('IEND', Buffer.alloc(0))
  ]);
}

for (const size of [16, 48, 128]) {
  const file = path.join(__dirname, `icon${size}.png`);
  fs.writeFileSync(file, makePNG(size));
  console.log('wrote', file);
}
