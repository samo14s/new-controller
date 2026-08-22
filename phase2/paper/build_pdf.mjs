/**
 * Turn paper_ar.md into a print-ready HTML page: RTL Arabic, LaTeX rendered to
 * SVG, fonts embedded, page rules for A4.
 *
 * The math is converted at build time to SVG paths, so the finished page needs
 * no scripts, no network and no math fonts -- which is what makes it print
 * identically anywhere.
 *
 *   npm install mathjax-full @fontsource/amiri @fontsource/ibm-plex-sans-arabic \
 *               @fontsource/ibm-plex-mono
 *   node phase2/paper/build_pdf.mjs [out.html]
 *
 * Then paginate it (Chromium, through Playwright):
 *   python phase2/paper/print_pdf.py out.html paper_ar.pdf
 */
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';

const HERE = path.dirname(new URL(import.meta.url).pathname);
const SRC = path.join(HERE, 'paper_ar.md');
const OUT = process.argv[2] || path.join(HERE, 'paper_ar.html');

// The four npm packages are a build dependency, not a repository one, so they
// are resolved from wherever they were installed rather than vendored here.
const MODULES = process.env.PAPER_NODE_MODULES
                || path.join(HERE, 'node_modules');
if (!fs.existsSync(MODULES)) {
  console.error(`no node_modules at ${MODULES}.\n`
    + 'Install the build dependencies somewhere and point at them:\n'
    + '  npm install mathjax-full @fontsource/amiri '
    + '@fontsource/ibm-plex-sans-arabic @fontsource/ibm-plex-mono\n'
    + '  PAPER_NODE_MODULES=<dir>/node_modules node '
    + 'phase2/paper/build_pdf.mjs out.html');
  process.exit(2);
}
const require = createRequire(path.join(MODULES, 'anchor.cjs'));
const load = spec => import(pathToFileURL(require.resolve(spec)).href);

const {mathjax} = await load('mathjax-full/js/mathjax.js');
const {TeX} = await load('mathjax-full/js/input/tex.js');
const {SVG} = await load('mathjax-full/js/output/svg.js');
const {liteAdaptor} = await load('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = await load('mathjax-full/js/handlers/html.js');
const {AllPackages} = await load('mathjax-full/js/input/tex/AllPackages.js');

// ---------------------------------------------------------------- fonts
// Embedded as data URIs: the page has to render the same with no network.
function fontFace(family, pkg, file, weight, unicodeRange) {
  const p = require.resolve(`${pkg}/files/${file}`);
  const b64 = fs.readFileSync(p).toString('base64');
  return `@font-face{font-family:"${family}";font-style:normal;` +
         `font-weight:${weight};font-display:block;` +
         `src:url(data:font/woff2;base64,${b64}) format("woff2");` +
         (unicodeRange ? `unicode-range:${unicodeRange};` : '') + '}';
}

const AR = 'U+0600-06FF,U+0750-077F,U+08A0-08FF,U+FB50-FDFF,U+FE70-FEFF,' +
           'U+200C-200F,U+2010-2011,U+204F,U+2E41,U+FBB2-FBC1';
const FONTS = [
  fontFace('Plex Arabic', '@fontsource/ibm-plex-sans-arabic',
           'ibm-plex-sans-arabic-arabic-400-normal.woff2', 400, AR),
  fontFace('Plex Arabic', '@fontsource/ibm-plex-sans-arabic',
           'ibm-plex-sans-arabic-arabic-600-normal.woff2', 600, AR),
  fontFace('Plex Arabic', '@fontsource/ibm-plex-sans-arabic',
           'ibm-plex-sans-arabic-arabic-700-normal.woff2', 700, AR),
  fontFace('Plex Arabic', '@fontsource/ibm-plex-sans-arabic',
           'ibm-plex-sans-arabic-latin-400-normal.woff2', 400),
  fontFace('Plex Arabic', '@fontsource/ibm-plex-sans-arabic',
           'ibm-plex-sans-arabic-latin-600-normal.woff2', 600),
  fontFace('Plex Arabic', '@fontsource/ibm-plex-sans-arabic',
           'ibm-plex-sans-arabic-latin-700-normal.woff2', 700),
  fontFace('Amiri', '@fontsource/amiri', 'amiri-arabic-400-normal.woff2', 400,
           AR),
  fontFace('Amiri', '@fontsource/amiri', 'amiri-arabic-700-normal.woff2', 700,
           AR),
  fontFace('Amiri', '@fontsource/amiri', 'amiri-latin-400-normal.woff2', 400),
  fontFace('Amiri', '@fontsource/amiri', 'amiri-latin-700-normal.woff2', 700),
  fontFace('Plex Mono', '@fontsource/ibm-plex-mono',
           'ibm-plex-mono-latin-400-normal.woff2', 400),
  fontFace('Plex Mono', '@fontsource/ibm-plex-mono',
           'ibm-plex-mono-latin-600-normal.woff2', 600),
].join('\n');

// ---------------------------------------------------------------- math
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const texIn = new TeX({packages: AllPackages});
const svgOut = new SVG({fontCache: 'local'});
const mjDoc = mathjax.document('', {InputJax: texIn, OutputJax: svgOut});

function tex(src, display) {
  const node = mjDoc.convert(src.trim(), {display, em: 11, ex: 5});
  return adaptor.outerHTML(node);
}

// ---------------------------------------------------------------- markdown
const CROSS = '❌';
const esc = t => t.replace(/&/g, '&amp;').replace(/</g, '&lt;')
                  .replace(/>/g, '&gt;');

/** Inline: math first, so its backslashes and underscores survive the
 *  emphasis pass; code next; then the ordinary markdown. */
function inline(t) {
  // Sentinels from the private-use area: a placeholder must never collide
  // with a real number in the text, and this document is mostly numbers.
  const OPEN = '\uE000', CLOSE = '\uE001';
  const slots = [];
  const hold = html => {
    slots.push(html);
    return OPEN + (slots.length - 1) + CLOSE;
  };
  // A quantity and its unit are one LTR island: bound together they cannot be
  // split across a line, nor reordered around each other by the bidi algorithm.
  // The trailing group catches an exponent written as its own math, s$^{-1}$.
  const UNIT = String.raw`(?:V\/N|rad\/s|V\u00B2s|mm\/[^\s\u060C,.)]+|[A-Za-z]{1,3})`;
  t = t.replace(new RegExp(`\\$([^$\\n]+)\\$ (${UNIT})(?![A-Za-z])(\\$[^$\\n]+\\$)?`, 'g'),
    (_, m, unit, exp) => hold('<span class="mi" dir="ltr">' + tex(m, false)
      + '&#160;' + esc(unit)
      + (exp ? tex(exp.slice(1, -1), false) : '') + '</span>'));
  // An acronym that labels the quantity right after it belongs to the same
  // island as well -- "PSO 20x20" is one thing, not two.  After the unit pass,
  // so that s$^{-1}$ is already spoken for.
  // The lookbehind keeps the token from being part of a control sequence
  // (\eta, \infty) or the tail of an earlier expression, and the lookahead
  // keeps the body from swallowing Arabic between two separate expressions.
  t = t.replace(/(?<![\w\\$])([A-Za-z][A-Za-z0-9]{1,7})[ ]?\u200F?\$(?![^$\n]*[\u0600-\u06FF])([^$\n]+)\$/g,
    (_, tok, m) => hold('<span class="mi" dir="ltr">' + esc(tok) + '&#160;'
      + tex(m, false) + '</span>'));
  t = t.replace(/\$(?![^$\n]*[\u0600-\u06FF])([^$\n]+)\$([\u2010-\u2011-][A-Za-z][A-Za-z0-9-]*)/g,
    (_, m, tail) => hold('<span class="mi" dir="ltr">' + tex(m, false)
      + esc(tail) + '</span>'));
  t = t.replace(/\$([^$\n]+)\$/g,
                (_, m) => hold(`<span class="mi" dir="ltr">${tex(m, false)}</span>`));
  t = t.replace(/`([^`]+)`/g, (_, c) => hold(`<code dir="ltr">${esc(c)}</code>`));
  // The same binding for quantities written as plain text: "40 kHz" is one
  // reading unit, and must not be reordered or split by the bidi algorithm.
  t = t.replace(/(?<![\w.])(\d[\d.,]*)[ ](kHz|MHz|Hz|rpm|mm|ms|V\/N|rad\/s|V|s)(?![\w])/g,
    (_, n, u) => hold(`<span class="mi" dir="ltr">${n}&#160;${u}</span>`));
  t = esc(t);
  t = t.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  t = t.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, '<em>$1</em>');
  t = t.split(CROSS).join('<span class="x">&#215;</span>');
  // Punctuation touching an island travels with it, so a full stop after a
  // quantity cannot be left alone at the top of the next line.
  t = t.replace(/([([\u00AB]?)(\uE000\d+\uE001)([.,\u060C\u061B:)\]\u00BB%]*)/g,
    (_, before, slot, after) => (before || after)
      ? `<span class="nb">${before}${slot}${after}</span>`
      : slot);
  return t.replace(/\uE000(\d+)\uE001/g, (_, i) => slots[+i]);
}

/** Base direction of a cell: its first strong character decides, which is the
 *  Unicode rule (P2/P3) written out so the result does not depend on how a
 *  particular renderer treats `unicode-bidi: plaintext` inside a table. */
function baseDir(text) {
  const bare = text.replace(/\$[^$\n]*\$/g, '').replace(/`[^`]*`/g, '');
  for (const ch of bare) {
    const c = ch.codePointAt(0);
    if ((c >= 0x0600 && c <= 0x08FF) || (c >= 0xFB50 && c <= 0xFEFF)) {
      return 'rtl';
    }
    if (c === 0x200F) return 'rtl';           // the author's own RLM marks
    if (c === 0x200E) return 'ltr';
    if ((c >= 0x41 && c <= 0x5A) || (c >= 0x61 && c <= 0x7A)) return 'ltr';
  }
  // Nothing strong at all -- a cell of pure numbers, or a range like
  // 0.0393-0.0451.  Those read left to right, not right to left.
  return 'ltr';
}

/** A display block whose \text{} runs hold Arabic.  MathJax has no Arabic
 *  glyphs and does no shaping, so it would set those runs letter by letter and
 *  backwards.  The block is laid out as a row instead: the symbols still go
 *  through MathJax, the Arabic is ordinary text in the page's own font. */
function mixedDisplay(body) {
  const items = body.split(/,?\s*\\quad\s*/).map(x => x.trim()).filter(Boolean);
  const cells = items.map(item => {
    const parts = [];
    const re = /\\text\{([^}]*)\}/g;
    let last = 0;
    let m;
    while ((m = re.exec(item)) !== null) {
      const before = item.slice(last, m.index).trim().replace(/\\$/, '').trim();
      if (before) {
        parts.push(`<span class="mi" dir="ltr">${tex(before, false)}</span>`);
      }
      parts.push(`<span class="mtxt">${esc(m[1])}</span>`);
      last = m.index + m[0].length;
    }
    const tail = item.slice(last).trim().replace(/\\$/, '').trim();
    if (tail) parts.push(`<span class="mi" dir="ltr">${tex(tail, false)}</span>`);
    return `<span class="mitem">${parts.join('')}</span>`;
  });
  return `<div class="mb mixed">${cells.join('')}</div>`;
}

function tableRow(line) {
  return line.replace(/^\s*\|/, '').replace(/\|\s*$/, '').split('|')
             .map(c => c.trim());
}

function convert(md) {
  const lines = md.split('\n');
  const out = [];
  const para = [];
  let i = 0;
  const flushPara = () => {
    if (para.length) {
      const raw = para.join(' ');
      out.push(`<p dir="${baseDir(raw)}">${inline(raw)}</p>`);
    }
    para.length = 0;
  };

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) { flushPara(); i++; continue; }

    if (line.trim() === '---') { flushPara(); out.push('<hr>'); i++; continue; }

    const h = /^(#{1,4})\s+(.*)$/.exec(line);
    if (h) {
      flushPara();
      out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`);
      i++; continue;
    }

    if (line.startsWith('$$')) {                       // display math
      flushPara();
      const body = [];
      const first = line.slice(2).trim();
      if (first) body.push(first);
      i++;
      while (i < lines.length && !lines[i].trim().startsWith('$$')) {
        body.push(lines[i]); i++;
      }
      i++;
      const tex_src = body.join('\n');
      out.push(/\\text\{[^}]*[\u0600-\u06FF]/.test(tex_src)
        ? mixedDisplay(tex_src)
        : `<div class="mb" dir="ltr">${tex(tex_src, true)}</div>`);
      continue;
    }

    if (line.startsWith('```')) {                      // code block
      flushPara();
      i++;
      const body = [];
      while (i < lines.length && !lines[i].startsWith('```')) {
        body.push(lines[i]); i++;
      }
      i++;
      out.push(`<pre dir="ltr"><code>${esc(body.join('\n'))}</code></pre>`);
      continue;
    }

    if (line.trim().startsWith('|')) {                 // table
      flushPara();
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        rows.push(lines[i]); i++;
      }
      const head = tableRow(rows[0]);
      const body = rows.slice(2).map(tableRow);
      const th = head.map(c => `<th dir="${baseDir(c)}">${inline(c)}</th>`)
                     .join('');
      const tb = body.map(r =>
        '<tr>' + r.map((c, k) =>
          `<td dir="${baseDir(c)}"${k === 0 ? ' class="lead"' : ''}>`
          + `${inline(c)}</td>`).join('')
        + '</tr>').join('\n');
      out.push(`<div class="tw"><table><thead><tr>${th}</tr></thead>` +
               `<tbody>${tb}</tbody></table></div>`);
      continue;
    }

    if (/^(\s*)([-*]|\d+\.)\s+(.*)$/.test(line)) {     // list
      flushPara();
      const ordered = /^\s*\d/.test(line);
      const items = [];
      while (i < lines.length) {
        const m = /^(\s*)([-*]|\d+\.)\s+(.*)$/.exec(lines[i]);
        if (m) { items.push([m[3]]); i++; continue; }
        // an indented line continues the item above it
        if (items.length && /^\s{2,}\S/.test(lines[i])) {
          items[items.length - 1].push(lines[i].trim()); i++; continue;
        }
        break;
      }
      const tag = ordered ? 'ol' : 'ul';
      out.push(`<${tag}>`
        + items.map(p => `<li>${inline(p.join(' '))}</li>`).join('')
        + `</${tag}>`);
      continue;
    }

    para.push(line.trim());
    i++;
  }
  flushPara();
  return out
    .filter((el, k) => !(el === '<hr>' && (out[k + 1] || '').startsWith('<h2')))
    .join('\n');
}

// ---------------------------------------------------------------- page
const CSS = `
${FONTS}

:root{
  --ink:#101619; --ink-2:#33454C; --muted:#5B6A70;
  --rule:#C9D2D5; --rule-soft:#E3E9EA; --tint:#F4F7F8;
  --accent:#0E7490; --bad:#9F1239;
}
@page{ size:A4; margin:20mm 18mm 18mm; }
*,*::before,*::after{box-sizing:border-box}
html{-webkit-print-color-adjust:exact; print-color-adjust:exact}
body{
  margin:0; direction:rtl; background:#fff; color:var(--ink);
  font-family:"Plex Arabic","DejaVu Sans",sans-serif;
  font-size:10.2pt; line-height:1.72; font-weight:400;
  text-rendering:optimizeLegibility;
}
p{margin:0 0 .62em; text-align:justify}
strong{font-weight:700}
em{font-style:normal; font-weight:600; color:var(--ink-2)}

h1{
  font-family:Amiri,"Plex Arabic",serif; font-weight:700;
  font-size:20pt; line-height:1.34; margin:0 0 10pt;
  text-wrap:balance; padding-bottom:9pt; border-bottom:1pt solid var(--ink);
}
h2{
  font-family:Amiri,"Plex Arabic",serif; font-weight:700;
  font-size:14.5pt; line-height:1.3; margin:19pt 0 6pt;
  padding-bottom:3pt; border-bottom:.6pt solid var(--rule);
  break-after:avoid; break-inside:avoid;
}
h3{
  font-family:"Plex Arabic",sans-serif; font-weight:600;
  font-size:11pt; margin:13pt 0 4pt; break-after:avoid;
}
h1+p, h2+p, h3+p, h2+.tw, h3+.tw{margin-top:0}
hr{border:0; border-top:.6pt solid var(--rule-soft); margin:13pt auto; width:32%}

code{
  font-family:"Plex Mono",ui-monospace,monospace; font-size:.86em;
  background:var(--tint); border:.4pt solid var(--rule-soft);
  border-radius:2pt; padding:0 3pt; unicode-bidi:isolate;
  font-variant-numeric:tabular-nums;
}
pre{
  font-family:"Plex Mono",monospace; font-size:8.4pt; line-height:1.62;
  background:var(--tint); border:.5pt solid var(--rule-soft);
  border-radius:3pt; padding:8pt 10pt; margin:8pt 0 10pt;
  white-space:pre-wrap; overflow-wrap:break-word; break-inside:avoid;
}
/* Plex Mono carries no Arabic, so the comments in a code block fall through
   to the proportional Arabic face rather than to a monospaced one, which
   renders Arabic letter by letter. */
pre code{background:none; border:0; padding:0; font-size:inherit}
pre, pre code, code{font-family:"Plex Mono","Plex Arabic",ui-monospace,monospace}

ul,ol{margin:0 0 .7em; padding-inline-start:1.35em}
li{margin:0 0 .32em; text-align:justify}

/* math ------------------------------------------------------------------ */
.mi{unicode-bidi:isolate; display:inline-block}
.nb{white-space:nowrap}
.mi svg{vertical-align:-0.22em}
.mb{unicode-bidi:isolate; text-align:center; margin:9pt 0 10pt;
    break-inside:avoid}
.mb svg{max-width:100%}
.mb.mixed{
  unicode-bidi:normal; direction:rtl; display:flex; flex-wrap:wrap;
  justify-content:center; gap:5pt 20pt;
}
.mitem{display:inline-flex; align-items:baseline; gap:5pt; direction:rtl}
.mtxt{font-family:"Plex Arabic","DejaVu Sans",sans-serif; font-size:9.8pt}
mjx-container, svg{color:inherit}

/* tables ---------------------------------------------------------------- */
.tw{margin:7pt 0 9pt; break-inside:avoid}
table{border-collapse:collapse; width:100%; font-size:8.9pt; line-height:1.55}
th,td{
  padding:3.6pt 6pt; text-align:start; vertical-align:top;
  border-bottom:.4pt solid var(--rule-soft);
  /* the build script stamps dir= on each cell from its own first strong
     character, so a Latin-leading value reads left to right inside a table
     that is otherwise right to left; the column stays right-aligned. */
  text-align:right;
}
thead th{
  font-weight:600; font-size:8.2pt; color:var(--muted);
  background:var(--tint); border-bottom:.7pt solid var(--rule);
  white-space:nowrap;
}
tbody tr:last-child td{border-bottom:.7pt solid var(--rule)}
td{font-variant-numeric:tabular-nums}
td.lead{font-weight:600}
.x{color:var(--bad); font-weight:700}
`;

const md = fs.readFileSync(SRC, 'utf8');
const html = `<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<title>تحكّم نشط مجدوَل على الموضع لقمع اهتزاز الفرز</title>
<style>${CSS}</style>
</head>
<body>
${convert(md)}
</body>
</html>
`;

fs.writeFileSync(OUT, html);
console.log(`wrote ${OUT}  (${(html.length / 1024).toFixed(0)} KB, `
            + `${(FONTS.length / 1024).toFixed(0)} KB of it fonts)`);
