const { chromium } = require('playwright');
const http = require('http');
const fs = require('fs');
const path = require('path');

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
                '.json': 'application/json' };

function serve(root) {
  return new Promise(resolve => {
    const server = http.createServer((req, res) => {
      const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '');
      const file = path.join(root, rel || 'page.html');
      if (!file.startsWith(root)) { res.writeHead(403).end(); return; }
      fs.readFile(file, (err, buf) => {
        if (err) { res.writeHead(404).end(String(err)); return; }
        res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream' });
        res.end(buf);
      });
    });
    server.listen(0, '127.0.0.1', () => resolve(server));
  });
}

(async () => {
  const q = process.argv[2] || '';
  const root = path.resolve('.');
  const server = await serve(root);
  const port = server.address().port;
  const b = await chromium.launch({
    args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'],
  });
  const p = await b.newPage();
  const errs = [];
  p.on('pageerror', e => errs.push(String(e)));
  p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  await p.goto(`http://127.0.0.1:${port}/page.html` + (q ? '?' + q : ''));
  try {
    await p.waitForFunction(() => window.DONE === true, { timeout: 30000 });
  } catch (e) {
    console.error('FAILED\n' + errs.join('\n'));
    await b.close(); server.close(); process.exit(1);
  }
  process.stdout.write(await p.evaluate(() => window.RESULT));
  await b.close();
  server.close();
})();
