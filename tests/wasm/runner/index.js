import { chromium } from 'playwright';
import express from 'express';
import path from 'path';

const app = express();
const port = 3000;


const browser = await chromium.launch({
  headless: process.argv.includes("--headless"),
  args: ["--no-sandbox", "--enable-gpu"]
});

const BASE_URL = 'http://127.0.0.1:3000';

app.get('/gpu_report', async (req, res) => {
  const page = await browser.newPage();
  let params = new URL(req.url, BASE_URL).searchParams;
  let wasm = params.get("wasm");

  let wait_for_report = page.waitForFunction(async () => {
    return window.gpu_report != null && await window.gpu_report()
  });

  let test_url = new URL(BASE_URL);
  test_url.search = new URLSearchParams({ wasm }).toString();
  await page.goto(test_url.toString());

  let report = (await wait_for_report).toString();

  res.status(200).send(report);
});

app.get('/run_test', async (req, res) => {
  const page = await browser.newPage();
  let params = new URL(req.url, BASE_URL).searchParams;
  let wasm = params.get("wasm");
  let name = params.get("name");

  let test_url = new URL(BASE_URL);
  test_url.search = new URLSearchParams({ name, wasm }).toString();
  await page.goto(test_url.toString());

  await Promise.race([
    page.waitForFunction(() => {
      return window.sessionStorage.test_success
    }).then(() => {
      res.sendStatus(200)
    }),
    page.waitForFunction(() => {
      return window.sessionStorage.test_failure
    }).then((message) => {
      res.status(500).send(message.toString())
    }),
  ]);

  await page.close();
});

app.use('/', express.static(path.join(import.meta.dirname, '../dist')))

app.listen(port, () => {
  console.log(`WASM test server running at http://127.0.0.1:3000`)
});
