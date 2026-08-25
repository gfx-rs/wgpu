import { chromium } from "playwright";
import express from "express";
import path from "path";

const PORT = 3000;
const TIMEOUT = 10000;
const BASE_URL = `http://127.0.0.1:${PORT}`;

const app = express();

async function init() {
  const browser = await chromium.launch({
    headless: process.argv.includes("--headless"),
    // Playwright >=1.59 sets this flag automatically, but listing
    // it since when this flag is not passed, it causes confusing errors
    args: ["--enable-unsafe-swiftshader"],
  });

  let context = await browser.newContext();
  context.setDefaultTimeout(TIMEOUT);
  return context;
}

let context = await init();

app.get("/gpu_report", async (req, res) => {
  const page = await context.newPage();
  let params = new URL(req.url, BASE_URL).searchParams;
  let wasm = params.get("wasm");

  let test_url = new URL(BASE_URL);
  test_url.search = new URLSearchParams({
    wasm,
    gpu_report: "true",
  }).toString();

  await page.goto(test_url.toString());

  await page
    .waitForFunction(() => {
      return window.sessionStorage.gpu_report;
    })
    .then((report) => {
      res.status(200).send(report.toString());
    });

  await page.close();
});

app.get("/run_test", async (req, res) => {
  const page = await context.newPage();
  let params = new URL(req.url, BASE_URL).searchParams;
  let wasm = params.get("wasm");
  let name = params.get("name");

  let test_url = new URL(BASE_URL);
  test_url.search = new URLSearchParams({ name, wasm }).toString();
  await page.goto(test_url.toString());

  await Promise.race([
    page
      .waitForFunction(() => {
        return window.sessionStorage.test_success;
      })
      .then(() => {
        res.sendStatus(200);
      }),
    page
      .waitForFunction(() => {
        return window.sessionStorage.test_failure;
      })
      .then((message) => {
        res.status(500).send(message.toString());
      }),
  ]);

  await page.close();
});

app.use("/", express.static(path.join(import.meta.dirname, "../dist")));

app.listen(PORT, () => {
  console.log(`WASM test server running at http://127.0.0.1:3000`);
});
