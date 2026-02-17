import { chromium } from 'playwright';

(async () => {
  const browser = await chromium.launch({
    args: ["--no-sandbox", "--enable-gpu"]
  });
  const page = await browser.newPage();

  await run_tests(browser, page);
})();

async function run_tests(browser, page) {
  await page.route(/report/, async route => {
    let params = new URL(route.request().url()).searchParams;
    let kind = params.get("kind");
    let message = params.get("message");

    if (kind == "Success") {
      console.log(message);
      await browser.close();
      process.exit(0);
    }
    else if (kind == "Failure") {
      console.error(message);
      await browser.close();
      process.exit(1);
    }
    else if (kind == "TestSuccess") {
      console.log(message);
    }
    else if (kind == "LogError") {
      console.error(message);
    }
    else if (kind == "LogWarn") {
      console.warn(message);
    }
    else if (kind == "LogInfo") {
      console.info(message);
    }
    else if (kind == "LogDebug") {
      console.debug(message);
    }
    else if (kind == "LogTrace") {
      console.trace(message);
    }

    route.fulfill();
  });

  await page.goto(`http://127.0.0.1:8000/index.html?testrunner=true`);
}
