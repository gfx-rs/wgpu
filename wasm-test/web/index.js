import init, { tests } from "./test.js"

export function start() {
  init().then(async () => {
    await tests();
  });
}
