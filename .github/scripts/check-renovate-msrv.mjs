import assert from "node:assert/strict";
import { globSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

assert(
  process.argv[2],
  "Usage: node check-renovate-msrv.mjs <renovate package directory>",
);
// CI pins Renovate because this check uses its internal API.
const load = (path) =>
  import(pathToFileURL(resolve(process.argv[2], `dist/${path}.js`)));
const { init } = await load("logger/index");
await init();
const { parse: parseToml } = await load("util/toml");
const { parseSingleYaml } = await load("util/yaml");
const { matchRegexOrGlobList } = await load("util/string-match");
const { extractPackageFile } = await load("modules/manager/custom/regex/index");
const { extractPackageFile: extractToolchain } = await load(
  "modules/manager/rust-toolchain/extract",
);
const { applyPackageRules } = await load("util/package-rules/index");
const { doAutoReplace } = await load(
  "workers/repository/update/branch/auto-replace",
);
const { GlobalConfig } = await load("config/global");

const read = (file) => readFileSync(file, "utf8");
const config = JSON.parse(read("renovate.json"));
const version = parseToml(read("rust-toolchain.toml")).toolchain.channel;
assert.match(version, /^\d+\.\d+(?:\.\d+)?$/);
const nextVersion = `${version.split(".")[0]}.${Number(version.split(".")[1]) + 1}`;
const managers = config.customManagers.filter(
  (manager) => manager.depNameTemplate === "rust",
);
assert(managers.length > 0, "No custom manager for Rust");

function expectedUpdate(file, content) {
  let values = [];
  let linePattern;
  if (file === "rust-toolchain.toml") {
    values = [parseToml(content).toolchain.channel];
    linePattern = /^channel\s*=.*$/gm;
  } else if (file === "Cargo.toml") {
    values = [parseToml(content).workspace.package["rust-version"]];
    linePattern = /^rust-version\s*=.*$/gm;
  } else if (/\.ya?ml$/.test(file)) {
    const value = parseSingleYaml(content).env?.REPO_MSRV;
    if (value !== undefined) {
      values = [value];
      linePattern = /^\s*REPO_MSRV\s*:.*$/gm;
    }
  } else if (file === "README.md") {
    const statements = [
      /If you're running our tests or examples,[^.]*\*\*([^*]+)\*\*/,
      /The rest of the workspace[^\n]*\*\*([^*]+)\*\*/,
    ];
    let output = content;
    for (const pattern of statements) {
      const match = content.match(pattern);
      assert(match, `${file}: missing repository MSRV statement`);
      assert.equal(
        match[1],
        version,
        `${file}: repository MSRV differs from toolchain`,
      );
      output = output.replace(
        match[0],
        match[0].replace(`**${version}**`, `**${nextVersion}**`),
      );
    }
    return { output, count: statements.length };
  }
  for (const value of values) {
    assert.equal(
      value,
      version,
      `${file}: repository MSRV differs from toolchain`,
    );
  }
  if (!values.length) return { output: content, count: 0 };
  const lines = [...content.matchAll(linePattern)];
  assert.equal(
    lines.length,
    values.length,
    `${file}: expected one MSRV declaration`,
  );
  return {
    output: content.replace(linePattern, (line) =>
      line.replace(version, nextVersion),
    ),
    count: values.length,
  };
}

async function checkRules(dep, manager, file) {
  for (const updateType of ["minor", "patch"]) {
    const rules = await applyPackageRules({
      ...dep,
      packageName: dep.packageName ?? dep.depName,
      manager,
      updateType,
      packageRules: config.packageRules,
    });
    assert.equal(rules.enabled, true, `${file}: Rust updates are disabled`);
    assert.equal(
      rules.groupName,
      "Rust repository MSRV",
      `${file}: wrong update group`,
    );
    assert.equal(
      rules.minimumReleaseAge,
      "18 weeks",
      `${file}: wrong release delay`,
    );
  }
}

const files = globSync(
  [
    "**/Cargo.toml",
    ".github/workflows/*.{yml,yaml}",
    "README.md",
    "rust-toolchain.toml",
  ],
  {
    exclude: [
      "**/node_modules/**",
      "**/target/**",
      "**/.worktrees/**",
      "**/.git/**",
    ],
  },
);
const scratch = mkdtempSync(join(tmpdir(), "wgpu-renovate-msrv-"));
GlobalConfig.set({ localDir: scratch });
let total = 0;
try {
  for (const file of files) {
    const content = read(file);
    const expected = expectedUpdate(file, content);
    const matches = [];
    for (const custom of managers) {
      if (!matchRegexOrGlobList(file, custom.managerFilePatterns)) continue;
      const extracted = extractPackageFile(content, file, custom);
      for (const [depIndex, dep] of (extracted?.deps ?? []).entries()) {
        matches.push({ ...custom, ...dep, depIndex, manager: "regex" });
      }
    }
    if (file === "rust-toolchain.toml") {
      assert.notEqual(
        config["rust-toolchain"]?.enabled,
        false,
        "Rust toolchain manager is disabled",
      );
      const extracted = extractToolchain(content, file);
      for (const [depIndex, dep] of extracted.deps.entries()) {
        matches.push({ ...dep, depIndex, manager: "rust-toolchain" });
      }
    }
    assert.equal(
      matches.length,
      expected.count,
      `${file}: Renovate did not match the expected MSRV declarations`,
    );
    let output = content;
    for (const dep of matches) {
      assert.equal(
        dep.currentValue,
        version,
        `${file}: Renovate extracted the wrong version`,
      );
      await checkRules(dep, dep.manager, file);
      output = await doAutoReplace(
        {
          ...dep,
          packageFile: file,
          newValue: nextVersion,
          autoReplaceGlobalMatch: true,
        },
        output,
        false,
      );
    }
    assert.equal(
      output,
      expected.output,
      `${file}: Renovate changed the wrong content`,
    );
    total += matches.length;
  }
} finally {
  rmSync(scratch, { recursive: true, force: true });
}
console.log(`Checked ${total} MSRV declarations across ${files.length} files.`);
