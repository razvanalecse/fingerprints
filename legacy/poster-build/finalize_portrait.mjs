import path from "node:path";
import fs from "node:fs/promises";
import { pathToFileURL } from "node:url";

const workspaceDir = "/Users/razvanalecse/Documents/Codex/2026-09-20/ac-ioneaz-ca-un-cercet-tor-2";
const SKILL_DIR = "/Users/razvanalecse/.codex/plugins/cache/openai-primary-runtime/presentations/26.915.20218/skills/presentations";
const RUNTIME_PYTHON = "/Users/razvanalecse/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR,"container_tools/artifact_tool_utils.mjs")).href);
const stagingDir = path.join(workspaceDir,".poster-build","finalizer");
const candidatePath = path.join(workspaceDir,".poster-build","poster-portrait-candidate.pptx");
const finalPath = path.join(workspaceDir,"artifacts","fingerprint_reconstruction_academic_poster_A0_portrait.pptx");
await fs.mkdir(stagingDir,{recursive:true});
await fs.mkdir(path.dirname(finalPath),{recursive:true});
const result = await finalizePresentation({
  workspaceDir,
  candidatePath,
  finalPath,
  explicitTotalSlideCount:1,
  requiredNativeTableOwnerSlides:[],
  requiredNativeChartOwnerSlides:[1],
  materializeLiteralChartWorkbooks:true,
  pythonExecutable:RUNTIME_PYTHON,
  integrityValidatorPath:path.join(SKILL_DIR,"container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath:path.join(SKILL_DIR,"container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs:["--expected-slide-size-emu","30270450,42795825","--validate-heading-fit"],
  fontPolicy:{basis:"design",families:["Helvetica Neue"]},
  verifyArtifactToolImport:true,
  receiptPath:path.join(stagingDir,"poster.validation.json"),
});
console.log(JSON.stringify(result,null,2));
