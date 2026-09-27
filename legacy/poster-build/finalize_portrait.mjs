import path from "node:path";
import fs from "node:fs/promises";
import { pathToFileURL } from "node:url";

const workspaceDir = process.env.PROJECT_ROOT;
const SKILL_DIR = process.env.PRESENTATION_SKILL_DIR;
const RUNTIME_PYTHON = process.env.RUNTIME_PYTHON;
if (!workspaceDir || !SKILL_DIR || !RUNTIME_PYTHON) {
  throw new Error("PROJECT_ROOT, PRESENTATION_SKILL_DIR and RUNTIME_PYTHON are required");
}
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
