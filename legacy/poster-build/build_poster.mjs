import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const build = path.join(root, ".poster-build");
const out = path.join(root, "artifacts");
const skillDir = process.env.PRESENTATION_SKILL_DIR;
if (!skillDir) throw new Error("PRESENTATION_SKILL_DIR is required");
const { resolvePresentationFont, applyPresentationChartFont } = await import(
  pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href,
);
const family = resolvePresentationFont();

await fs.mkdir(build, { recursive: true });
await fs.mkdir(out, { recursive: true });

const W = 4493;
const H = 3178;
const C = {
  navy: "#0B1F33",
  navy2: "#173A56",
  teal: "#0F8B8D",
  teal2: "#54B8B8",
  cyan: "#DDF2F1",
  bluePale: "#EAF2F8",
  ink: "#132433",
  muted: "#526474",
  light: "#F5F8FA",
  line: "#C8D5DF",
  white: "#FFFFFF",
  orange: "#D97706",
  red: "#B6463A",
  green: "#2E7D5B",
};

const pres = Presentation.create({ slideSize: { width: W, height: H } });
const slide = pres.slides.add();
slide.background.fill = C.white;

function rect(x, y, w, h, fill, line = "none", radius = false) {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: line === "none" ? { fill: "none", width: 0 } : { fill: line, width: 2 },
  });
}

function textBox(text, x, y, w, h, opts = {}) {
  const s = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  s.text = text;
  s.text.style = {
    typeface: family,
    fontSize: opts.size ?? 28,
    bold: opts.bold ?? false,
    italic: opts.italic ?? false,
    color: opts.color ?? C.ink,
    alignment: opts.align ?? "left",
    verticalAlignment: opts.valign ?? "top",
    autoFit: opts.autoFit ?? "none",
  };
  return s;
}

function heading(label, x, y, w, number) {
  textBox(number, x, y - 3, 72, 58, { size: 34, bold: true, color: C.teal });
  textBox(label, x + 72, y, w - 72, 56, { size: 38, bold: true, color: C.navy });
  rect(x, y + 58, w, 4, C.teal);
}

async function addImage(rel, x, y, w, h, alt, fit = "contain") {
  rect(x - 4, y - 4, w + 8, h + 8, C.white, C.line, true);
  const bytes = new Uint8Array(await fs.readFile(path.join(root, rel)));
  return slide.images.add({
    blob: bytes,
    contentType: "image/png",
    alt,
    fit,
    position: { left: x, top: y, width: w, height: h },
    geometry: "roundRect",
    borderRadius: 12,
  });
}

function label(text, x, y, w, color = C.teal) {
  rect(x, y, w, 42, color, "none", true);
  textBox(text, x + 10, y + 3, w - 20, 34, { size: 21, bold: true, color: C.white, align: "center", valign: "middle" });
}

// Header
rect(0, 0, W, 330, C.navy);
rect(0, 330, W, 13, C.teal);
textBox("FROM FRAGMENTS TO STRUCTURE", 118, 56, 3100, 116, { size: 91, bold: true, color: C.white });
textBox("Reconstrucția probabilistică și conștientă de incertitudine a amprentelor parțiale", 122, 178, 3220, 70, { size: 41, color: "#CFE9EA" });
textBox("Răzvan-Cristian Alecse", 3470, 72, 880, 60, { size: 39, bold: true, color: C.white, align: "right" });
textBox("Proiect academic · Machine Learning · Computer Vision · Biometrie", 3310, 142, 1040, 80, { size: 25, color: "#BCD0DE", align: "right" });
label("PUBLIC / SYNTHETIC BENCHMARKS", 3490, 237, 410, C.teal);
label("TEST NIST RAPORTAT SEPARAT", 3914, 237, 430, C.orange);

// Research question band
rect(0, 343, W, 300, C.light);
textBox("ÎNTREBAREA CENTRALĂ", 120, 385, 710, 48, { size: 30, bold: true, color: C.teal });
textBox("Când regiunea lipsă nu este determinată unic, poate modelul să recupereze structură de creste plauzibilă și să exprime corect incertitudinea?", 120, 442, 2050, 130, { size: 36, bold: true, color: C.navy });
rect(2270, 390, 4, 190, C.line);
textBox("FORMULARE", 2350, 385, 430, 48, { size: 30, bold: true, color: C.teal });
textBox("Y = M ⊙ X     →     pθ(Xmissing | Xobserved, M)", 2350, 445, 1840, 75, { size: 42, bold: true, color: C.navy });
textBox("Separăm riguros: recuperarea ground truth · plauzibilitatea structurală · incertitudinea predictivă", 2350, 528, 1840, 48, { size: 27, color: C.muted });

const x1 = 120, x2 = 1537, x3 = 2954;
const cw1 = 1297, cw2 = 1297, cw3 = 1419;

// Column 1
heading("Design experimental", x1, 690, cw1, "01");
textBox(
  "• Split la nivel de subiect/deget; fără identități comune între train, validation și test.\n" +
  "• Metricile principale se calculează pe Ωmissing; pixelii observați sunt reinjectați prin data consistency.\n" +
  "• Modele probabilistice: single sample, mean-K, best-of-K, diversitate și calibrare — raportate separat.",
  x1, 770, cw1, 250, { size: 27, color: C.ink }
);

heading("Date și roluri", x1, 1044, cw1, "02");
rect(x1, 1125, cw1, 282, C.bluePale, "none", true);
textBox("SOCOFing", x1 + 28, 1145, 300, 45, { size: 31, bold: true, color: C.navy });
textBox("6.000 imagini · 600 subiecți\nMăști controlate, ablații, transfer inițial", x1 + 28, 1200, 555, 115, { size: 25, color: C.ink });
rect(x1 + 625, 1145, 3, 205, C.line);
textBox("NIST SD302", x1 + 665, 1145, 350, 45, { size: 31, bold: true, color: C.navy });
textBox("9.990 latente · 200 subiecți\nPerechi latent–exemplar și adnotări EFS; subset înregistrat aproximativ", x1 + 665, 1200, 580, 145, { size: 25, color: C.ink });
textBox("Înregistrarea folosește corespondențe de minutiae verificate; țintele NIST sunt registered_approximate, nu ground truth pixel-perfect.", x1 + 28, 1323, 1218, 65, { size: 21, italic: true, color: C.red });

heading("Pipeline reproductibil", x1, 1445, cw1, "03");
const py = 1530;
const pw = 211;
const gap = 54;
const stages = [
  ["Date", "SOCOFing / SD302"],
  ["Split", "subiect / deget"],
  ["Condiție", "Y, M, quality"],
  ["Modele", "det. + probabilistice"],
  ["K samples", "consistență date"],
];
stages.forEach((s, i) => {
  const xx = x1 + i * (pw + gap);
  rect(xx, py, pw, 128, i === 4 ? C.cyan : C.light, C.line, true);
  textBox(s[0], xx + 8, py + 18, pw - 16, 36, { size: 25, bold: true, color: C.navy, align: "center" });
  textBox(s[1], xx + 8, py + 63, pw - 16, 45, { size: 19, color: C.muted, align: "center" });
  if (i < stages.length - 1) {
    rect(xx + pw + 10, py + 60, gap - 20, 5, C.teal);
    textBox("›", xx + pw + 7, py + 36, gap - 14, 44, { size: 39, bold: true, color: C.teal, align: "center" });
  }
});
textBox("Evaluare → MAE/PSNR/SSIM · orientare π-periodică · frecvența crestelor · diversitate · coverage · teste pereche/mixed effects", x1, 1680, cw1, 72, { size: 22, color: C.muted });

heading("SOCOFing: structură vs. pixeli", x1, 1787, cw1, "04");
await addImage("outputs/gated_ridge_hybrid_full/reconstruction-preview.png", x1, 1870, cw1, 226, "Reconstrucție SOCOFing: target, observație, mască, predicții și eroare");
textBox("Model ridge-aware: Orientation error 0,2289 → 0,0991 (−56,7%), cu SSIM 0,2585 → 0,2837; MAE crește ușor 0,1740 → 0,1772.", x1, 2114, cw1, 96, { size: 26, bold: true, color: C.navy });
textBox("Interpretare: loss-ul structural poate îmbunătăți geometria crestelor fără a optimiza simultan fidelitatea pixel-wise.", x1, 2215, cw1, 82, { size: 23, color: C.muted });

heading("Modele comparate", x1, 2335, cw1, "05");
textBox(
  "Classical inpainting · U-Net · gated/partial convolution · CVAE global/spațial · conditional DDPM · residual DDPM · RePaint · latent diffusion · flow matching · BBDM · ensemble/structure-guided ablations",
  x1, 2420, cw1, 145, { size: 27, color: C.ink }
);
rect(x1, 2580, cw1, 284, C.navy, "none", true);
textBox("Principiul de siguranță", x1 + 30, 2606, cw1 - 60, 46, { size: 30, bold: true, color: C.teal2 });
textBox("Reconstrucție plauzibilă ≠ recuperarea amprentei reale lipsă.\nScop: benchmark academic, analiză structurală și cuantificarea incertitudinii — nu autentificare, impersonare sau artefacte fizice.", x1 + 30, 2668, cw1 - 60, 150, { size: 25, color: C.white });

// Column 2
heading("Înregistrare NIST SD302", x2, 690, cw2, "06");
await addImage("outputs/nist302_registration/registered_example.png", x2, 773, cw2, 304, "Exemplu de înregistrare latent–exemplar bazată pe corespondențe verificate");
textBox("Exemplu: 13 corespondențe oficiale, RMSE geometric ≈ 0,102 mm. Validitatea scade în afara înfășurătorii punctelor de ancorare.", x2, 1090, cw2, 80, { size: 23, color: C.muted });

heading("Validare probabilistică · K=5", x2, 1200, cw2, "07");
textBox("NIST SD302 · n=287 imagini de validare · aceeași sămânță și aceleași cazuri", x2, 1272, cw2, 42, { size: 21, color: C.muted });

const chartMae = slide.charts.add("bar", {
  position: { left: x2, top: 1328, width: 620, height: 420 },
  categories: ["DDPM", "Residual", "RePaint"],
  series: [{ name: "Mean MAE ↓", values: [0.1555, 0.1226, 0.1111], fill: C.teal, valuesFormatCode: "0.000" }],
  barOptions: { direction: "column", grouping: "clustered", gapWidth: 65 },
  hasLegend: false,
  xAxis: { visible: true, title: "Model" },
  yAxis: { visible: true, title: "MAE pe regiunea lipsă", min: 0, max: 0.18, numberFormatCode: "0.000" },
  dataLabels: { showValue: true, position: "outEnd", textStyle: { typeface: family, fontSize: 18, fill: C.navy, bold: true } },
  chartFill: C.white,
  plotAreaFill: C.white,
});
applyPresentationChartFont(chartMae, { fontFamily: family });

const chartSsim = slide.charts.add("bar", {
  position: { left: x2 + 657, top: 1328, width: 640, height: 420 },
  categories: ["DDPM", "Residual", "RePaint"],
  series: [{ name: "SSIM ↑", values: [0.2988, 0.3734, 0.4318], fill: C.navy2, valuesFormatCode: "0.000" }],
  barOptions: { direction: "column", grouping: "clustered", gapWidth: 65 },
  hasLegend: false,
  xAxis: { visible: true, title: "Model" },
  yAxis: { visible: true, title: "SSIM pe regiunea lipsă", min: 0, max: 0.50, numberFormatCode: "0.00" },
  dataLabels: { showValue: true, position: "outEnd", textStyle: { typeface: family, fontSize: 18, fill: C.navy, bold: true } },
  chartFill: C.white,
  plotAreaFill: C.white,
});
applyPresentationChartFont(chartSsim, { fontFamily: family });
textBox("RePaint are cea mai bună fidelitate, dar ≈31× timpul DDPM direct (19,35 vs 0,63 s/imagine). Residual DDPM este compromisul practic.", x2, 1758, cw2, 86, { size: 25, bold: true, color: C.navy });

heading("Test final NIST · n=217", x2, 1880, cw2, "08");
const chartTest = slide.charts.add("bar", {
  position: { left: x2, top: 1962, width: cw2, height: 490 },
  categories: ["Zero-shot\ndeterminist", "CVAE\nmean-K50", "Boundary\nfine-tune", "Support-spectrum\nfine-tune"],
  series: [{ name: "SSIM ↑", values: [0.4884, 0.3647, 0.3036, 0.2802], fill: C.orange, valuesFormatCode: "0.000" }],
  barOptions: { direction: "column", grouping: "clustered", gapWidth: 50 },
  hasLegend: false,
  xAxis: { visible: true },
  yAxis: { visible: true, title: "SSIM pe Ωmissing", min: 0, max: 0.55, numberFormatCode: "0.00" },
  dataLabels: { showValue: true, position: "outEnd", textStyle: { typeface: family, fontSize: 18, fill: C.navy, bold: true } },
  chartFill: C.white,
  plotAreaFill: C.white,
});
applyPresentationChartFont(chartTest, { fontFamily: family });

rect(x2, 2468, cw2, 390, C.bluePale, "none", true);
textBox("Rezultat contraintuitiv, dar important", x2 + 28, 2492, cw2 - 56, 45, { size: 31, bold: true, color: C.navy });
textBox("Pe testul final, baseline-ul determinist zero-shot este cel mai bun: MAE 0,1054 · SSIM 0,4884 · Orientation error 0,2764. Fine-tuning-ul pe ținte aproximativ înregistrate degradează performanța (MAE 0,1416 · SSIM 0,2802).", x2 + 28, 2555, cw2 - 56, 140, { size: 26, color: C.ink });
textBox("Concluzie metodologică: pseudo-ground-truth-ul geometric imperfect poate induce bias; îmbunătățirea vizuală/structurală nu garantează generalizare.", x2 + 28, 2715, cw2 - 56, 102, { size: 24, bold: true, color: C.red });

// Column 3
heading("Incertitudine: unde eșuează calibrarea?", x3, 690, cw3, "09");
const chartDist = slide.charts.add("line", {
  position: { left: x3, top: 780, width: cw3, height: 500 },
  categories: ["0–2 mm", "2–5 mm", ">5 mm"],
  series: [
    { name: "MAE", values: [0.1305, 0.1578, 0.1747], line: { fill: C.red, width: 5 }, marker: { symbol: "circle", size: 11 } },
    { name: "σ predictiv", values: [0.0877, 0.0887, 0.0841], line: { fill: C.teal, width: 5 }, marker: { symbol: "diamond", size: 11 } },
  ],
  lineOptions: { grouping: "standard", smooth: false },
  hasLegend: true,
  legend: { position: "top", overlay: false, textStyle: { typeface: family, fontSize: 20, fill: C.ink } },
  xAxis: { visible: true, title: "Distanță față de corespondențe verificate" },
  yAxis: { visible: true, min: 0.06, max: 0.19, numberFormatCode: "0.00" },
  chartFill: C.white,
  plotAreaFill: C.white,
});
applyPresentationChartFont(chartDist, { fontFamily: family });
textBox("Eroarea crește cu distanța, dar dispersia predictivă rămâne aproape constantă. Spearman uncertainty–error scade 0,527 → 0,262 → 0,167.", x3, 1292, cw3, 98, { size: 27, bold: true, color: C.navy });

heading("Predicții probabilistice pe test", x3, 1420, cw3, "10");
await addImage("outputs/nist302_TEST_FINAL_cvae_spatial/samples-preview.png", x3, 1502, cw3, 430, "CVAE spațial: observație, pseudo-target, medie, samples și hărți de incertitudine");
textBox("CVAE spațial, K=50: single MAE 0,1562 · mean-K MAE 0,1378 · best-of-50 MAE 0,1277 · SSIM(mean) 0,3647 · ρ(σ,|e|)=−0,1525.", x3, 1948, cw3, 86, { size: 24, color: C.ink });

heading("Ce susțin datele acum", x3, 2065, cw3, "11");
textBox(
  "1. Structura de ridge poate fi îmbunătățită fără câștig pixel-wise.\n" +
  "2. RePaint maximizează fidelitatea pe validation, cu un cost temporal mare.\n" +
  "3. Best-of-K măsoară potențialul distribuției, nu performanța unui singur sample.\n" +
  "4. Varianța samples nu este încă o incertitudine calibrată; corelația poate fi slabă sau negativă.\n" +
  "5. Transferul către latente reale este limitat de domain gap și de eroarea de înregistrare.",
  x3, 2145, cw3, 320, { size: 25, color: C.ink }
);

heading("Limitări și pasul următor", x3, 2505, cw3, "12");
rect(x3, 2586, cw3, 272, C.light, "none", true);
textBox("Limitări", x3 + 24, 2605, 310, 40, { size: 29, bold: true, color: C.red });
textBox("subset NIST selectat prin calitatea corespondențelor · deformare elastică nereprezentată complet · ținte pseudo-aliniate · rezoluție/compute limitate", x3 + 24, 2652, cw3 - 48, 78, { size: 23, color: C.ink });
textBox("Direcție prioritară", x3 + 24, 2735, 390, 40, { size: 29, bold: true, color: C.teal });
textBox("diffusion rezidual progresiv, condiționat explicit pe orientare/frecvență și antrenat pe patch-uri native; pierdere ponderată prin încredere geometrică + calibrare post-hoc pe subiecți", x3 + 24, 2780, cw3 - 48, 62, { size: 23, color: C.ink });

// Footer references and provenance
rect(0, 2920, W, 258, C.navy);
textBox("REFERINȚE CHEIE", 120, 2944, 360, 38, { size: 25, bold: true, color: C.teal2 });
textBox(
  "[1] Ho et al., DDPM, NeurIPS 2020 · [2] Lugmayr et al., RePaint, CVPR 2022 · [3] Rombach et al., Latent Diffusion, CVPR 2022 · [4] Li et al., BBDM, CVPR 2023 · [5] NIST SD302 / TN 2190 · [6] Hussein, Jain & Nandakumar, progressive diffusion fingerprint inpainting, IJCB 2026",
  120, 2990, 3280, 90, { size: 20, color: "#D9E4EC" }
);
textBox("Metrici din rulările locale ale proiectului · CI 95% salvate per imagine/subiect · cod și configurații reproductibile", 120, 3082, 3280, 34, { size: 19, italic: true, color: "#9EB5C6" });
textBox("GROUND TRUTH RECOVERY  ≠  STRUCTURAL PLAUSIBILITY  ≠  PREDICTIVE UNCERTAINTY", 3420, 2980, 950, 105, { size: 25, bold: true, color: C.white, align: "center" });
textBox("Academic research only", 3600, 3090, 590, 32, { size: 19, color: C.teal2, align: "center" });

slide.speakerNotes.textFrame.setText(
  "Poster academic construit din rezultatele proiectului local.\n" +
  "Proveniență rezultate: outputs/gated_ridge_hybrid_full/metrics.json; outputs/nist302_repaint_full_validation/metrics.json; outputs/nist302_residual_ddpm_full_validation/metrics.json; outputs/nist302_registered_ddpm_full/metrics.json; outputs/nist302_TEST_FINAL_zeroshot_deterministic/metrics.json; outputs/nist302_TEST_FINAL_finetuned_support_spectrum/metrics.json; outputs/nist302_TEST_FINAL_boundary_continuity/metrics.json; outputs/nist302_TEST_FINAL_cvae_spatial/metrics.json.\n" +
  "Surse: Ho et al., Denoising Diffusion Probabilistic Models, NeurIPS 2020, https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html ; Lugmayr et al., RePaint, CVPR 2022, https://openaccess.thecvf.com/content/CVPR2022/html/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.html ; Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models, CVPR 2022, https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html ; Li et al., BBDM, CVPR 2023, https://openaccess.thecvf.com/content/CVPR2023/html/Li_BBDM_Image-to-Image_Translation_With_Brownian_Bridge_Diffusion_Models_CVPR_2023_paper.html ; NIST SD302, https://www.nist.gov/itl/iad/btg/nist-special-database-302 ; NIST TN 2190, https://doi.org/10.6028/NIST.TN.2190 ; Hussein, Jain, Nandakumar, Progressive Learning of a Diffusion-based Inpainting Model for Separating Overlapped Fingerprints, arXiv:2608.03937."
);

const candidate = path.join(build, "poster-candidate.pptx");
await (await PresentationFile.exportPptx(pres)).save(candidate);
const preview = await pres.export({ slide, format: "png", scale: 0.5 });
await fs.writeFile(path.join(build, "poster-preview.png"), new Uint8Array(await preview.arrayBuffer()));
const layout = await slide.export({ format: "layout" });
await fs.writeFile(path.join(build, "poster-layout.json"), await layout.text());

console.log(JSON.stringify({ candidate, preview: path.join(build, "poster-preview.png"), layout: path.join(build, "poster-layout.json"), font: family }, null, 2));
