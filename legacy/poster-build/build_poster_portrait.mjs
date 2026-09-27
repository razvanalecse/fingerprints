import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const root = "/Users/razvanalecse/Documents/Codex/2026-09-20/ac-ioneaz-ca-un-cercet-tor-2";
const build = path.join(root, ".poster-build");
const skillDir = "/Users/razvanalecse/.codex/plugins/cache/openai-primary-runtime/presentations/26.915.20218/skills/presentations";
const { resolvePresentationFont, applyPresentationChartFont } = await import(pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href);
const family = resolvePresentationFont();
const W = 3178, H = 4493;
const C = { ink:"#111827", gray:"#5B6470", pale:"#F5F7F9", line:"#111827", blue:"#0A5FA8", blue2:"#0B75BC", bluePale:"#EAF3FA", teal:"#0E8B8C", orange:"#D97706", red:"#B23A32", white:"#FFFFFF", green:"#2F855A" };

const p = Presentation.create({ slideSize:{ width:W, height:H } });
const s = p.slides.add();
s.background.fill = C.white;

function rect(x,y,w,h,fill,line="none",radius=false){return s.shapes.add({geometry:radius?"roundRect":"rect",position:{left:x,top:y,width:w,height:h},fill,line:line==="none"?{fill:"none",width:0}:{fill:line,width:3}})}
function txt(text,x,y,w,h,o={}){const q=s.shapes.add({geometry:"textbox",position:{left:x,top:y,width:w,height:h},fill:"none",line:{fill:"none",width:0}});q.text=text;q.text.style={typeface:family,fontSize:o.size??25,bold:o.bold??false,italic:o.italic??false,color:o.color??C.ink,alignment:o.align??"left",verticalAlignment:o.valign??"top",autoFit:o.autoFit??"none"};return q}
function rule(x,y,w){rect(x,y,w,6,C.ink)}
function head(label,x,y,w){txt(label,x,y,w,52,{size:34,bold:true});rule(x,y+54,w)}
async function img(rel,x,y,w,h,alt,fit="contain"){const b=new Uint8Array(await fs.readFile(path.join(root,rel)));rect(x-3,y-3,w+6,h+6,C.white,"#B6BEC7",true);return s.images.add({blob:b,contentType:"image/png",alt,fit,position:{left:x,top:y,width:w,height:h},geometry:"roundRect",borderRadius:8})}
function bullet(lines,x,y,w,h,size=24){return txt(lines.map(v=>"▪ "+v).join("\n"),x,y,w,h,{size,color:C.ink})}

const lx=90,lw=610,cx=760,cw=1658,rx=2480,rw=610;

// Header patterned after the supplied academic poster: small side identities, dominant central title.
txt("ACADEMIC RESEARCH POSTER · 2026",cx,60,cw,35,{size:20,color:C.gray,align:"center"});
txt("From Fragments to Structure",cx,105,cw,86,{size:66,bold:true,align:"center"});
txt("Uncertainty-Aware Reconstruction of Partial Fingerprints Using Diffusion Models",cx,194,cw,112,{size:44,bold:true,align:"center"});
txt("Răzvan-Cristian Alecse",cx,315,cw,40,{size:27,align:"center",color:C.gray});
rule(cx,375,cw);

txt("RESEARCH SCOPE",lx,102,lw,40,{size:24,bold:true,color:C.blue});
txt("Public and synthetic benchmarks\nNo authentication bypass\nNo physical artefact fabrication",lx,155,lw,122,{size:23});
rule(lx,305,lw);
txt("CORE DISTINCTION",lx,330,lw,38,{size:24,bold:true,color:C.red});
txt("Ground-truth recovery\n≠ structural plausibility\n≠ predictive uncertainty",lx,380,lw,130,{size:26,bold:true});

txt("DATASETS",rx,102,rw,40,{size:24,bold:true,color:C.blue});
txt("SOCOFing: 6,000 images / 600 subjects\nNIST SD302: 9,990 latents / 200 subjects\nNIST test: 217 unseen images",rx,155,rw,140,{size:23});
rule(rx,315,rw);
txt("LEAKAGE CONTROL",rx,340,rw,38,{size:24,bold:true,color:C.red});
txt("Subject/finger-disjoint splits.\nThresholds and model choices frozen before the final test.",rx,390,rw,112,{size:23});

// Left column
head("Problem formulation",lx,560,lw);
txt("A complete fingerprint X∈[0,1]ᴴˣᵂ is observed only through a binary mask M:",lx,635,lw,88,{size:24});
txt("Y = M ⊙ X",lx,735,lw,62,{size:42,bold:true,align:"center",color:C.blue});
txt("The target is not a single deterministic image, but the conditional distribution:",lx,816,lw,90,{size:24});
txt("pθ(Xmissing | Xobserved, M)",lx,918,lw,76,{size:34,bold:true,align:"center"});
txt("For severe partial observations, multiple completions may be structurally plausible while only one matches the unknown original.",lx,1010,lw,138,{size:23,color:C.gray});

head("Experimental setup",lx,1185,lw);
bullet([
  "8 observed fractions: r=0.10…0.80",
  "rectangular, irregular, central, peripheral, fragments and stripe masks",
  "exact reinjection of observed pixels",
  "single / mean-K / best-of-K reported separately",
  "K∈{5,10,20,50} samples"
],lx,1262,lw,330,23);

head("Fingerprint structure",lx,1625,lw);
txt("Axial ridge orientation is π-periodic:",lx,1705,lw,48,{size:23});
txt("dθ(θ,θ̂)=1−cos[2(θ−θ̂)]",lx,1765,lw,72,{size:31,bold:true,align:"center",color:C.blue});
await img("outputs/socofing_eda/orientation-diagnostic.png",lx,1860,lw,208,"ROI, ridge tangents and orientation coherence");
txt("Orientation is estimated only inside the fingerprint foreground; doubled-angle smoothing avoids discontinuities at θ≡θ+π.",lx,2082,lw,112,{size:22,color:C.gray});

head("Metrics",lx,2230,lw);
txt("Pixel fidelity",lx,2305,lw,38,{size:27,bold:true,color:C.blue});
txt("MAE = |Ω|⁻¹ Σᵢ∈Ω |Xᵢ−X̂ᵢ|\nPSNR = 10 log₁₀(MAX²/MSE)\nSSIM: luminance + contrast + structure",lx,2355,lw,168,{size:23});
txt("Structural fidelity",lx,2540,lw,38,{size:27,bold:true,color:C.blue});
txt("orientation error · ridge frequency error · ridge continuity",lx,2590,lw,104,{size:23});
txt("Probabilistic quality",lx,2705,lw,38,{size:27,bold:true,color:C.blue});
txt("diversity · uncertainty–error correlation · interval coverage · calibration width",lx,2755,lw,115,{size:23});

head("Statistical protocol",lx,2915,lw);
bullet([
  "paired comparisons on the same images",
  "subject-balanced confidence intervals",
  "Wilcoxon / paired t-test as distribution permits",
  "mixed-effects: metric ~ model × r + mask + dataset + (1|subject)",
  "effect size and multiple-comparison correction"
],lx,2992,lw,340,22);

head("Methodological risks",lx,3375,lw);
bullet([
  "synthetic-to-real domain gap",
  "pseudo-target misregistration",
  "selection bias toward correspondence-qualified latents",
  "pixel metrics may penalize valid ridge shifts",
  "sample variance is not automatically calibrated uncertainty"
],lx,3452,lw,305,22);
rect(lx,3790,lw,350,C.ink,"none",true);
txt("Safety & interpretation",lx+25,3818,lw-50,42,{size:28,bold:true,color:"#70C7E8"});
txt("A plausible generated ridge field must never be described as recovery of an individual's true missing fingerprint without evidence. The project is limited to academic benchmark reconstruction and uncertainty analysis.",lx+25,3880,lw-50,220,{size:23,color:C.white});
txt("github.com/razvanalecse/fingerprints",lx,4210,lw,35,{size:20,color:C.gray,align:"center"});

// Center column — visual narrative
head("Controlled partial-fingerprint reconstruction",cx,440,cw);
await img("outputs/claude_integration_showcase.png",cx,520,cw,1580,"SOCOFing qualitative comparison across masks and ridge-aware models");
txt("Qualitative comparison across six mask geometries. Ridge-aware variants recover sharper local flow, but the best structural reconstruction is not always the lowest-MAE reconstruction.",cx,2115,cw,76,{size:25,bold:true});

head("Conditional generative model",cx,2235,cw);
txt("xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε,     ε~N(0,I)",cx,2310,cw,62,{size:34,bold:true,align:"center",color:C.blue});
txt("LDDPM = E[‖ε − εθ(xₜ,t,Y,M)‖²]  +  λori Lori  +  λridge Lridge",cx,2380,cw,66,{size:31,bold:true,align:"center"});
txt("reverse diffusion  →  K conditional samples  →  data consistency  →  fidelity / structure / uncertainty",cx,2455,cw,48,{size:24,align:"center",color:C.gray});

head("Real-latent registration and evaluation",cx,2540,cw);
await img("outputs/nist302_registration/registered_example.png",cx,2620,cw,386,"NIST SD302 latent–exemplar registration using official correspondences");
txt("Examiner-verified minutiae correspondences define similarity/affine registration. The warped exemplar is labeled registered_approximate; evaluation reliability decreases with distance from verified anchors.",cx,3020,cw,84,{size:23});

head("Validation: conditional diffusion",cx,3140,cw);
const c1=s.charts.add("bar",{position:{left:cx,top:3218,width:790,height:470},categories:["DDPM","Residual","RePaint"],series:[{name:"MAE ↓",values:[0.1555,0.1226,0.1111],fill:C.teal,valuesFormatCode:"0.000"}],barOptions:{direction:"column",grouping:"clustered",gapWidth:55},hasLegend:false,xAxis:{visible:true},yAxis:{visible:true,title:"MAE on Ωmissing",min:0,max:0.18,numberFormatCode:"0.00"},dataLabels:{showValue:true,position:"outEnd",textStyle:{typeface:family,fontSize:18,fill:C.ink,bold:true}},chartFill:C.white,plotAreaFill:C.white});applyPresentationChartFont(c1,{fontFamily:family});
const c2=s.charts.add("bar",{position:{left:cx+835,top:3218,width:823,height:470},categories:["DDPM","Residual","RePaint"],series:[{name:"SSIM ↑",values:[0.2988,0.3734,0.4318],fill:C.blue,valuesFormatCode:"0.000"}],barOptions:{direction:"column",grouping:"clustered",gapWidth:55},hasLegend:false,xAxis:{visible:true},yAxis:{visible:true,title:"SSIM on Ωmissing",min:0,max:0.50,numberFormatCode:"0.00"},dataLabels:{showValue:true,position:"outEnd",textStyle:{typeface:family,fontSize:18,fill:C.ink,bold:true}},chartFill:C.white,plotAreaFill:C.white});applyPresentationChartFont(c2,{fontFamily:family});
txt("RePaint gives the strongest validation fidelity, but requires ≈31× the inference time of direct DDPM. Residual DDPM is the practical trade-off.",cx,3692,cw,74,{size:24,bold:true});

rect(cx,3795,cw,330,C.blue,"none",true);
txt("Findings",cx+28,3820,cw-56,46,{size:34,bold:true,color:C.white});
bullet([
  "Ridge-aware loss reduced SOCOFing orientation error by 56.7%, with a small MAE regression.",
  "Best-of-K improves with K, but does not represent deployable single-sample performance.",
  "Error increases far from verified geometry; predictive spread often fails to increase with it.",
  "On final NIST test, zero-shot deterministic transfer outperformed fine-tuning on approximate registered targets."
],cx+35,3880,cw-70,210,24);

head("Residual diffusion: mean, sample and uncertainty",cx,4145,cw);
await img("outputs/nist302_registered_residual_ddpm_full/samples-preview.png",cx,4218,cw,155,"Residual DDPM outputs and predictive uncertainty on NIST SD302");

// Right column
head("Model family",rx,560,rw);
txt("Deterministic baselines",rx,637,rw,38,{size:27,bold:true,color:C.blue});
txt("classical · U-Net · partial/gated convolution",rx,686,rw,80,{size:23});
txt("Probabilistic baselines",rx,785,rw,38,{size:27,bold:true,color:C.blue});
txt("global/spatial CVAE",rx,834,rw,58,{size:23});
txt("Generative restoration",rx,910,rw,38,{size:27,bold:true,color:C.blue});
txt("conditional DDPM · residual DDPM · RePaint · latent diffusion · flow matching · BBDM",rx,960,rw,125,{size:23});

head("Predictive uncertainty",rx,1125,rw);
txt("μᵢ = K⁻¹ Σₖ X̂ᵢ⁽ᵏ⁾",rx,1200,rw,55,{size:31,bold:true,align:"center"});
txt("σᵢ² = (K−1)⁻¹ Σₖ(X̂ᵢ⁽ᵏ⁾−μᵢ)²",rx,1260,rw,70,{size:29,bold:true,align:"center"});
txt("Calibration asks whether pixels assigned larger σ are actually more erroneous and whether predictive intervals attain nominal coverage.",rx,1345,rw,132,{size:23});
const c3=s.charts.add("line",{position:{left:rx,top:1490,width:rw,height:450},categories:["0–2 mm","2–5 mm",">5 mm"],series:[{name:"MAE",values:[0.1305,0.1578,0.1747],line:{fill:C.red,width:5},marker:{symbol:"circle",size:10}},{name:"σ",values:[0.0877,0.0887,0.0841],line:{fill:C.teal,width:5},marker:{symbol:"diamond",size:10}}],lineOptions:{grouping:"standard",smooth:false},hasLegend:true,legend:{position:"top",textStyle:{typeface:family,fontSize:18,fill:C.ink}},xAxis:{visible:true,title:"Distance to verified points"},yAxis:{visible:true,min:0.06,max:0.19,numberFormatCode:"0.00"},chartFill:C.white,plotAreaFill:C.white});applyPresentationChartFont(c3,{fontFamily:family});
txt("MAE rises, while predictive σ stays nearly flat. Spearman ρ falls from 0.527 to 0.167.",rx,1948,rw,90,{size:23,bold:true,color:C.red});

head("Final NIST test · n=217",rx,2080,rw);
const c4=s.charts.add("bar",{position:{left:rx,top:2160,width:rw,height:500},categories:["Zero-shot","CVAE\nmean-K","Boundary","Support\nFT"],series:[{name:"SSIM ↑",values:[0.4884,0.3647,0.3036,0.2802],fill:C.orange,valuesFormatCode:"0.000"}],barOptions:{direction:"column",grouping:"clustered",gapWidth:45},hasLegend:false,xAxis:{visible:true},yAxis:{visible:true,min:0,max:0.55,title:"SSIM",numberFormatCode:"0.00"},dataLabels:{showValue:true,position:"outEnd",textStyle:{typeface:family,fontSize:17,fill:C.ink,bold:true}},chartFill:C.white,plotAreaFill:C.white});applyPresentationChartFont(c4,{fontFamily:family});
txt("Zero-shot deterministic baseline",rx,2673,rw,42,{size:26,bold:true,color:C.blue});
txt("MAE 0.1054 · SSIM 0.4884\nOrientation error 0.2764",rx,2723,rw,92,{size:23});
txt("Fine-tuning on approximate targets degraded test fidelity, revealing pseudo-ground-truth bias rather than a validated gain.",rx,2830,rw,118,{size:23,bold:true,color:C.red});

head("Single vs. mean vs. best-of-K",rx,2990,rw);
txt("Single:  E[d(X,X̂⁽¹⁾)]",rx,3070,rw,45,{size:27,bold:true});
txt("Mean:    d(X, K⁻¹ΣₖX̂⁽ᵏ⁾)",rx,3125,rw,45,{size:27,bold:true});
txt("Best:    minₖ d(X,X̂⁽ᵏ⁾)",rx,3180,rw,45,{size:27,bold:true});
txt("CVAE test, K=50: single MAE 0.1562; mean 0.1378; best-of-50 0.1277; uncertainty–error ρ=−0.1525.",rx,3250,rw,130,{size:23});

head("Interpretation",rx,3420,rw);
txt("Positive evidence",rx,3500,rw,38,{size:27,bold:true,color:C.green});
txt("diffusion improves validation fidelity; structural losses can preserve ridge flow; multiple samples expose ambiguity",rx,3550,rw,120,{size:23});
txt("Negative evidence",rx,3690,rw,38,{size:27,bold:true,color:C.red});
txt("uncertainty is miscalibrated; fine-tuning can overfit registration artefacts; latent diffusion underperforms without a stronger codec",rx,3740,rw,125,{size:23});

head("Next model",rx,3905,rw);
txt("Progressive structure-conditioned residual diffusion",rx,3982,rw,76,{size:28,bold:true,color:C.blue});
bullet([
  "native-resolution ridge patches",
  "orientation + frequency conditioning",
  "geometry-confidence weighted losses",
  "elastic registration uncertainty",
  "subject-level calibration"
],rx,4072,rw,255,22);

// Footer
rule(90,4390,3000);
txt("[1] Ho et al., DDPM, NeurIPS 2020  ·  [2] Lugmayr et al., RePaint, CVPR 2022  ·  [3] Rombach et al., Latent Diffusion, CVPR 2022  ·  [4] Li et al., BBDM, CVPR 2023  ·  [5] NIST SD302 / TN 2190  ·  [6] Hussein, Jain & Nandakumar, progressive diffusion fingerprint inpainting, IJCB 2026",90,4410,3000,52,{size:16,color:C.gray});

s.speakerNotes.textFrame.setText("Poster A0 portrait styled after the user-provided academic reference. Local evidence: outputs/claude_integration_showcase.png; outputs/socofing_eda/orientation-diagnostic.png; outputs/nist302_registration/registered_example.png; outputs/nist302_registered_residual_ddpm_full/samples-preview.png; metrics from outputs/gated_claude_hybrid_full/metrics.json, outputs/nist302_repaint_full_validation/metrics.json, outputs/nist302_residual_ddpm_full_validation/metrics.json, outputs/nist302_TEST_FINAL_zeroshot_deterministic/metrics.json, outputs/nist302_TEST_FINAL_finetuned_support_spectrum/metrics.json, outputs/nist302_TEST_FINAL_cvae_spatial/metrics.json. Sources: https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html ; https://openaccess.thecvf.com/content/CVPR2022/html/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.html ; https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html ; https://openaccess.thecvf.com/content/CVPR2023/html/Li_BBDM_Image-to-Image_Translation_With_Brownian_Bridge_Diffusion_Models_CVPR_2023_paper.html ; https://www.nist.gov/itl/iad/btg/nist-special-database-302 ; https://doi.org/10.6028/NIST.TN.2190 ; https://arxiv.org/abs/2608.03937");

const candidate=path.join(build,"poster-portrait-candidate.pptx");
await (await PresentationFile.exportPptx(p)).save(candidate);
const preview=await p.export({slide:s,format:"png",scale:0.5});
await fs.writeFile(path.join(build,"poster-portrait-preview.png"),new Uint8Array(await preview.arrayBuffer()));
const layout=await s.export({format:"layout"});
await fs.writeFile(path.join(build,"poster-portrait-layout.json"),await layout.text());
console.log(JSON.stringify({candidate,preview:path.join(build,"poster-portrait-preview.png"),font:family},null,2));
