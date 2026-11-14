"use client";

import { useState, useRef, useEffect, ChangeEvent, DragEvent, KeyboardEvent, ReactNode } from "react";
import Image from "next/image";
import { RefreshCw, AlertTriangle, ImageOff, Cpu, Info, Loader2, ArrowRight, Layers, Crosshair } from "lucide-react";
import { COMPARISON_METRICS, REAL_SUBSET_METRICS } from "@/data/projectData";
import { INTERPRETATION } from "@/data/interpretation";
import { Eyebrow, Badge, SpecimenLabel, Dot, cx } from "@/components/ui";
import { ScanViewport, StatusLedger, ConfidenceValue, MeterBar, useElapsed, useReducedMotion, type RequestPhase } from "@/components/analysis";

interface ModelResult {
  model_key: string;
  model_name: string;
  prediction: "Healthy" | "Alternaria";
  label: "Healthy Leaf" | "Alternaria Leaf Spot";
  confidence: number;
  probabilities: { Healthy: number; Alternaria: number };
  inference_ms: number;
}

interface DiagnosisResult {
  status: "success" | "rejected_non_leaf" | "rejected_non_apple" | "rejected_low_confidence" | "error";
  mode?: "single" | "all";
  stage?: string;
  final_prediction?: "Healthy" | "Alternaria";
  final_label?: "Healthy Leaf" | "Alternaria Leaf Spot";
  confidence?: number;
  model_key?: string;
  model_name?: string;
  inference_ms?: number;
  probabilities?: { Healthy: number; Alternaria: number };
  model_results?: ModelResult[];
  results_note?: string;
  message?: string;
}

type ModelOptionId = "mobilenetv2" | "resnet18" | "vgg16" | "vgg19" | "all";

interface ModelOption {
  id: ModelOptionId;
  name: string;
  description: string;
  badge: string;
}

const sizeLabel = (mb: number) => (mb >= 1000 ? `${(mb / 1000).toFixed(2)} GB` : `${Math.round(mb)} MB`);
const MODEL_TRAITS: Record<string, string> = {
  mobilenetv2: "Compact and fastest; recommended.",
  resnet18: "Highest accuracy on the full test set.",
  vgg16: "Large network, slower inference.",
  vgg19: "Largest network, slowest inference.",
};
const realMetric = (id: string) => REAL_SUBSET_METRICS.find((m) => m.id === id);
const realAccuracy = (id: string) => {
  const r = realMetric(id);
  return r ? `${(r.testAccuracy * 100).toFixed(2)}% accuracy on original test photographs.` : "";
};
const MODEL_OPTIONS: ModelOption[] = [
  ...COMPARISON_METRICS.map((m) => ({
    id: m.id as ModelOptionId,
    name: m.name,
    description: `${MODEL_TRAITS[m.id] ?? ""} ${realAccuracy(m.id)}`.trim(),
    badge: sizeLabel(m.fileSizeMb),
  })),
  {
    id: "all",
    name: "All Models",
    description: "Run all four models on the same photograph and compare their predictions side by side.",
    badge: "4 models",
  },
];
const ALL_MODEL_NAMES = COMPARISON_METRICS.map((m) => m.name);

/* Sample specimens shipped with the frontend (copies of held-out test images). They go through the
   exact same upload path as a user's own file. */
const SAMPLES = [
  { file: "healthy-field.jpg", label: "Healthy", capture: "field", kind: "healthy" as const },
  { file: "alternaria-field.jpg", label: "Alternaria", capture: "field", kind: "alternaria" as const },
  { file: "alternaria-lab.jpg", label: "Alternaria", capture: "bench", kind: "alternaria" as const },
  { file: "healthy-lab.jpg", label: "Healthy", capture: "bench", kind: "healthy" as const },
];

/* "compare" is the All Models header: four separate results, never merged into one verdict. */
type Verdict = "healthy" | "alternaria" | "split" | "compare";
type GuidanceKey = Exclude<Verdict, "compare">;

const VERDICT_TONE: Record<Verdict, { band: string; border: string; text: string }> = {
  healthy: { band: "bg-forest-50/70", border: "border-forest-100", text: "text-forest-800" },
  alternaria: { band: "bg-rust-50/70", border: "border-rust-100", text: "text-rust-700" },
  split: { band: "bg-ochre-50/70", border: "border-ochre-100", text: "text-ochre-700" },
  compare: { band: "bg-paper/60", border: "border-line", text: "text-ink" },
};

const verdictDetail = (v: Verdict) =>
  v === "healthy"
    ? "No signs of Alternaria leaf spot were detected in this photograph."
    : v === "alternaria"
      ? "Lesion patterns consistent with Alternaria leaf spot, a fungal disease caused by Alternaria alternata."
      : v === "compare"
        ? "Each model analysed this photograph independently. Their results are listed below."
        : "The models gave different results for this photograph. A clearer image or expert inspection is advised.";

/** Headline of the screening report: specimen thumbnail, verdict, and a key figure on the right. */
function VerdictCard({
  verdict,
  kicker,
  label,
  preview,
  aside,
  children,
}: {
  verdict: Verdict;
  kicker: string;
  label: string;
  preview: string | null;
  aside: ReactNode;
  children: ReactNode;
}) {
  const tone = VERDICT_TONE[verdict];
  return (
    <div className={cx("rounded-lg border overflow-hidden bg-white shadow-hair reveal", tone.border)} style={{ ["--delay" as string]: "0ms" }}>
      <div className={cx("flex flex-wrap sm:flex-nowrap items-start gap-4 p-5 sm:p-6", tone.band)}>
        {preview && (
          <div className="relative w-16 h-16 sm:w-[84px] sm:h-[84px] shrink-0 rounded-md overflow-hidden border border-white shadow-hair bg-paper-2">
            <Image src={preview} alt="Analysed specimen" fill className="object-cover" unoptimized />
          </div>
        )}
        <div className="flex-1 min-w-0">
          <div className="text-xs font-medium text-ink-3">{kicker}</div>
          <h3 className={cx("font-display text-[1.75rem] sm:text-[2.1rem] leading-tight mt-1", tone.text)}>{label}</h3>
          <p className="text-[13px] text-ink-2 mt-1.5 leading-snug max-w-md">{verdictDetail(verdict)}</p>
        </div>
        <div className="basis-full sm:basis-auto shrink-0 flex sm:block items-baseline justify-between gap-3 pt-3 sm:pt-0 border-t sm:border-0 border-ink/10 sm:text-right">
          {aside}
        </div>
      </div>
      {children}
    </div>
  );
}

function ClassRow({ name, value, tone, emphasis, delay }: { name: string; value: number; tone: "forest" | "rust"; emphasis: boolean; delay: number }) {
  return (
    <div className="grid grid-cols-[1fr_auto] sm:grid-cols-[10rem_1fr_4rem] items-center gap-x-4 gap-y-1.5 text-[13px]">
      <span className={emphasis ? "text-ink font-medium" : "text-ink-3"}>{name}</span>
      <span className="col-span-2 sm:col-span-1 order-last sm:order-none">
        <MeterBar value={value} tone={tone} delay={delay} />
      </span>
      <span className={cx("text-right tabular", emphasis ? "text-ink font-medium" : "text-ink-3")}>{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

/** Compact reference notes for the outcome: key signs, context chips and a strip of next steps. */
function InterpretationPanel({ verdict, delay }: { verdict: GuidanceKey; delay: number }) {
  const info = INTERPRETATION[verdict];
  const tone = VERDICT_TONE[verdict];
  return (
    <div className="space-y-5 reveal" style={{ ["--delay" as string]: `${delay}ms` }}>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div className="rounded-md border border-line bg-white p-4">
          <div className="text-xs font-medium text-ink-3 mb-3">{info.signsTitle}</div>
          <ul className="space-y-2.5">
            {info.signs.map(({ icon: Icon, text }) => (
              <li key={text} className="flex items-center gap-2.5 text-[13px] text-ink">
                <span className={cx("w-6 h-6 rounded-full flex items-center justify-center shrink-0", tone.band)}>
                  <Icon className={cx("w-3.5 h-3.5", tone.text)} strokeWidth={1.75} />
                </span>
                {text}
              </li>
            ))}
          </ul>
        </div>
        <div className="rounded-md border border-line bg-white p-4">
          <div className="text-xs font-medium text-ink-3 mb-3">{info.factorsTitle}</div>
          <div className="flex flex-wrap gap-1.5">
            {info.factors.map((f) => (
              <span key={f} className="text-xs text-ink-2 bg-paper-2 border border-line rounded-full px-2.5 py-1">{f}</span>
            ))}
          </div>
          {info.lookalikes && (
            <>
              <div className="text-xs font-medium text-ink-3 mt-4 mb-2">Can be confused with</div>
              <div className="flex flex-wrap gap-1.5">
                {info.lookalikes.map((l) => (
                  <span key={l} className="text-xs text-ink-2 border border-dashed border-line-2 rounded-full px-2.5 py-1">{l}</span>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      <div className="space-y-3">
        <h3 className="text-lg text-ink">Next steps</h3>
        <ol className={cx("grid grid-cols-2 gap-3", info.steps.length === 3 ? "sm:grid-cols-3" : "sm:grid-cols-4")}>
          {info.steps.map(({ icon: Icon, title, text }, i) => (
            <li key={title} className="rounded-md border border-line bg-white p-3.5">
              <div className="flex items-center justify-between">
                <span className={cx("w-8 h-8 rounded-full flex items-center justify-center", tone.band)}>
                  <Icon className={cx("w-4 h-4", tone.text)} strokeWidth={1.75} />
                </span>
                <span className="text-xs text-ink-4 tabular">0{i + 1}</span>
              </div>
              <div className="mt-3 text-sm font-medium text-ink">{title}</div>
              <p className="text-xs text-ink-3 leading-snug mt-1">{text}</p>
            </li>
          ))}
        </ol>
        <p className="text-xs text-ink-3">{info.footnote}</p>
      </div>
    </div>
  );
}

// Test accuracy / size get their own columns where the panel is wide (sm–lg single-column layout, xl+);
// on phones and in the narrow lg two-column band they fold into a line under the model name.
const MODEL_GRID =
  "grid grid-cols-[1fr_auto] sm:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)_minmax(0,0.8fr)_4.5rem_4rem_4rem_4.5rem] lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1.1fr)_minmax(0,1fr)_4.5rem_4.5rem] xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)_minmax(0,0.8fr)_4.5rem_4rem_4rem_4.5rem] items-center gap-x-4 gap-y-1";
const WIDE_ONLY = "hidden sm:block lg:hidden xl:block";
const NARROW_ONLY = "block sm:hidden lg:block xl:hidden";

export default function DetectionPage() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<DiagnosisResult | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelOptionId>("mobilenetv2");
  const [sampleLoading, setSampleLoading] = useState<string | null>(null);

  // request lifecycle (drives the choreography; set only from real fetch events)
  const [phase, setPhase] = useState<RequestPhase>("idle");
  const [runId, setRunId] = useState(0);
  const [resultSeq, setResultSeq] = useState(0);
  const abortRef = useRef<AbortController | null>(null);
  const elapsedMs = useElapsed(isLoading, runId);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const readoutRef = useRef<HTMLElement>(null);
  const reducedMotion = useReducedMotion();
  const selectedOption = MODEL_OPTIONS.find((o) => o.id === selectedModel) ?? MODEL_OPTIONS[0];
  const runningModelNames = selectedModel === "all" ? ALL_MODEL_NAMES : [selectedOption.name];

  // abort any in-flight request when the page unmounts (navigation away)
  useEffect(() => () => abortRef.current?.abort(), []);

  const abortInFlight = () => {
    abortRef.current?.abort();
    abortRef.current = null;
  };

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith("image/")) {
      setApiError("Selected file must be a valid image (JPG, PNG, WEBP).");
      return;
    }
    abortInFlight();
    setApiError(null);
    setResult(null);
    setPhase("idle");
    setImageFile(file);
    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleReset = () => {
    abortInFlight();
    setIsLoading(false);
    setPhase("idle");
    setImageFile(null);
    setImagePreview(null);
    setResult(null);
    setApiError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleModelChange = (id: ModelOptionId) => {
    if (isLoading) return;
    setSelectedModel(id);
    // a previous result belongs to a different selection; clear it so results are never mis-attributed
    setResult(null);
    setApiError(null);
    setPhase("idle");
  };

  const handleOptionKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const i = MODEL_OPTIONS.findIndex((o) => o.id === selectedModel);
    let next = -1;
    if (e.key === "ArrowDown" || e.key === "ArrowRight") next = (i + 1) % MODEL_OPTIONS.length;
    if (e.key === "ArrowUp" || e.key === "ArrowLeft") next = (i - 1 + MODEL_OPTIONS.length) % MODEL_OPTIONS.length;
    if (e.key === "Home") next = 0;
    if (e.key === "End") next = MODEL_OPTIONS.length - 1;
    if (next >= 0) {
      e.preventDefault();
      handleModelChange(MODEL_OPTIONS[next].id);
      optionRefs.current[next]?.focus();
    }
  };

  const loadSample = async (file: string) => {
    if (isLoading) return;
    setSampleLoading(file);
    try {
      const res = await fetch(`/samples/${file}`);
      const blob = await res.blob();
      handleFileSelect(new File([blob], file, { type: blob.type || "image/jpeg" }));
    } catch {
      setApiError("Could not load the sample image.");
    } finally {
      setSampleLoading(null);
    }
  };

  const handleAnalyze = async () => {
    if (!imageFile) {
      setApiError("Please select or upload a leaf image first.");
      return;
    }

    abortInFlight();
    const controller = new AbortController();
    abortRef.current = controller;

    setIsLoading(true);
    setApiError(null);
    setResult(null);
    setPhase("prepared"); // the file is in memory and encoded into the multipart body
    setRunId((r) => r + 1); // starts the real elapsed-time clock for this run
    if (window.innerWidth < 1024) {
      // on stacked layouts the readout is below the controls: bring it into view
      readoutRef.current?.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" });
    }

    const formData = new FormData();
    formData.append("file", imageFile);
    formData.append("model", selectedModel);

    try {
      // Primary: Try relative /api/predict (proxied by Next.js); Fallback: Direct backend port 8000
      let response: Response;
      setPhase("awaiting"); // request dispatched; waiting for the backend
      try {
        response = await fetch("/api/predict", {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });
        if (response.status === 404) {
          response = await fetch("http://127.0.0.1:8000/api/predict", {
            method: "POST",
            body: formData,
            signal: controller.signal,
          });
        }
      } catch (err) {
        if (controller.signal.aborted) throw err;
        response = await fetch("http://127.0.0.1:8000/api/predict", {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });
      }
      setPhase("receiving"); // response headers arrived; reading the body

      if (!response.ok) {
        const errJson = await response.json().catch(() => null);
        const detailMsg = errJson?.detail || errJson?.message;
        if (response.status === 404) {
          throw new Error(
            "Backend endpoint not found (HTTP 404). Ensure FastAPI is running on http://127.0.0.1:8000."
          );
        }
        if (response.status === 503) {
          throw new Error(detailMsg || "A model checkpoint could not be loaded on the server (HTTP 503).");
        }
        throw new Error(detailMsg || `Inference error: HTTP ${response.status}`);
      }

      const data: DiagnosisResult = await response.json();
      if (controller.signal.aborted) return;
      setPhase("done");
      setResultSeq((s) => s + 1);
      setResult(data);
    } catch (err: unknown) {
      if (controller.signal.aborted) {
        return; // cancelled by reset / new upload / navigation: nothing to show
      }
      setPhase("error");
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes("Failed to fetch") || msg.includes("NetworkError")) {
        setApiError(
          "Could not connect to the inference backend. Please ensure the FastAPI backend is running via 'python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000'."
        );
      } else {
        setApiError(msg);
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        setIsLoading(false);
      }
    }
  };

  const isAllMode = result?.mode === "all";
  const isHealthy = result?.final_prediction === "Healthy";
  const isRejected = result?.status === "rejected_non_leaf" || result?.status === "rejected_non_apple" || result?.status === "rejected_low_confidence";
  // All Models: guidance follows the class only when every model reports it; otherwise the "models differ" notes.
  const allPredictions = new Set(result?.model_results?.map((r) => r.prediction));
  const allGuidance: GuidanceKey =
    allPredictions.size !== 1 ? "split" : allPredictions.has("Healthy") ? "healthy" : "alternaria";
  const modelCount = result?.model_results?.length ?? 4;

  return (
    <div className="relative">
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 lg:py-14 space-y-8">
        {/* Header */}
        <header className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-end border-b border-line pb-7">
          <div className="lg:col-span-8 space-y-4">
            <Eyebrow>Screening workspace</Eyebrow>
            <h1 className="text-[2.1rem] sm:text-5xl leading-[1.02] text-ink font-normal">Disease Detection</h1>
            <p className="text-base text-ink-2 leading-relaxed max-w-2xl">
              Upload a photograph of an apple leaf and select a model. The analysis reports whether the leaf appears healthy or shows
              Alternaria leaf spot, together with the model&apos;s confidence. Results support early screening and do not replace an expert
              diagnosis.
            </p>
          </div>
          <div className="lg:col-span-4 lg:justify-self-end">
            <SpecimenLabel className="lg:justify-end">
              <span className="inline-flex items-center gap-1.5">
                <Dot tone={isLoading ? "ochre" : "forest"} /> Inference on CPU
              </span>
              <span>·</span>
              <span>{selectedModel === "all" ? "All four models" : selectedOption.name}</span>
            </SpecimenLabel>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 items-start">
          {/* ============================================================ LEFT: INPUT */}
          <div className="lg:col-span-5 space-y-5">
            <section className="glass border rounded-xl">
              <div className="flex items-center justify-between px-5 pt-4 pb-3 border-b border-line">
                <h2 className="text-lg text-ink flex items-baseline gap-2"><span className="text-sm text-ink-4 tabular">01</span>Leaf photograph</h2>
                {imageFile && (
                  <button
                    type="button"
                    onClick={handleReset}
                    className="press text-xs text-ink-3 hover:text-rust-700 inline-flex items-center gap-1 cursor-pointer rounded-sm px-1"
                  >
                    <RefreshCw className="w-3 h-3" /> Replace
                  </button>
                )}
              </div>

              <div className="p-5 space-y-4">
                {!imagePreview ? (
                  <div
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                    onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), fileInputRef.current?.click())}
                    role="button"
                    tabIndex={0}
                    aria-label="Upload a leaf image"
                    className={cx(
                      "relative aspect-[16/10] lg:aspect-[4/3] max-h-[360px] lg:max-h-none w-full rounded-xl border border-dashed cursor-pointer flex flex-col items-center justify-center text-center p-6",
                      "transition-[background-color,border-color,transform] duration-200",
                      isDragging ? "border-forest-600 bg-forest-50/80 scale-[1.01]" : "border-line-2 bg-paper hover:bg-paper-2"
                    )}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/jpeg,image/png,image/webp,image/jpg"
                      onChange={handleInputChange}
                      className="hidden"
                    />
                    <Crosshair className={cx("w-6 h-6 mb-3 transition-transform duration-200", isDragging ? "text-forest-700 scale-110" : "text-ink-3")} strokeWidth={1.5} />
                    <div className="text-sm font-medium text-ink">{isDragging ? "Release to add the photograph" : "Drop a leaf photograph here"}</div>
                    <div className="text-xs text-ink-3 mt-1">or click to browse · JPG, PNG, WEBP</div>
                  </div>
                ) : (
                  <div className="space-y-2 fade">
                    <div className="relative aspect-[16/10] lg:aspect-[4/3] max-h-[360px] lg:max-h-none w-full rounded-xl overflow-hidden border border-line bg-paper-2">
                      <Image src={imagePreview} alt="Uploaded leaf specimen" fill className="object-contain" unoptimized />
                    </div>
                    <div className="flex items-center justify-between gap-3">
                      <SpecimenLabel className="min-w-0">
                        <span className="truncate max-w-[220px] normal-case tracking-normal font-mono text-ink-2">{imageFile?.name}</span>
                      </SpecimenLabel>
                      <SpecimenLabel>
                        {isLoading ? (
                          <span className="inline-flex items-center gap-1.5 text-forest-800"><Dot tone="forest" /> Analysing</span>
                        ) : (
                          <span>{imageFile ? `${(imageFile.size / 1024).toFixed(0)} KB` : ""}</span>
                        )}
                      </SpecimenLabel>
                    </div>
                  </div>
                )}

                {/* Sample specimens: real held-out photographs, loaded through the same upload path */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="text-xs text-ink-3">Or start from a sample held-out test photograph.</div>
                    {sampleLoading && <Loader2 className="w-3.5 h-3.5 animate-spin text-forest-600" aria-label="Loading sample" />}
                  </div>
                  <ul className="grid grid-cols-4 gap-2">
                    {SAMPLES.map((s) => {
                      const current = imageFile?.name === s.file;
                      return (
                        <li key={s.file}>
                          <button
                            type="button"
                            onClick={() => loadSample(s.file)}
                            disabled={isLoading || sampleLoading !== null}
                            aria-label={`Load sample: ${s.label} leaf, ${s.capture} photo`}
                            aria-pressed={current}
                            className={cx(
                              "group press relative w-full aspect-square rounded-lg overflow-hidden border bg-paper-2 text-left",
                              current ? "border-forest-700 ring-2 ring-forest-700/30" : "border-line hover:border-line-2",
                              isLoading || sampleLoading !== null ? "opacity-60 cursor-not-allowed" : "cursor-pointer"
                            )}
                          >
                            <Image src={`/samples/${s.file}`} alt="" fill sizes="120px" className="object-cover transition-transform duration-500 group-hover:scale-[1.05]" />
                            <span className="absolute inset-x-0 bottom-0 p-1.5 bg-gradient-to-t from-ink/60 to-transparent">
                              <span className="flex items-center gap-1 text-[11px] font-medium text-white leading-none">
                                <span className={cx("w-1.5 h-1.5 rounded-full shrink-0", s.kind === "healthy" ? "bg-forest-500" : "bg-rust-600")} aria-hidden />
                                <span className="truncate">{s.label}</span>
                              </span>
                              <span className="block text-[10px] text-white/75 leading-none mt-0.5 pl-2.5">{s.capture}</span>
                            </span>
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              </div>
            </section>

            {/* Model selection */}
            <section className="bg-white border border-line rounded-xl shadow-hair">
              <div className="flex items-center justify-between px-5 pt-4 pb-3 border-b border-line">
                <h2 className="text-lg text-ink flex items-baseline gap-2"><span className="text-sm text-ink-4 tabular">02</span>Analysis model</h2>
                <span className="text-xs text-ink-3">Default: MobileNetV2</span>
              </div>
              <div role="radiogroup" aria-label="Analysis model" className="p-2" onKeyDown={handleOptionKey}>
                {MODEL_OPTIONS.map((o, i) => {
                  const selected = selectedModel === o.id;
                  const isAll = o.id === "all";
                  return (
                    <button
                      key={o.id}
                      ref={(el) => {
                        optionRefs.current[i] = el;
                      }}
                      type="button"
                      role="radio"
                      aria-checked={selected}
                      tabIndex={selected ? 0 : -1}
                      onClick={() => handleModelChange(o.id)}
                      disabled={isLoading}
                      className={cx(
                        "press w-full text-left flex items-start gap-3 px-3 py-2.5 rounded-sm border",
                        isAll && "mt-1 border-dashed",
                        selected ? "glass-tint border-forest-700" : "border-transparent hover:bg-paper-2",
                        isLoading ? "opacity-60 cursor-not-allowed" : "cursor-pointer"
                      )}
                    >
                      <span
                        aria-hidden
                        className={cx(
                          "mt-1 w-3.5 h-3.5 rounded-full border flex items-center justify-center shrink-0 transition-colors",
                          selected ? "border-forest-700" : "border-line-2"
                        )}
                      >
                        <span className={cx("w-1.5 h-1.5 rounded-full bg-forest-700 transition-transform duration-150", selected ? "scale-100" : "scale-0")} />
                      </span>
                      <span className="flex-1 min-w-0">
                        <span className="flex items-center justify-between gap-2">
                          <span className="text-sm font-medium text-ink flex items-center gap-1.5">
                            {isAll ? <Layers className="w-3.5 h-3.5 text-forest-700" /> : <Cpu className="w-3.5 h-3.5 text-ink-4" />}
                            {o.name}
                          </span>
                          <Badge tone="mono">{o.badge}</Badge>
                        </span>
                        <span className="block text-xs text-ink-3 mt-0.5 leading-snug">{o.description}</span>
                      </span>
                    </button>
                  );
                })}
              </div>
              {selectedModel === "all" && (
                <p className="fade mx-4 mb-4 text-xs text-ochre-700 bg-ochre-50 border border-ochre-100 rounded-sm px-2.5 py-1.5 leading-snug">
                  The first comparison run loads the larger VGG networks and can take 10 to 30 seconds. Later runs are fast.
                </p>
              )}
            </section>

            {/* Primary action */}
            <button
              type="button"
              onClick={handleAnalyze}
              disabled={!imageFile || isLoading}
              className={cx(
                "press w-full py-3 px-4 rounded-md text-sm font-medium flex items-center justify-center gap-2 border",
                !imageFile || isLoading
                  ? "bg-paper-2 border-line text-ink-4 cursor-not-allowed"
                  : "bg-forest-800 border-forest-800 text-white hover:bg-forest-900 cursor-pointer shadow-hair"
              )}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analysing with {selectedOption.name}…
                </>
              ) : (
                <>
                  Analyse with {selectedOption.name}
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>

          </div>

          {/* ============================================================ RIGHT: READOUT */}
          <div className="lg:col-span-7 lg:sticky lg:top-24">
            <section ref={readoutRef} className="glass border rounded-xl shadow-lift min-h-[520px] flex flex-col scroll-mt-24" aria-busy={isLoading}>
              <div className="flex items-center justify-between px-5 sm:px-6 pt-4 pb-3 border-b border-line">
                <h2 className="text-lg text-ink flex items-baseline gap-2"><span className="text-sm text-ink-4 tabular">03</span>Screening report</h2>
                <SpecimenLabel>
                  {isLoading ? (
                    <span className="inline-flex items-center gap-1.5 text-forest-800"><span className="w-1.5 h-1.5 rounded-full bg-forest-600 pulse-soft" /> Analysing · {(elapsedMs / 1000).toFixed(1)} s</span>
                  ) : result?.status === "success" ? (
                    <span className="inline-flex items-center gap-1.5"><Dot tone="forest" /> Complete · {(elapsedMs / 1000).toFixed(1)} s</span>
                  ) : isRejected ? (
                    <span className="inline-flex items-center gap-1.5"><Dot tone="ochre" /> Not analysed</span>
                  ) : apiError || result?.status === "error" ? (
                    <span className="inline-flex items-center gap-1.5"><Dot tone="rust" /> Error</span>
                  ) : (
                    <span className="inline-flex items-center gap-1.5"><Dot tone="sage" /> Ready</span>
                  )}
                </SpecimenLabel>
              </div>

              <div className="p-5 sm:p-6 flex-1 flex flex-col">
                {/* Error */}
                {apiError && (
                  <div className="bg-rust-50 border border-rust-100 rounded-md p-4 text-xs text-rust-700 flex items-start gap-2.5 reveal" role="alert">
                    <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                    <div className="space-y-2 flex-1">
                      <div className="font-medium">Analysis could not be completed</div>
                      <div className="leading-relaxed">{apiError}</div>
                      <div className="flex flex-wrap gap-2">
                        {imageFile && (
                          <button
                            type="button"
                            onClick={handleAnalyze}
                            disabled={isLoading}
                            className="press inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-sm border border-rust-600/40 bg-white text-rust-700 hover:bg-rust-100 cursor-pointer"
                          >
                            <RefreshCw className="w-3.5 h-3.5" /> Retry with {selectedOption.name}
                          </button>
                        )}
                        <button type="button" onClick={handleReset} className="press inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-sm border border-line bg-white text-ink-2 hover:bg-paper-2 cursor-pointer">
                          Start over
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Idle */}
                {!result && !apiError && !isLoading && (
                  <div className="flex-1 bg-paper-2/60 rounded-lg border border-dashed border-line flex flex-col items-center justify-center text-center p-10 fade">
                    <div className="relative w-24 h-24 mb-5" aria-hidden>
                      <div className="absolute inset-0 rounded-full border border-line-2" />
                      <div className="absolute inset-4 rounded-full border border-line-2" />
                      <div className="absolute left-1/2 top-0 bottom-0 w-px bg-line-2" />
                      <div className="absolute top-1/2 left-0 right-0 h-px bg-line-2" />
                      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full bg-ink-3" />
                    </div>
                    <h3 className="text-xl text-ink">{imagePreview ? "Ready to analyse" : "Awaiting a leaf photograph"}</h3>
                    <p className="text-[13px] text-ink-3 max-w-sm mt-2 leading-relaxed">
                      {imagePreview
                        ? `Run the analysis with ${selectedOption.name} to generate the screening report.`
                        : "Upload or select a leaf photograph, choose a model and run the analysis. The screening report will appear here."}
                    </p>
                  </div>
                )}

                {/* Active analysis: the uploaded image under the instrument */}
                {isLoading && imagePreview && (
                  <div className="grid grid-cols-1 md:grid-cols-5 gap-5 fade">
                    <div className="md:col-span-3">
                      <ScanViewport src={imagePreview} alt="Uploaded leaf under analysis" active />
                    </div>
                    <div className="md:col-span-2 flex flex-col">
                      <h3 className="text-lg text-ink mb-3">{selectedModel === "all" ? "Running all four models" : `Running ${selectedOption.name}`}</h3>
                      <StatusLedger phase={phase} modelNames={runningModelNames} elapsedMs={elapsedMs} />
                    </div>
                  </div>
                )}

                {result && (
                  <div key={`${result.mode ?? result.status}-${resultSeq}`} className="flex-1 flex flex-col space-y-5">
                    {/* NOT SUITABLE FOR ANALYSIS */}
                    {isRejected && (
                      <div className="flex-1 flex flex-col items-center justify-center text-center rounded-lg border border-line bg-paper/70 px-6 py-10 reveal">
                        {imagePreview ? (
                          <div className="relative w-20 h-20 rounded-md overflow-hidden border border-line bg-paper-2 mb-5 opacity-80">
                            <Image src={imagePreview} alt="Submitted photograph" fill className="object-cover grayscale" unoptimized />
                          </div>
                        ) : (
                          <ImageOff className="w-7 h-7 text-ink-3 mb-4" strokeWidth={1.5} />
                        )}
                        <h3 className="text-xl text-ink">Image not suitable for analysis</h3>
                        <p className="text-[13px] text-ink-2 max-w-md mt-2 leading-relaxed">
                          This photograph does not clearly show an apple leaf, so no assessment was made. For a reliable result, please provide a
                          close-up of a single apple leaf.
                        </p>
                        <ul className="mt-5 grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs text-ink-3 w-full max-w-lg">
                          {["One leaf filling most of the frame", "Even daylight, no harsh flash", "Leaf surface sharply in focus"].map((tip) => (
                            <li key={tip} className="rounded-sm border border-line bg-white px-3 py-2">{tip}</li>
                          ))}
                        </ul>
                        <button type="button" onClick={handleReset} className="press mt-6 inline-flex items-center gap-1.5 text-xs font-medium px-3.5 py-2 rounded-md border border-line-2 bg-white text-ink hover:bg-paper-2 cursor-pointer">
                          Upload another photograph <ArrowRight className="w-3.5 h-3.5 text-ink-3" />
                        </button>
                      </div>
                    )}

                    {/* BACKEND-REPORTED ERROR */}
                    {result.status === "error" && (
                      <div className="bg-rust-50 border border-rust-100 rounded-md p-4 text-xs text-rust-700 flex items-start gap-2.5 reveal" role="alert">
                        <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                        <div className="space-y-2 flex-1">
                          <div className="font-medium">Analysis could not be completed</div>
                          <div className="leading-relaxed">{result.message}</div>
                          <button type="button" onClick={handleAnalyze} className="press inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-sm border border-rust-600/40 bg-white text-rust-700 hover:bg-rust-100 cursor-pointer">
                            <RefreshCw className="w-3.5 h-3.5" /> Retry
                          </button>
                        </div>
                      </div>
                    )}

                    {/* SUCCESS */}
                    {result.status === "success" && (
                      <div className="space-y-6">
                        {/* ALL MODELS: one row per model, results not combined */}
                        {isAllMode && result.model_results && (
                          <VerdictCard
                            verdict="compare"
                            kicker="All models"
                            label="Individual model results"
                            preview={imagePreview}
                            aside={
                              <>
                                <div className="text-xs text-ink-3">Models run</div>
                                <div className="font-display text-3xl text-ink tabular leading-none sm:mt-1.5">{modelCount}</div>
                              </>
                            }
                          >
                            <div role="table" aria-label="Per-model predictions">
                              <div role="row" className={cx(MODEL_GRID, "hidden sm:grid px-6 py-2.5 text-xs text-ink-3 border-b border-line bg-paper/50")}>
                                <span role="columnheader">Model</span>
                                <span role="columnheader">Prediction</span>
                                <span role="columnheader">P(Alternaria)</span>
                                <span role="columnheader" className="text-right">Confidence</span>
                                <span role="columnheader" className={cx(WIDE_ONLY, "text-right")}>Test acc.</span>
                                <span role="columnheader" className={cx(WIDE_ONLY, "text-right")}>Size</span>
                                <span role="columnheader" className="text-right">Inference</span>
                              </div>
                              <div className="divide-y divide-line">
                                {result.model_results.map((r, i) => {
                                  const healthy = r.prediction === "Healthy";
                                  const delay = 120 + i * 80;
                                  const metric = realMetric(r.model_key);
                                  const testAcc = metric ? `${(metric.testAccuracy * 100).toFixed(2)}%` : "–";
                                  const size = metric ? sizeLabel(metric.fileSizeMb) : "–";
                                  return (
                                    <div key={r.model_key} role="row" className={cx(MODEL_GRID, "px-5 sm:px-6 py-3.5 reveal")} style={{ ["--delay" as string]: `${delay}ms` }}>
                                      <span role="cell" className="text-sm font-medium text-ink min-w-0">
                                        <span className="block truncate">{r.model_name}</span>
                                        <span className={cx(NARROW_ONLY, "text-xs font-normal text-ink-3 tabular mt-0.5")}>
                                          <span className="whitespace-nowrap">Test {testAcc}</span> · <span className="whitespace-nowrap">{size}</span>
                                        </span>
                                      </span>
                                      <span role="cell" className={cx("order-last sm:order-none text-[13px] inline-flex items-center gap-1.5", healthy ? "text-forest-800" : "text-rust-700")}>
                                        <Dot tone={healthy ? "forest" : "rust"} /> {r.label}
                                      </span>
                                      <span role="cell" className="hidden sm:block">
                                        <MeterBar value={r.probabilities.Alternaria} tone="rust" delay={delay} />
                                      </span>
                                      <span role="cell" className="text-right">
                                        <ConfidenceValue value={r.confidence} delay={delay} className="font-display text-lg text-ink" />
                                      </span>
                                      <span role="cell" className={cx(WIDE_ONLY, "text-right text-xs text-ink-2 tabular")}>{testAcc}</span>
                                      <span role="cell" className={cx(WIDE_ONLY, "text-right text-xs text-ink-3 tabular")}>{size}</span>
                                      <span role="cell" className="hidden sm:block text-right text-xs text-ink-3 tabular">{r.inference_ms.toFixed(1)} ms</span>
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          </VerdictCard>
                        )}

                        {/* SINGLE MODEL */}
                        {!isAllMode && (
                          <VerdictCard
                            verdict={isHealthy ? "healthy" : "alternaria"}
                            kicker="Screening result"
                            label={result.final_label || (isHealthy ? "Healthy Leaf" : "Alternaria Leaf Spot")}
                            preview={imagePreview}
                            aside={
                              result.confidence !== undefined && (
                                <>
                                  <div className="text-xs text-ink-3">Model confidence</div>
                                  <ConfidenceValue
                                    value={result.confidence}
                                    delay={150}
                                    className={cx("block font-display text-3xl leading-none sm:mt-1.5", isHealthy ? "text-forest-800" : "text-rust-700")}
                                  />
                                </>
                              )
                            }
                          >
                            <div className="px-5 sm:px-6 py-5 space-y-4">
                              {result.probabilities && (
                                <div className="space-y-3">
                                  <div className="text-xs font-medium text-ink-3">Class probabilities</div>
                                  <ClassRow name="Healthy foliage" value={result.probabilities.Healthy} tone="forest" emphasis={isHealthy} delay={200} />
                                  <ClassRow name="Alternaria leaf spot" value={result.probabilities.Alternaria} tone="rust" emphasis={!isHealthy} delay={260} />
                                </div>
                              )}
                              {result.mode === "single" && result.model_name && (
                                <SpecimenLabel className="pt-3 border-t border-line">
                                  <span>Model · {result.model_name}</span>
                                  {result.inference_ms !== undefined && (
                                    <>
                                      <span>·</span>
                                      <span>Inference {result.inference_ms.toFixed(1)} ms</span>
                                    </>
                                  )}
                                  <span>·</span>
                                  <span>Input 224 × 224</span>
                                </SpecimenLabel>
                              )}
                            </div>
                          </VerdictCard>
                        )}

                        <InterpretationPanel
                          verdict={isAllMode ? allGuidance : isHealthy ? "healthy" : "alternaria"}
                          delay={isAllMode ? 480 : 220}
                        />

                        <div
                          className="pt-4 border-t border-line flex flex-col sm:flex-row sm:items-center justify-between gap-4 reveal"
                          style={{ ["--delay" as string]: isAllMode ? "560ms" : "300ms" }}
                        >
                          <p className="text-xs text-ink-3 leading-relaxed max-w-sm">
                            {isAllMode
                              ? "Each row is an independent prediction from one model. Test accuracy is each model's score on the 1,299 original test photographs, not on this image. "
                              : ""}
                            Confidence reflects the model&apos;s certainty for this photograph, not a confirmed diagnosis.
                          </p>
                          <div className="flex flex-wrap gap-2 shrink-0">
                            <button type="button" onClick={handleAnalyze} className="press inline-flex items-center gap-1.5 text-xs font-medium px-3.5 py-2 rounded-md border border-line bg-white text-ink-2 hover:bg-paper-2 cursor-pointer">
                              <RefreshCw className="w-3.5 h-3.5 text-ink-3" /> Run again
                            </button>
                            <button type="button" onClick={handleReset} className="press inline-flex items-center gap-1.5 text-xs font-medium px-3.5 py-2 rounded-md border border-line-2 bg-white text-ink hover:bg-paper-2 cursor-pointer">
                              Analyse another leaf <ArrowRight className="w-3.5 h-3.5 text-ink-3" />
                            </button>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </section>
          </div>

          {/* reading notes: after the readout on stacked layouts, under the controls on desktop */}
          <div className="lg:col-span-5">
            <div className="text-[13px] text-ink-3 leading-relaxed space-y-1.5 px-1 lg:-mt-2">
              <div className="flex items-center gap-1.5 text-ink-2 font-medium">
                <Info className="w-3.5 h-3.5 text-forest-700" /> Reading the result
              </div>
              <ul className="space-y-1 pl-5 list-disc marker:text-sage-400">
                <li>Each photograph is classified as healthy apple foliage or Alternaria leaf spot.</li>
                <li>Confidence is the model&apos;s probability for the predicted class; values close to 50% indicate an uncertain image.</li>
                <li>Other diseases, pest damage and nutrient disorders are outside the scope of these models.</li>
                <li>Results are for early screening and should be confirmed by a qualified agricultural expert before any treatment.</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
