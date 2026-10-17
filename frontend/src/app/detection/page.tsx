"use client";

import { useState, useRef, useEffect, ChangeEvent, DragEvent, KeyboardEvent } from "react";
import Image from "next/image";
import {
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  ShieldAlert,
  ShieldCheck,
  Cpu,
  Info,
  Loader2,
  ArrowRight,
  Layers,
  Crosshair,
} from "lucide-react";
import { COMPARISON_METRICS, REAL_SUBSET_METRICS } from "@/data/projectData";
import { Eyebrow, Badge, SpecimenLabel, Dot, cx } from "@/components/ui";
import { ScanViewport, StatusLedger, ConfidenceValue, ProbabilityBar, useElapsed, useReducedMotion, type RequestPhase } from "@/components/analysis";

interface TopCrop {
  crop: string;
  confidence: number;
}

interface ModelResult {
  model_key: string;
  model_name: string;
  prediction: "Healthy" | "Alternaria";
  label: "Healthy Leaf" | "Alternaria Leaf Spot";
  confidence: number;
  probabilities: { Healthy: number; Alternaria: number };
  inference_ms: number;
}

interface Agreement {
  unanimous: boolean;
  votes: { Healthy: number; Alternaria: number };
  majority_prediction: "Healthy" | "Alternaria" | null;
  majority_label: string;
  disagreeing_models: string[];
  note: string;
}

interface DiagnosisResult {
  status: "success" | "rejected_non_leaf" | "rejected_non_apple" | "rejected_low_confidence" | "error";
  mode?: "ensemble" | "single" | "all";
  stage?: string;
  similarity?: number;
  gatekeeper_threshold?: number;
  crop_status?: "confirmed" | "likely_apple" | "ambiguous_foliage";
  crop_notice?: string;
  predicted_crop?: string;
  crop_confidence?: number;
  apple_confidence?: number;
  top3_crops?: TopCrop[];
  final_prediction?: "Healthy" | "Alternaria";
  final_label?: "Healthy Leaf" | "Alternaria Leaf Spot";
  confidence?: number;
  confidence_type?: "model_confidence";
  confidence_note?: string;
  explanation?: string;
  model_key?: string;
  model_name?: string;
  inference_ms?: number;
  probabilities?: { Healthy: number; Alternaria: number };
  model_results?: ModelResult[];
  agreement?: Agreement;
  treatment_guidance?: {
    title: string;
    pathogen: string;
    symptoms: string;
    recommended_action: string;
  } | null;
  message?: string;
  limitation_note?: string;
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
  mobilenetv2: "Small and fast; recommended.",
  resnet18: "Best on the full test set.",
  vgg16: "Large file, slower to run.",
  vgg19: "Largest file, slowest to run.",
};
const realAccuracy = (id: string) => {
  const r = REAL_SUBSET_METRICS.find((m) => m.id === id);
  return r ? `${(r.testAccuracy * 100).toFixed(2)}% correct on original test photographs.` : "";
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
  const confidenceLabel = result?.mode === "single" ? "Confidence score" : "Combined confidence";
  const isHealthy = result?.final_prediction === "Healthy";

  return (
    <div className="relative">
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 lg:py-14 space-y-8">
        {/* Header */}
        <header className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-end border-b border-line pb-7">
          <div className="lg:col-span-8 space-y-4">
            <Eyebrow>Leaf check · three steps</Eyebrow>
            <h1 className="text-[2.1rem] sm:text-5xl leading-[1.02] text-ink font-normal">Disease Detection</h1>
            <p className="text-base text-ink-2 leading-relaxed max-w-2xl">
              Upload a photograph of an apple leaf. The application checks the image first, then the model you choose reports whether the
              leaf looks healthy or shows Alternaria leaf spot, with a confidence score. The result is a screening aid, not a confirmed
              diagnosis.
            </p>
          </div>
          <div className="lg:col-span-4 lg:justify-self-end">
            <SpecimenLabel className="lg:justify-end">
              <span className="inline-flex items-center gap-1.5">
                <Dot tone={isLoading ? "ochre" : "forest"} /> Backend: FastAPI · CPU
              </span>
              <span>·</span>
              <span>Mode: {selectedModel === "all" ? "All four models" : "Single model"}</span>
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
                          <span className="inline-flex items-center gap-1.5 text-forest-800"><Dot tone="forest" /> being analysed</span>
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
                    <div className="text-xs text-ink-3">No photograph at hand? Try one of these test photographs.</div>
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
                  All Models loads about 3 GB of VGG weights the first time it runs; the first analysis can take 10 to 30 seconds. Later analyses are fast.
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
                  Checking with {selectedOption.name}…
                </>
              ) : (
                <>
                  Check leaf with {selectedOption.name}
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>

          </div>

          {/* ============================================================ RIGHT: READOUT */}
          <div className="lg:col-span-7 lg:sticky lg:top-24">
            <section ref={readoutRef} className="glass border rounded-xl shadow-lift min-h-[520px] flex flex-col scroll-mt-24" aria-busy={isLoading}>
              <div className="flex items-center justify-between px-5 sm:px-6 pt-4 pb-3 border-b border-line">
                <h2 className="text-lg text-ink flex items-baseline gap-2"><span className="text-sm text-ink-4 tabular">03</span>Result</h2>
                <SpecimenLabel>
                  {isLoading ? (
                    <span className="inline-flex items-center gap-1.5 text-forest-800"><span className="w-1.5 h-1.5 rounded-full bg-forest-600 pulse-soft" /> scanning · {(elapsedMs / 1000).toFixed(1)} s</span>
                  ) : result ? (
                    <span className="inline-flex items-center gap-1.5"><Dot tone={result.status === "success" ? "forest" : "ochre"} /> {result.status.replace(/_/g, " ")} · {(elapsedMs / 1000).toFixed(1)} s</span>
                  ) : apiError ? (
                    <span className="inline-flex items-center gap-1.5"><Dot tone="rust" /> error</span>
                  ) : (
                    <span className="inline-flex items-center gap-1.5"><Dot tone="sage" /> idle</span>
                  )}
                </SpecimenLabel>
              </div>

              <div className="p-5 sm:p-6 flex-1 flex flex-col">
                {/* Error */}
                {apiError && (
                  <div className="bg-rust-50 border border-rust-100 rounded-md p-4 text-xs text-rust-700 flex items-start gap-2.5 reveal" role="alert">
                    <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                    <div className="space-y-2 flex-1">
                      <div className="font-medium">Analysis error</div>
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
                    <h3 className="text-xl text-ink">{imagePreview ? "Photograph ready" : "Waiting for a photograph"}</h3>
                    <p className="text-[13px] text-ink-3 max-w-sm mt-2 leading-relaxed">
                      {imagePreview
                        ? `Choose a model, or All Models, then press Check leaf. Currently selected: ${selectedOption.name}.`
                        : "Upload or pick a leaf photograph, choose a model, and press Check leaf. The image is checked first; the result appears here."}
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
                      <h3 className="text-lg text-ink mb-3">{selectedModel === "all" ? "Checking with all four models" : `Checking with ${selectedOption.name}`}</h3>
                      <StatusLedger phase={phase} modelNames={runningModelNames} elapsedMs={elapsedMs} />
                    </div>
                  </div>
                )}

                {result && (
                  <div key={`${result.mode ?? result.status}-${resultSeq}`} className="space-y-5">
                    {/* STAGE 1 REJECTION */}
                    {result.status === "rejected_non_leaf" && (
                      <div className="border border-ochre-100 bg-ochre-50/60 rounded-md p-6 space-y-4 reveal">
                        <div className="flex items-center gap-2 text-ochre-700 font-medium text-sm">
                          <ShieldAlert className="w-5 h-5" />
                          Step 1 · This does not look like a leaf
                        </div>
                        <p className="text-sm text-ink-2 leading-relaxed">{result.message}</p>
                        <dl className="grid grid-cols-2 gap-3 text-xs font-mono tabular">
                          <div className="bg-white border border-line rounded-sm p-3">
                            <dt className="text-ink-3">Similarity to apple leaves</dt>
                            <dd className="text-base text-ink mt-1">{result.similarity?.toFixed(4)}</dd>
                          </div>
                          <div className="bg-white border border-line rounded-sm p-3">
                            <dt className="text-ink-3">Needed to pass</dt>
                            <dd className="text-base text-ink mt-1">&ge; 0.3300</dd>
                          </div>
                        </dl>
                        <button type="button" onClick={handleReset} className="press text-xs font-medium text-forest-800 hover:text-forest-900 inline-flex items-center gap-1 cursor-pointer">
                          Try another photograph <ArrowRight className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    )}

                    {/* STAGE 2 REJECTION */}
                    {result.status === "rejected_non_apple" && (
                      <div className="border border-ochre-100 bg-ochre-50/60 rounded-md p-6 space-y-4 reveal">
                        <div className="flex items-center gap-2 text-ochre-700 font-medium text-sm">
                          <XCircle className="w-5 h-5" />
                          Step 2 · This does not look like an apple leaf
                        </div>
                        <p className="text-sm text-ink-2 leading-relaxed">{result.message}</p>
                        <dl className="grid grid-cols-2 gap-3 text-xs">
                          <div className="bg-white border border-line rounded-sm p-3">
                            <dt className="text-ink-3">Looks most like</dt>
                            <dd className="text-sm font-medium text-ink mt-1">{result.predicted_crop}</dd>
                          </div>
                          <div className="bg-white border border-line rounded-sm p-3">
                            <dt className="text-ink-3">Classifier confidence</dt>
                            <dd className="text-sm font-medium text-ink mt-1 font-mono tabular">{((result.crop_confidence ?? 0) * 100).toFixed(1)}%</dd>
                          </div>
                        </dl>
                        <button type="button" onClick={handleReset} className="press text-xs font-medium text-forest-800 hover:text-forest-900 inline-flex items-center gap-1 cursor-pointer">
                          Try another photograph <ArrowRight className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    )}

                    {/* BACKEND-REPORTED ERROR */}
                    {result.status === "error" && (
                      <div className="bg-rust-50 border border-rust-100 rounded-md p-4 text-xs text-rust-700 flex items-start gap-2.5 reveal" role="alert">
                        <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                        <div className="space-y-2 flex-1">
                          <div className="font-medium">{result.stage || "Inference error"}</div>
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
                        {/* Specimen reference + validation telemetry */}
                        <div className="flex items-stretch gap-4 reveal" style={{ ["--delay" as string]: "0ms" }}>
                          {imagePreview && (
                            <div className="relative w-20 h-20 shrink-0 rounded-md overflow-hidden border border-line bg-paper-2">
                              <Image src={imagePreview} alt="Analyzed specimen" fill className="object-cover" unoptimized />
                            </div>
                          )}
                          <div className="flex-1 min-w-0 flex flex-col justify-between gap-2 text-xs border border-line rounded-sm px-3.5 py-2.5 bg-paper">
                            <div className="flex items-center gap-1.5 font-medium">
                              {result.crop_status === "confirmed" ? (
                                <>
                                  <CheckCircle2 className="w-4 h-4 text-forest-700" />
                                  <span className="text-forest-800">Apple leaf confirmed</span>
                                </>
                              ) : (
                                <>
                                  <Info className="w-4 h-4 text-ink-3" />
                                  <span className="text-ink">Image checks passed</span>
                                </>
                              )}
                            </div>
                            <span className="font-mono text-ink-3 tabular truncate">
                              Crop: {result.predicted_crop} ({((result.crop_confidence ?? 0) * 100).toFixed(1)}%) · leaf similarity {result.similarity?.toFixed(3)}
                            </span>
                            {result.crop_notice && result.crop_status !== "confirmed" && (
                              <span className="text-xs text-ink-3 leading-snug line-clamp-2">{result.crop_notice}</span>
                            )}
                          </div>
                        </div>

                        {/* ALL MODELS */}
                        {isAllMode && result.model_results && result.agreement && (
                          <div className="space-y-4">
                            <div
                              className={cx(
                                "rounded-md border p-4 flex items-start gap-3 reveal",
                                result.agreement.unanimous ? "bg-forest-50/70 border-forest-100" : "bg-ochre-50/70 border-ochre-100"
                              )}
                              style={{ ["--delay" as string]: "80ms" }}
                            >
                              {result.agreement.unanimous ? (
                                <CheckCircle2 className="w-5 h-5 shrink-0 text-forest-700 mt-0.5" />
                              ) : (
                                <AlertTriangle className="w-5 h-5 shrink-0 text-ochre-700 mt-0.5" />
                              )}
                              <div className="space-y-1">
                                <div className="font-display text-xl text-ink leading-tight">
                                  {result.agreement.unanimous
                                    ? `All four models agree: ${result.agreement.majority_label}`
                                    : `The models disagree: ${result.agreement.votes.Healthy} say Healthy, ${result.agreement.votes.Alternaria} say Alternaria`}
                                </div>
                                {!result.agreement.unanimous && (
                                  <div className="text-xs text-ink-2">
                                    Majority: <strong>{result.agreement.majority_label}</strong>
                                    {result.agreement.disagreeing_models.length > 0 && <> · Differing: {result.agreement.disagreeing_models.join(", ")}</>}
                                  </div>
                                )}
                                <div className="text-xs text-ink-3 leading-snug">{result.agreement.note}</div>
                              </div>
                            </div>

                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                              {result.model_results.map((r, i) => {
                                const differs = result.agreement?.majority_prediction != null && r.prediction !== result.agreement.majority_prediction;
                                const healthy = r.prediction === "Healthy";
                                const delay = 160 + i * 90;
                                return (
                                  <article
                                    key={r.model_key}
                                    className={cx("rounded-md border p-4 space-y-3 bg-white reveal", differs ? "border-ochre-500/60" : "border-line")}
                                    style={{ ["--delay" as string]: `${delay}ms` }}
                                  >
                                    <div className="flex items-center justify-between gap-2">
                                      <span className="font-display text-lg text-ink">{r.model_name}</span>
                                      {differs && <Badge tone="ochre">Differs from majority</Badge>}
                                    </div>
                                    <div className="flex items-center gap-2.5">
                                      {healthy ? <CheckCircle2 className="w-5 h-5 text-forest-700 shrink-0" /> : <AlertTriangle className="w-5 h-5 text-rust-600 shrink-0" />}
                                      <span className={cx("font-display text-2xl leading-none", healthy ? "text-forest-800" : "text-rust-700")}>{r.label}</span>
                                    </div>
                                    <div className="flex items-baseline justify-between text-[13px]">
                                      <span className="text-ink-3">Confidence score</span>
                                      <ConfidenceValue value={r.confidence} delay={delay} className="font-display text-2xl text-ink" />
                                    </div>
                                    <ProbabilityBar healthy={r.probabilities.Healthy} alternaria={r.probabilities.Alternaria} delay={delay} />
                                    <div className="flex items-center justify-between font-mono text-xs text-ink-3 tabular">
                                      <span>H {(r.probabilities.Healthy * 100).toFixed(1)}% · A {(r.probabilities.Alternaria * 100).toFixed(1)}%</span>
                                      <span>{r.inference_ms.toFixed(1)} ms</span>
                                    </div>
                                  </article>
                                );
                              })}
                            </div>
                            <p className="text-xs text-ink-3 leading-relaxed reveal" style={{ ["--delay" as string]: "560ms" }}>
                              Each card shows one model&apos;s own prediction; nothing is averaged. Time is that model&apos;s forward pass on the server CPU and excludes the image checks. {result.confidence_note}
                            </p>
                          </div>
                        )}

                        {/* SINGLE MODEL / LEGACY CONSENSUS */}
                        {!isAllMode && (
                          <div
                            className={cx("rounded-md border p-6 space-y-4 reveal", isHealthy ? "bg-forest-50/60 border-forest-100" : "bg-rust-50/60 border-rust-100")}
                            style={{ ["--delay" as string]: "80ms" }}
                          >
                            <div className="flex items-center justify-between gap-3">
                              <span className="text-[13px] text-ink-3">{result.mode === "single" ? "Prediction" : "Combined prediction"}</span>
                              {result.confidence !== undefined && (
                                <span className={cx("inline-flex items-baseline gap-1.5 text-[13px]", isHealthy ? "text-forest-800" : "text-rust-700")}>
                                  <ConfidenceValue value={result.confidence} delay={200} className="font-display text-2xl" /> {confidenceLabel}
                                </span>
                              )}
                            </div>
                            <div className="flex items-start gap-4">
                              {isHealthy ? <CheckCircle2 className="w-9 h-9 text-forest-700 shrink-0 mt-1" strokeWidth={1.5} /> : <AlertTriangle className="w-9 h-9 text-rust-600 shrink-0 mt-1" strokeWidth={1.5} />}
                              <div className="min-w-0">
                                <h2 className={cx("text-3xl sm:text-4xl leading-none", isHealthy ? "text-forest-800" : "text-rust-700")}>
                                  {result.final_label || (isHealthy ? "Healthy Leaf" : "Alternaria Leaf Spot")}
                                </h2>
                                {result.mode === "single" && result.model_name && (
                                  <SpecimenLabel className="mt-2">
                                    <span>Model · {result.model_name}</span>
                                    {result.inference_ms !== undefined && (
                                      <>
                                        <span>·</span>
                                        <span>{result.inference_ms.toFixed(1)} ms inference</span>
                                      </>
                                    )}
                                  </SpecimenLabel>
                                )}
                              </div>
                            </div>
                            {result.probabilities && (
                              <div className="space-y-1.5 pt-1">
                                <ProbabilityBar healthy={result.probabilities.Healthy} alternaria={result.probabilities.Alternaria} delay={200} />
                                <div className="flex justify-between font-mono text-xs text-ink-3 tabular">
                                  <span>Healthy {(result.probabilities.Healthy * 100).toFixed(1)}%</span>
                                  <span>Alternaria {(result.probabilities.Alternaria * 100).toFixed(1)}%</span>
                                </div>
                              </div>
                            )}
                            {result.explanation && <p className="text-xs text-ink-2 leading-relaxed border-t border-line/70 pt-3">{result.explanation}</p>}
                            {result.confidence_note && <p className="text-xs text-ink-3 leading-relaxed">{result.confidence_note}</p>}
                          </div>
                        )}

                        {/* Agronomic advisory */}
                        {result.treatment_guidance && (
                          <div className="space-y-2.5 reveal" style={{ ["--delay" as string]: isAllMode ? "640ms" : "260ms" }}>
                            <div className="flex items-baseline gap-2 flex-wrap">
                              <h3 className="text-lg text-ink">General guidance</h3>
                              <span className="text-xs text-ink-3">
                                {isAllMode ? "based on the majority prediction · " : ""}general reference notes, not advice for your orchard
                              </span>
                            </div>
                            <dl className="grid grid-cols-1 sm:grid-cols-3 divide-y sm:divide-y-0 sm:divide-x divide-line border border-line rounded-sm text-[13px] bg-white">
                              <div className="p-4 space-y-1">
                                <dt className="text-ink-3">What this result means</dt>
                                <dd className="text-ink leading-relaxed">{result.treatment_guidance.pathogen}</dd>
                              </div>
                              <div className="p-4 space-y-1">
                                <dt className="text-ink-3">Typical signs</dt>
                                <dd className="text-ink leading-relaxed">{result.treatment_guidance.symptoms}</dd>
                              </div>
                              <div className="p-4 space-y-1">
                                <dt className="text-ink-3">Suggested next step</dt>
                                <dd className="text-ink leading-relaxed">{result.treatment_guidance.recommended_action}</dd>
                              </div>
                            </dl>
                            <p className="text-xs text-ink-3 leading-relaxed">
                              A prediction is an initial screening result. Before deciding on any treatment, confirm the finding with a qualified
                              agricultural expert or your local horticulture extension service.
                            </p>
                          </div>
                        )}

                        <div className="pt-3 border-t border-line flex flex-wrap justify-end gap-2 reveal" style={{ ["--delay" as string]: isAllMode ? "720ms" : "340ms" }}>
                          <button type="button" onClick={handleAnalyze} className="press inline-flex items-center gap-1.5 text-xs font-medium px-3.5 py-2 rounded-md border border-line bg-white text-ink-2 hover:bg-paper-2 cursor-pointer">
                            <RefreshCw className="w-3.5 h-3.5 text-ink-3" /> Run again
                          </button>
                          <button type="button" onClick={handleReset} className="press inline-flex items-center gap-1.5 text-xs font-medium px-3.5 py-2 rounded-md border border-line-2 bg-white text-ink hover:bg-paper-2 cursor-pointer">
                            Check another leaf <ArrowRight className="w-3.5 h-3.5 text-ink-3" />
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </section>
          </div>

          {/* validation notes: after the readout on stacked layouts, under the controls on desktop */}
          <div className="lg:col-span-5">
            <div className="text-[13px] text-ink-3 leading-relaxed space-y-1.5 px-1 lg:-mt-2">
              <div className="flex items-center gap-1.5 text-ink-2 font-medium">
                <ShieldCheck className="w-3.5 h-3.5 text-forest-700" /> What happens to your photograph
              </div>
              <ul className="space-y-1 pl-5 list-disc marker:text-sage-400">
                <li>Step 1 stops images that do not look like a leaf, such as desks, screens or scenery.</li>
                <li>Step 2 checks that the leaf appears to be from an apple tree, using a classifier trained on 36 crops.</li>
                <li>Step 3 runs the model you selected. All Models runs all four on the same prepared image.</li>
                <li>
                  The confidence score is the model&apos;s probability for its predicted class. It is not the probability that the plant is truly
                  diseased, and no result here is a confirmed diagnosis.
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
