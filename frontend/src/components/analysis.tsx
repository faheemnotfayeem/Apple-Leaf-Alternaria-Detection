"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { Check, Loader2 } from "lucide-react";
import { cx } from "@/components/ui";

/* ------------------------------------------------------------------
   Motion hooks (no synchronous setState inside effects; all updates come from rAF / timers / media-query events)
   ------------------------------------------------------------------ */

/** True when the user prefers reduced motion; updates live if the setting changes. */
export function useReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = () => setReduced(mq.matches);
    const raf = requestAnimationFrame(onChange);
    mq.addEventListener("change", onChange);
    return () => {
      cancelAnimationFrame(raf);
      mq.removeEventListener("change", onChange);
    };
  }, []);
  return reduced;
}

/** Eases a displayed number from 0 to `target` (the real value) over `duration` ms.
 *  Under reduced motion, or when disabled, the real value is shown immediately. The final frame is always exactly `target`. */
export function useCountUp(target: number, enabled: boolean, duration = 900, delay = 0) {
  const reduced = useReducedMotion();
  const animate = enabled && !reduced;
  const [progress, setProgress] = useState(0); // 0..1 eased progress, updated only from rAF callbacks
  useEffect(() => {
    if (!animate) return;
    let raf = 0;
    let start: number | null = null;
    const ease = (t: number) => 1 - Math.pow(1 - t, 3);
    const tick = (now: number) => {
      if (start === null) start = now + delay;
      const t = Math.min(1, Math.max(0, (now - start) / duration));
      setProgress(ease(t));
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    // safety net: if rAF is throttled (background tab, frozen renderer) the exact value still lands
    const settle = window.setTimeout(() => setProgress(1), delay + duration + 80);
    return () => {
      cancelAnimationFrame(raf);
      window.clearTimeout(settle);
    };
  }, [target, animate, duration, delay]);
  return animate ? target * progress : target;
}

/** Real elapsed time (ms) of the current run: starts when `running` turns true for a given `runId`,
 *  ticks every 100 ms (1 s under reduced motion) and holds its last value when the run ends. */
export function useElapsed(running: boolean, runId: number) {
  const reduced = useReducedMotion();
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!running) return;
    const start = performance.now();
    const tick = () => setElapsed(performance.now() - start);
    const raf = requestAnimationFrame(tick);
    const id = window.setInterval(tick, reduced ? 1000 : 100);
    return () => {
      cancelAnimationFrame(raf);
      window.clearInterval(id);
      tick(); // final value at the moment the run ended
    };
  }, [running, runId, reduced]);
  return elapsed;
}

/* ------------------------------------------------------------------
   Request lifecycle ledger (labels describe the frontend request, not model internals)
   ------------------------------------------------------------------ */
export type RequestPhase = "idle" | "prepared" | "awaiting" | "receiving" | "done" | "error";

const PHASES: { id: Exclude<RequestPhase, "idle" | "done" | "error">; label: string; detail: string }[] = [
  { id: "prepared", label: "Preparing the photograph", detail: "file packaged for the server" },
  { id: "awaiting", label: "Checking and analysing", detail: "waiting for the server: image checks, then the disease model" },
  { id: "receiving", label: "Reading the result", detail: "response received" },
];
const ORDER: RequestPhase[] = ["prepared", "awaiting", "receiving", "done"];

export function StatusLedger({ phase, modelNames, elapsedMs }: { phase: RequestPhase; modelNames: string[]; elapsedMs: number }) {
  const idx = ORDER.indexOf(phase);
  return (
    <div className="space-y-3">
      <ol className="space-y-1.5" aria-live="polite">
        {PHASES.map((p) => {
          const pIdx = ORDER.indexOf(p.id);
          const state = idx > pIdx ? "done" : idx === pIdx ? "active" : "pending";
          return (
            <li key={p.id} className="flex items-start gap-2.5 text-[13px]">
              <span className="mt-[3px] w-3.5 h-3.5 shrink-0 flex items-center justify-center">
                {state === "done" ? (
                  <Check className="w-3.5 h-3.5 text-forest-700" strokeWidth={2.5} />
                ) : state === "active" ? (
                  <span className="w-2 h-2 rounded-full bg-forest-600 pulse-soft" />
                ) : (
                  <span className="w-1.5 h-1.5 rounded-full border border-line-2" />
                )}
              </span>
              <span className={cx("leading-snug", state === "pending" ? "text-ink-4" : "text-ink")}>
                <span className={cx("font-medium", state === "active" && "text-forest-800")}>{p.label}</span>
                <span className="block text-xs text-ink-3">{p.detail}</span>
              </span>
            </li>
          );
        })}
      </ol>
      <div className="flex flex-wrap items-center gap-1.5 pt-1 border-t border-line">
        {modelNames.map((n) => (
          <span key={n} className="inline-flex items-center gap-1.5 text-xs px-2 py-1 rounded-sm border border-line bg-white text-ink-2">
            {phase === "done" ? <Check className="w-3 h-3 text-forest-700" /> : <Loader2 className="w-3 h-3 animate-spin text-forest-600" />}
            {n}
          </span>
        ))}
        <span className="ml-auto text-xs text-ink-3 tabular">elapsed {(elapsedMs / 1000).toFixed(1)} s</span>
      </div>
      <p className="text-xs text-ink-4 leading-snug">
        {modelNames.length > 1
          ? "All four models run in one request and are reported together when it returns."
          : "These steps describe the request to the server, not what happens inside the model."}
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------
   Scan viewport — the uploaded image under decorative instrument overlays
   ------------------------------------------------------------------ */
export function ScanViewport({ src, alt, active, compact = false }: { src: string; alt: string; active: boolean; compact?: boolean }) {
  return (
    <div className={cx("relative w-full overflow-hidden rounded-xl border border-line bg-paper-2", compact ? "aspect-[4/3]" : "aspect-[4/3] sm:aspect-[16/10]")}>
      <Image src={src} alt={alt} fill className="object-contain" unoptimized />

      {/* a thin veil while the request is in flight, so the line reads against bright photos */}
      {active && <div className="absolute inset-0 bg-ink/5 pointer-events-none" aria-hidden />}

      {/* precise scanning line while the request is in flight */}
      {active && <div className="scan-beam" aria-hidden />}

      {/* subtle focus frame: four corner marks around the specimen, decorative only */}
      {active && (
        <div className="absolute inset-3 sm:inset-4 pointer-events-none fade" aria-hidden>
          <span className="absolute top-0 left-0 w-5 h-5 border-t-2 border-l-2 border-white/90 rounded-tl-sm" />
          <span className="absolute top-0 right-0 w-5 h-5 border-t-2 border-r-2 border-white/90 rounded-tr-sm" />
          <span className="absolute bottom-0 left-0 w-5 h-5 border-b-2 border-l-2 border-white/90 rounded-bl-sm" />
          <span className="absolute bottom-0 right-0 w-5 h-5 border-b-2 border-r-2 border-white/90 rounded-br-sm" />
        </div>
      )}

      {/* honesty label */}
      <div className="absolute left-2.5 bottom-2.5 right-2.5 flex items-center justify-between pointer-events-none text-[11px] font-medium">
        <span className="text-white bg-ink/60 px-2 py-0.5 rounded-sm">{active ? "Scanning" : "Specimen"}</span>
        <span className="text-white/90 bg-ink/60 px-2 py-0.5 rounded-sm">Decorative animation · no spots are being marked</span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------
   Animated value readouts (real values; only the approach to them is animated)
   ------------------------------------------------------------------ */
export function ConfidenceValue({ value, decimals = 1, delay = 0, className }: { value: number; decimals?: number; delay?: number; className?: string }) {
  const v = useCountUp(value * 100, true, 900, delay);
  return (
    <span className={cx("tabular", className)} aria-label={`${(value * 100).toFixed(decimals)} percent`}>
      {v.toFixed(decimals)}%
    </span>
  );
}

export function ProbabilityBar({ healthy, alternaria, delay = 0 }: { healthy: number; alternaria: number; delay?: number }) {
  const reduced = useReducedMotion();
  const [grown, setGrown] = useState(false);
  useEffect(() => {
    if (reduced) return;
    const id = window.setTimeout(() => setGrown(true), 40 + delay);
    return () => window.clearTimeout(id);
  }, [reduced, delay]);
  const show = reduced || grown;
  return (
    <div className="h-1 w-full rounded-sm overflow-hidden bg-paper-3 flex" title="Healthy vs Alternaria probability" aria-hidden>
      <div className="h-full bg-forest-600 bar-grow" style={{ width: show ? `${healthy * 100}%` : "0%" }} />
      <div className="h-full bg-rust-600 bar-grow" style={{ width: show ? `${alternaria * 100}%` : "0%" }} />
    </div>
  );
}
