import Link from "next/link";
import { Leaf } from "lucide-react";
import { REAL_SUBSET_METRICS, EVAL_SETS } from "@/data/projectData";

const PAGES = [
  { href: "/", label: "Home" },
  { href: "/detection", label: "Disease Detection" },
  { href: "/model-comparison", label: "Model Comparison" },
  { href: "/performance-analysis", label: "Performance Analysis" },
  { href: "/about", label: "About Project" },
];

export default function Footer() {
  return (
    <footer className="bg-paper-2 mt-20 border-t border-line">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-14">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-10">
          {/* Brand */}
          <div className="md:col-span-5 space-y-4">
            <div className="flex items-center gap-3">
              <span className="w-8 h-8 rounded-full bg-forest-700 text-white flex items-center justify-center">
                <Leaf className="w-4 h-4" strokeWidth={1.75} />
              </span>
              <div className="leading-tight">
                <span className="block font-display text-lg text-ink">AppleAlt</span>
                <span className="block text-xs text-ink-3">AI-Based Apple Leaf Alternaria Detection</span>
              </div>
            </div>
            <p className="text-sm text-ink-2 leading-relaxed max-w-md">
              A final-year B.Tech project that screens photographs of apple leaves for Alternaria leaf spot using four transfer-learning
              models, each evaluated on images kept apart from training.
            </p>
            <p className="text-xs text-ink-3 leading-relaxed max-w-md pt-3 border-t border-line">
              Research prototype. Results are model predictions for screening, not a confirmed diagnosis. Confirm with a qualified
              agricultural expert before any treatment decision.
            </p>
          </div>

          {/* Pages */}
          <div className="md:col-span-3">
            <div className="eyebrow mb-4">Application</div>
            <ul className="space-y-2.5 text-sm">
              {PAGES.map((p) => (
                <li key={p.href}>
                  <Link href={p.href} className="text-ink-2 hover:text-ink transition-colors">
                    {p.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Models */}
          <div className="md:col-span-4">
            <div className="eyebrow mb-4">
              Correct on the {EVAL_SETS.real.images.toLocaleString()} original test photographs
            </div>
            <ul className="space-y-2.5 text-sm font-mono tabular">
              {REAL_SUBSET_METRICS.map((m) => (
                <li key={m.id} className="flex items-center justify-between gap-4 border-b border-line pb-2">
                  <span className="text-ink-2">
                    {m.name}
                    {/* leading space (not margin) so the badge can wrap below the name in narrow columns */}
                    {m.isRecommended && <span className="block sm:inline text-xs text-forest-700"> recommended</span>}
                  </span>
                  {/* md–lg: the column is too narrow for "99.54% · 11.6 ms" on one line, so latency drops below */}
                  <span className="text-ink whitespace-nowrap text-right">
                    {(m.testAccuracy * 100).toFixed(2)}%
                    <span className="text-ink-3 md:block lg:inline">
                      <span className="md:hidden lg:inline"> · </span>
                      {m.latencyMs.toFixed(1)} ms
                    </span>
                  </span>
                </li>
              ))}
            </ul>
            <p className="text-xs text-ink-3 mt-3 leading-relaxed">Time per image on a CPU, averaged over batches of 32.</p>
          </div>
        </div>

        <div className="mt-12 pt-6 border-t border-line flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs text-ink-3">
          <span>&copy; 2026 AppleAlt</span>
          <span>
            Two classes: healthy apple foliage and Alternaria leaf spot &middot; {EVAL_SETS.full.images.toLocaleString()} test images, {EVAL_SETS.real.images.toLocaleString()} original photographs
          </span>
        </div>
      </div>
    </footer>
  );
}
