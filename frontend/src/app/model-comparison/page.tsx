"use client";

import { useState } from "react";
import { METRICS_BY_SET, EVAL_SETS, type EvalSetId } from "@/data/projectData";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { Info } from "lucide-react";
import { PageHeader, Panel, PanelHeader, Segmented, Badge, Eyebrow, pct, num, cx } from "@/components/ui";

const FOREST = "#357055";
const SAGE = "#d2ddd0";
const TOOLTIP_STYLE = { backgroundColor: "#ffffff", borderColor: "#e7e5e0", borderRadius: "4px", fontSize: "12px", fontFamily: "var(--font-mono)" };
const AXIS_TICK = { fontSize: 11, fill: "#68716b" };

export default function ModelComparisonPage() {
  const [evalSet, setEvalSet] = useState<EvalSetId>("full");
  const set = EVAL_SETS[evalSet];
  const metrics = METRICS_BY_SET[evalSet];
  const recommended = metrics.find((m) => m.isRecommended) || metrics[0];
  const fullMetrics = METRICS_BY_SET.full;
  const realRecommended = METRICS_BY_SET.real.find((m) => m.isRecommended) || METRICS_BY_SET.real[0];
  const fullRecommended = fullMetrics.find((m) => m.isRecommended) || fullMetrics[0];
  const fullBest = fullMetrics.reduce((a, b) => (b.testAccuracy > a.testAccuracy ? b : a), fullMetrics[0]);
  const vgg16 = fullMetrics.find((m) => m.id === "vgg16") || fullMetrics[0];
  const vgg19 = fullMetrics.find((m) => m.id === "vgg19") || fullMetrics[0];
  const slowest = fullMetrics.reduce((a, b) => (b.latencyMs > a.latencyMs ? b : a), fullMetrics[0]);

  const accuracyChartData = metrics.map((m) => ({ name: m.name, accuracy: Number((m.testAccuracy * 100).toFixed(2)), isRecommended: m.isRecommended }));
  const efficiencyChartData = metrics.map((m) => ({ name: m.name, latency: Number(m.latencyMs.toFixed(2)), size: Number(m.fileSizeMb.toFixed(1)), isRecommended: m.isRecommended }));

  const setOptions = (Object.keys(EVAL_SETS) as EvalSetId[]).map((id) => ({
    id,
    label: (
      <span>
        {EVAL_SETS[id].label} <span className="text-ink-4">· {num(EVAL_SETS[id].images)}</span>
      </span>
    ),
  }));

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 lg:py-14 space-y-10">
      <PageHeader
        eyebrow="Model comparison · final test results"
        title="How the four models compare"
        lede="Each model was tested once, on a CPU, on the same held-out images that were never used for training or model selection. Results are shown separately for two views of the test set and are never mixed."
        aside={<Segmented options={setOptions} value={evalSet} onChange={setEvalSet} ariaLabel="Evaluation set" />}
      />

      {/* Evaluation set explainer */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-12">
        {(Object.keys(EVAL_SETS) as EvalSetId[]).map((id) => {
          const s = EVAL_SETS[id];
          const active = id === evalSet;
          return (
            <div key={id} className={cx("rule-accent pt-4 space-y-2 transition-opacity", !active && "opacity-60")}>
              <div className="flex items-baseline justify-between gap-3">
                <h3 className="text-2xl text-ink">{s.label}</h3>
                {active ? (
                  <Badge tone="forest">Showing</Badge>
                ) : (
                  <button type="button" onClick={() => setEvalSet(id)} className="text-[13px] font-medium text-forest-800 hover:text-forest-900 cursor-pointer">
                    Show this set
                  </button>
                )}
              </div>
              <div className="text-[13px] text-ink-3 tabular">
                {num(s.images)} images · {num(s.groups)} independent groups · {num(s.healthy)} Healthy / {num(s.alternaria)} Alternaria
              </div>
              <p className="text-[15px] text-ink-2 leading-relaxed">{s.description}</p>
            </div>
          );
        })}
      </div>

      {/* Table */}
      <Panel>
        <PanelHeader
          title={`${set.label}: results for all four models`}
          meta={`${num(set.images)} images · ${num(set.groups)} independent leaf groups · one evaluation run`}
        />
        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-xs">
            <thead>
              <tr className="text-ink-3 text-xs border-b border-line">
                <th className="py-3 px-5 font-medium">Architecture</th>
                <th className="py-3 px-3 text-right font-medium">Accuracy</th>
                <th className="py-3 px-3 text-right font-medium">Macro precision</th>
                <th className="py-3 px-3 text-right font-medium">Macro recall</th>
                <th className="py-3 px-3 text-right font-medium">Healthy F1</th>
                <th className="py-3 px-3 text-right font-medium">Alternaria F1</th>
                <th className="py-3 px-3 text-right font-medium">Macro F1</th>
                <th className="py-3 px-3 text-right font-medium">Params</th>
                <th className="py-3 px-3 text-right font-medium">Size</th>
                <th className="py-3 px-3 text-right font-medium">Latency (batch 32)</th>
                <th className="py-3 px-3 text-right font-medium">Throughput</th>
                <th className="py-3 px-5 text-center font-medium">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line tabular">
              {metrics.map((m) => (
                <tr key={m.id} className={cx("transition-colors hover:bg-paper", m.isRecommended && "bg-sage-50/60")}>
                  <td className="py-3.5 px-5 font-medium text-ink">
                    <div className="flex items-center gap-2">
                      <span className="font-display text-lg">{m.name}</span>
                    </div>
                  </td>
                  <td className="py-3.5 px-3 text-right font-mono text-ink font-semibold">{pct(m.testAccuracy)}</td>
                  <td className="py-3.5 px-3 text-right font-mono text-ink-2">{pct(m.precisionMacro)}</td>
                  <td className="py-3.5 px-3 text-right font-mono text-ink-2">{pct(m.recallMacro)}</td>
                  <td className="py-3.5 px-3 text-right font-mono text-ink-2">{pct(m.f1Healthy)}</td>
                  <td className="py-3.5 px-3 text-right font-mono text-ink-2">{pct(m.f1Alternaria)}</td>
                  <td className="py-3.5 px-3 text-right font-mono text-ink font-semibold">{pct(m.f1Macro)}</td>
                  <td className="py-3.5 px-3 text-right font-mono text-ink-3">{(m.totalParams / 1e6).toFixed(2)}M</td>
                  <td className="py-3.5 px-3 text-right font-mono text-ink-3">{m.fileSizeMb.toFixed(1)} MB</td>
                  <td className="py-3.5 px-3 text-right font-mono text-ink">{m.latencyMs.toFixed(2)} ms</td>
                  <td className="py-3.5 px-3 text-right font-mono text-ink-3">{m.throughput.toFixed(1)} img/s</td>
                  <td className="py-3.5 px-5 text-center">
                    {m.isRecommended ? <Badge tone="forest">Recommended</Badge> : <span className="text-xs text-ink-4">Compared</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <dl className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-x-6 gap-y-3 px-5 py-4 border-t border-line text-[13px] leading-relaxed">
          <div><dt className="font-medium text-ink">Accuracy</dt><dd className="text-ink-2">The share of evaluated images the model classified correctly.</dd></div>
          <div><dt className="font-medium text-ink">Precision</dt><dd className="text-ink-2">When the model says a class, how often it is right.</dd></div>
          <div><dt className="font-medium text-ink">Recall</dt><dd className="text-ink-2">How many of the actual examples of a class the model finds.</dd></div>
          <div><dt className="font-medium text-ink">Macro F1</dt><dd className="text-ink-2">A balanced summary of precision and recall across both classes.</dd></div>
        </dl>
      </Panel>

      {/* Charts */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Panel className="p-6 space-y-4">
          <div className="flex items-start justify-between gap-3 border-b border-line pb-3">
            <div>
              <h3 className="text-lg text-ink">{set.label}: accuracy (%)</h3>
              <p className="text-xs text-ink-3 mt-0.5">Share of the {num(set.images)} images classified correctly, with identical preprocessing for every model.</p>
            </div>
            <Eyebrow className="whitespace-nowrap">Fig. 1</Eyebrow>
          </div>
          <div className="h-64 w-full min-w-0">
            <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 320, height: 256 }}>
              <BarChart data={accuracyChartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="2 4" stroke="#ebe8e2" vertical={false} />
                <XAxis dataKey="name" tick={AXIS_TICK} axisLine={{ stroke: "#e7e5e0" }} tickLine={false} />
                <YAxis domain={[94, 100]} tick={AXIS_TICK} axisLine={{ stroke: "#e7e5e0" }} tickLine={false} unit="%" />
                <Tooltip formatter={(val: unknown) => [`${val}%`, "Accuracy"]} contentStyle={TOOLTIP_STYLE} cursor={{ fill: "#f4f2ee" }} />
                <Bar dataKey="accuracy" radius={[2, 2, 0, 0]} isAnimationActive={false}>
                  {accuracyChartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.isRecommended ? FOREST : SAGE} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-ink-3 text-center">
            Dark green = recommended model ({recommended.name}: {pct(recommended.testAccuracy)} on this set) · light green = the other models
          </p>
        </Panel>

        <Panel className="p-6 space-y-4">
          <div className="flex items-start justify-between gap-3 border-b border-line pb-3">
            <div>
              <h3 className="text-lg text-ink">Time per image on a CPU (ms)</h3>
              <p className="text-xs text-ink-3 mt-0.5">Averaged over batches of 32 images at 224×224 pixels. A single photograph on your own device may take longer.</p>
            </div>
            <Eyebrow className="whitespace-nowrap">Fig. 2</Eyebrow>
          </div>
          <div className="h-64 w-full min-w-0">
            <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 320, height: 256 }}>
              <BarChart data={efficiencyChartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="2 4" stroke="#ebe8e2" vertical={false} />
                <XAxis dataKey="name" tick={AXIS_TICK} axisLine={{ stroke: "#e7e5e0" }} tickLine={false} />
                <YAxis domain={[0, 120]} tick={AXIS_TICK} axisLine={{ stroke: "#e7e5e0" }} tickLine={false} unit="ms" />
                <Tooltip formatter={(val: unknown) => [`${val} ms`, "Latency"]} contentStyle={TOOLTIP_STYLE} cursor={{ fill: "#f4f2ee" }} />
                <Bar dataKey="latency" radius={[2, 2, 0, 0]} isAnimationActive={false}>
                  {efficiencyChartData.map((entry, index) => (
                    <Cell key={`cell-lat-${index}`} fill={entry.isRecommended ? FOREST : SAGE} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-ink-3 text-center">
            {fullRecommended.name}: {fullRecommended.latencyMs.toFixed(2)} ms at batch 32 ({(slowest.latencyMs / fullRecommended.latencyMs).toFixed(1)}× faster than {slowest.name} at {slowest.latencyMs.toFixed(2)} ms)
          </p>
        </Panel>
      </section>

      {/* Trade-off analysis */}
      <section className="pt-4">
        <div className="flex items-center gap-2 mb-6">
          <Info className="w-4 h-4 text-forest-700" />
          <h3 className="text-2xl text-ink">Why MobileNetV2 is recommended</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-12 text-[15px] text-ink-2 leading-relaxed">
          <div className="space-y-2 rule-accent pt-4">
            <h4 className="text-xl text-ink">1. Accuracy and size</h4>
            <p>
              MobileNetV2 is built from compact building blocks (inverted residual bottlenecks with depthwise separable convolutions), so it has only{" "}
              <strong className="text-ink">{(fullRecommended.totalParams / 1e6).toFixed(2)} million parameters</strong> and a {fullRecommended.fileSizeMb.toFixed(1)} MB file, about{" "}
              {(vgg16.totalParams / fullRecommended.totalParams).toFixed(0)} times smaller than VGG16 and {(vgg19.totalParams / fullRecommended.totalParams).toFixed(0)} times smaller than VGG19.
              Despite its size it was correct on <strong className="text-ink">{pct(realRecommended.testAccuracy)}</strong> of the {num(EVAL_SETS.real.images)} original test
              photographs (macro F1 {pct(realRecommended.f1Macro)}). On the full test set it reached {pct(fullRecommended.testAccuracy)}, where {fullBest.name} scored
              highest at {pct(fullBest.testAccuracy)}. The full set contains many flipped and cropped copies of small web images, which is why the two views are kept
              separate.
            </p>
          </div>
          <div className="space-y-2 rule-accent pt-4">
            <h4 className="text-xl text-ink">2. Speed and practical use</h4>
            <p>
              A tool used on ordinary computers or phones needs to be quick. MobileNetV2 took <strong className="text-ink">{fullRecommended.latencyMs.toFixed(1)} ms</strong> per image on a
              CPU (averaged over batches of 32, about {fullRecommended.throughput.toFixed(0)} images per second), compared with {vgg16.latencyMs.toFixed(1)} ms for VGG16 and{" "}
              {vgg19.latencyMs.toFixed(1)} ms for VGG19. A single photograph on a particular device may take longer than these figures. Together with its accuracy on original
              photographs and its small file, that makes MobileNetV2 a practical deployment choice, not proof that it is always the most accurate model in a real orchard. The
              three strongest models differ by only a handful of images out of {num(EVAL_SETS.real.images)}, so all four remain available to compare in the workspace.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
