"use client";

import { useState } from "react";
import { METRICS_BY_SET, EVAL_SETS, EPOCH_HISTORIES, EVALUATION_INFO, type EvalSetId } from "@/data/projectData";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { PageHeader, Panel, PanelHeader, Segmented, Badge, Eyebrow, pct, num, cx } from "@/components/ui";

const TOOLTIP_STYLE = { backgroundColor: "#ffffff", borderColor: "#e7e5e0", borderRadius: "4px", fontSize: "12px", fontFamily: "var(--font-mono)" };
const AXIS_TICK = { fontSize: 11, fill: "#68716b" };

export default function PerformanceAnalysisPage() {
  const [selectedModelKey, setSelectedModelKey] = useState<string>("mobilenetv2");
  const [evalSet, setEvalSet] = useState<EvalSetId>("full");
  const set = EVAL_SETS[evalSet];
  const metrics = METRICS_BY_SET[evalSet];
  const realMnv2 = METRICS_BY_SET.real.find((m) => m.id === "mobilenetv2") || METRICS_BY_SET.real[0];
  const maxValTestGap = Math.max(...METRICS_BY_SET.full.map((m) => Math.abs(m.bestValAccuracy - m.testAccuracy))) * 100;

  const currentHistory = EPOCH_HISTORIES[selectedModelKey] || EPOCH_HISTORIES.mobilenetv2;
  const currentModel = metrics.find((m) => m.id === selectedModelKey) || metrics[0];
  const finalEpoch = currentHistory[currentHistory.length - 1];

  const lossData = currentHistory.map((item) => ({ epoch: `Epoch ${item.epoch}`, "Train Loss": item.trainLoss, "Val Loss": item.valLoss }));
  const accData = currentHistory.map((item) => ({
    epoch: `Epoch ${item.epoch}`,
    "Train Acc": Number((item.trainAcc * 100).toFixed(2)),
    "Val Acc": Number((item.valAcc * 100).toFixed(2)),
  }));

  const setOptions = (Object.keys(EVAL_SETS) as EvalSetId[]).map((id) => ({
    id,
    label: (
      <span>
        {EVAL_SETS[id].shortLabel} <span className="text-ink-4">· {num(EVAL_SETS[id].images)}</span>
      </span>
    ),
  }));
  const modelOptions = metrics.map((m) => ({ id: m.id, label: m.name }));

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 lg:py-14 space-y-10">
      <PageHeader
        eyebrow="Performance analysis · where each model is right and wrong"
        title="Performance Analysis"
        lede="For each model: how its test predictions break down into correct and incorrect calls for both classes, and how training progressed over five epochs. Validation figures show how each model's checkpoint was chosen; they are not test results."
        aside={<Segmented options={setOptions} value={evalSet} onChange={setEvalSet} ariaLabel="Evaluation set" />}
      />

      {/* Confusion matrices */}
      <section className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3 border-b border-line pb-3">
          <div>
            <h2 className="text-2xl text-ink">
              Confusion matrices <span className="text-ink-3">· {set.label}, {num(set.images)} images</span>
            </h2>
            <p className="text-xs text-ink-3 mt-1 max-w-3xl leading-relaxed">
              {num(set.healthy)} Healthy and {num(set.alternaria)} Alternaria test images across {num(set.groups)} independent groups. {set.description}
            </p>
          </div>
          <Eyebrow>Rows: true class · Columns: predicted class</Eyebrow>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 divide-y md:divide-y-0 md:divide-x divide-line border-y border-line">
          {metrics.map((m) => {
            const [[tn, fp], [fn, tp]] = m.confusionMatrix;
            const healthyRecall = ((tn / (tn + fp)) * 100).toFixed(2);
            const alternariaRecall = ((tp / (fn + tp)) * 100).toFixed(2);
            const total = tn + fp + fn + tp;
            const shade = (v: number) => {
              const t = v / Math.max(1, total);
              return t > 0.25 ? "bg-forest-100 text-forest-900" : t > 0.01 ? "bg-sage-50 text-ink" : "bg-paper text-ink-2";
            };
            return (
              <article key={m.id} className="py-6 md:px-6 first:md:pl-0 last:md:pr-0 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-2xl text-ink">{m.name}</h3>
                  {m.isRecommended && <Badge tone="forest">Recommended</Badge>}
                </div>

                {/* 2x2 matrix */}
                <table className="w-full border-separate border-spacing-1 text-[13px] tabular -mx-1">
                  <thead>
                    <tr className="text-ink-3">
                      <th className="text-left font-normal pb-1">
                        <span className="sr-only">Actual class</span>
                      </th>
                      <th className="font-normal pb-1">Pred. Healthy</th>
                      <th className="font-normal pb-1">Pred. Alternaria</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <th className="text-left font-normal text-ink-3 pr-1 whitespace-nowrap">Actual Healthy</th>
                      <td className={cx("p-2.5 text-center rounded-sm", shade(tn))}>
                        <span className="font-display text-2xl leading-none">{num(tn)}</span>
                        <span className="block text-[11px] text-ink-3 mt-1">TN</span>
                      </td>
                      <td className={cx("p-2.5 text-center rounded-sm", fp ? "bg-rust-50 text-rust-700" : "bg-paper text-ink-2")}>
                        <span className="font-display text-2xl leading-none">{num(fp)}</span>
                        <span className="block text-[11px] text-ink-3 mt-1">FP</span>
                      </td>
                    </tr>
                    <tr>
                      <th className="text-left font-normal text-ink-3 pr-1 whitespace-nowrap">Actual Alternaria</th>
                      <td className={cx("p-2.5 text-center rounded-sm", fn ? "bg-rust-50 text-rust-700" : "bg-paper text-ink-2")}>
                        <span className="font-display text-2xl leading-none">{num(fn)}</span>
                        <span className="block text-[11px] text-ink-3 mt-1">FN</span>
                      </td>
                      <td className={cx("p-2.5 text-center rounded-sm", shade(tp))}>
                        <span className="font-display text-2xl leading-none">{num(tp)}</span>
                        <span className="block text-[11px] text-ink-3 mt-1">TP</span>
                      </td>
                    </tr>
                  </tbody>
                </table>

                <dl className="space-y-1.5 text-[13px] tabular border-t border-line pt-3">
                  <div className="flex justify-between"><dt className="text-ink-3">Overall accuracy</dt><dd className="text-ink font-semibold">{pct(m.testAccuracy)}</dd></div>
                  <div className="flex justify-between"><dt className="text-ink-3">Healthy specificity</dt><dd className="text-ink">{healthyRecall}%</dd></div>
                  <div className="flex justify-between"><dt className="text-ink-3">Alternaria sensitivity</dt><dd className="text-ink">{alternariaRecall}%</dd></div>
                  <div className="flex justify-between"><dt className="text-ink-3">Total errors</dt><dd className="text-rust-700 font-semibold">{fp + fn} of {num(set.images)}</dd></div>
                </dl>
              </article>
            );
          })}
        </div>
      </section>

      {/* Learning curves */}
      <Panel>
        <PanelHeader
          title="Training & validation learning curves"
          meta={`${EVALUATION_INFO.epochsTrained} epochs · group-aware train/validation splits · validation used for checkpoint selection only`}
          action={<Segmented options={modelOptions} value={selectedModelKey} onChange={setSelectedModelKey} ariaLabel="Model" />}
        />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 p-6">
          <div className="space-y-3">
            <div className="flex items-start justify-between gap-3">
              <h3 className="text-lg text-ink">Cross-entropy loss <span className="text-ink-3">· {currentModel.name}</span></h3>
              <Eyebrow className="whitespace-nowrap">Fig. 3</Eyebrow>
            </div>
            <div className="h-64 w-full min-w-0">
              <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 320, height: 256 }}>
                <LineChart data={lossData} margin={{ top: 10, right: 20, left: -20, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="2 4" stroke="#ebe8e2" />
                  <XAxis dataKey="epoch" tick={AXIS_TICK} axisLine={{ stroke: "#e7e5e0" }} tickLine={false} />
                  <YAxis tick={AXIS_TICK} axisLine={{ stroke: "#e7e5e0" }} tickLine={false} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "10px" }} />
                  <Line type="monotone" dataKey="Train Loss" stroke="#b4543f" strokeWidth={1.75} dot={{ r: 3 }} isAnimationActive={false} />
                  <Line type="monotone" dataKey="Val Loss" stroke="#357055" strokeWidth={1.75} dot={{ r: 3 }} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-ink-3 text-center tabular">
              Epoch {finalEpoch.epoch} final: train loss {finalEpoch.trainLoss.toFixed(4)} · val loss {finalEpoch.valLoss.toFixed(4)} · selected checkpoint: epoch {currentModel.bestEpoch}
            </p>
          </div>

          <div className="space-y-3">
            <div className="flex items-start justify-between gap-3">
              <h3 className="text-lg text-ink">Classification accuracy <span className="text-ink-3">· {currentModel.name}</span></h3>
              <Eyebrow className="whitespace-nowrap">Fig. 4</Eyebrow>
            </div>
            <div className="h-64 w-full min-w-0">
              <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 320, height: 256 }}>
                <LineChart data={accData} margin={{ top: 10, right: 20, left: -20, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="2 4" stroke="#ebe8e2" />
                  <XAxis dataKey="epoch" tick={AXIS_TICK} axisLine={{ stroke: "#e7e5e0" }} tickLine={false} />
                  <YAxis domain={[88, 100]} unit="%" tick={AXIS_TICK} axisLine={{ stroke: "#e7e5e0" }} tickLine={false} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "10px" }} />
                  <Line type="monotone" dataKey="Train Acc" stroke="#95aa92" strokeWidth={1.75} dot={{ r: 3 }} isAnimationActive={false} />
                  <Line type="monotone" dataKey="Val Acc" stroke="#357055" strokeWidth={1.75} dot={{ r: 3 }} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-ink-3 text-center tabular">
              Epoch {finalEpoch.epoch} final: train acc {(finalEpoch.trainAcc * 100).toFixed(2)}% · val acc {(finalEpoch.valAcc * 100).toFixed(2)}% · selected checkpoint val acc {pct(currentModel.bestValAccuracy)} (epoch {currentModel.bestEpoch})
            </p>
          </div>
        </div>
      </Panel>

      {/* Findings */}
      <section className="pt-4">
        <h3 className="text-2xl text-ink mb-6">What the results show</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 md:gap-10 text-[15px] text-ink-2 leading-relaxed">
          <div className="space-y-2 rule-accent pt-4">
            <h4 className="text-xl text-ink">Transfer learning made training fast</h4>
            <p>
              Each model started from weights pre-trained on ImageNet, a large general image collection, and was then fine-tuned on the apple-leaf data. All four passed 95%
              validation accuracy within the first epoch. Training used the Adam optimizer with a learning rate of 1e-4, which was stable for every architecture.
            </p>
          </div>
          <div className="space-y-2 rule-accent pt-4">
            <h4 className="text-xl text-ink">Validation and test were kept apart</h4>
            <p>
              The validation set was used only to pick the best epoch for each model ({METRICS_BY_SET.full.map((m) => `${m.name}: epoch ${m.bestEpoch}`).join(", ")}). The
              test set was never used for any decision. Full-test accuracy stayed within {maxValTestGap.toFixed(1)} percentage points of the selected validation accuracy for
              every model, so the test results are not inflated by selection.
            </p>
          </div>
          <div className="space-y-2 rule-accent pt-4">
            <h4 className="text-xl text-ink">Where the mistakes are</h4>
            <p>
              On the {num(EVAL_SETS.real.images)} original test photographs, MobileNetV2 misread {realMnv2.confusionMatrix[0][1]} of {realMnv2.supportHealthy} healthy leaves
              as diseased and missed {realMnv2.confusionMatrix[1][0]} of {realMnv2.supportAlternaria} leaves with Alternaria leaf spot. On the full test set, most mistakes of
              the three strongest models fall on small, low-resolution web images, while VGG19&apos;s mistakes concentrate on original photographs of healthy leaves.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
