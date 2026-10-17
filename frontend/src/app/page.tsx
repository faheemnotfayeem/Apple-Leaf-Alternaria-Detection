import Link from "next/link";
import Image from "next/image";
import { ArrowRight, ArrowUpRight } from "lucide-react";
import { COMPARISON_METRICS, REAL_SUBSET_METRICS, EVAL_SETS } from "@/data/projectData";
import { STAGE_COPY } from "@/data/stageCopy";
import { Eyebrow, Button, Badge, Panel, pct, num, cx } from "@/components/ui";
import { LeafEditorial, PhotoBand, PhotoChip } from "@/components/visuals";

const STAGE_VISUALS = [
  { image: "/leaves/stage-foliage.jpg", alt: "Apple foliage on a branch" },
  { image: "/leaves/stage-apple.jpg", alt: "Apple leaf lamina" },
  { image: "/leaves/stage-alternaria.jpg", alt: "Apple leaf with Alternaria lesions" },
];

export default function HomePage() {
  const recommendedModel = COMPARISON_METRICS.find((m) => m.isRecommended) || COMPARISON_METRICS[0];
  const recommendedReal = REAL_SUBSET_METRICS.find((m) => m.isRecommended) || REAL_SUBSET_METRICS[0];
  const fullBest = COMPARISON_METRICS.reduce((a, b) => (b.testAccuracy > a.testAccuracy ? b : a), COMPARISON_METRICS[0]);
  const slowest = COMPARISON_METRICS.reduce((a, b) => (b.latencyMs > a.latencyMs ? b : a), COMPARISON_METRICS[0]);

  return (
    <div className="pb-8">
      {/* ------------------------------------------------------------ HERO */}
      <section className="px-3 sm:px-6 lg:px-8 pt-5">
        <div className="relative max-w-7xl mx-auto rounded-3xl overflow-hidden shadow-deep min-h-[560px] md:min-h-[640px] lg:min-h-[calc(100vh-6rem)] flex flex-col justify-between">
          <Image src="/hero-leaf.jpg" alt="Healthy apple leaf photographed in an orchard" fill priority sizes="100vw" className="object-cover" />
          {/* soft daylight wash so the panel and figures read against the photograph */}
          <div className="absolute inset-0 bg-gradient-to-r from-white/70 via-white/25 to-transparent" aria-hidden />
          <div className="absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-ink/35 to-transparent" aria-hidden />

          <div className="relative p-4 sm:p-8 lg:p-10 lg:[@media(max-height:820px)]:p-8">
            <div className="glass border rounded-2xl shadow-lift p-6 sm:p-7 lg:p-8 lg:[@media(max-height:820px)]:p-6 max-w-xl md:max-w-md lg:max-w-lg space-y-5 lg:[@media(max-height:820px)]:space-y-4">
              <Eyebrow>Apple leaf health check</Eyebrow>
              <h1 className="text-[2.4rem] sm:text-[3rem] lg:text-[3.1rem] xl:text-[3.3rem] lg:[@media(max-height:820px)]:text-[2.75rem] leading-[1.03] text-ink">
                Is this apple leaf healthy? <span className="text-forest-700">Find out from a photograph.</span>
              </h1>
              <p className="text-base sm:text-[17px] text-ink-2 leading-relaxed">
                Upload a photo of an apple leaf and a trained model tells you whether it looks healthy or shows signs of Alternaria leaf
                spot, with a confidence score. An early screening aid, not a confirmed diagnosis.
              </p>
              <div className="flex flex-wrap items-center gap-x-5 gap-y-3 pt-1">
                <Button href="/detection" size="lg">
                  Check a leaf <ArrowRight className="w-4 h-4" />
                </Button>
                <Link href="/model-comparison" className="text-sm font-medium text-ink hover:text-forest-800 inline-flex items-center gap-1.5">
                  See how accurate it is <ArrowUpRight className="w-4 h-4" />
                </Link>
              </div>
            </div>
          </div>

          <div className="relative p-4 sm:p-8 lg:p-10 lg:[@media(max-height:820px)]:p-8 pt-0 sm:pt-0 lg:pt-0 flex items-end">
            <dl className="float-card rounded-2xl grid grid-cols-1 sm:grid-cols-3 divide-y sm:divide-y-0 sm:divide-x divide-line/80 w-full sm:w-auto">
              {[
                [pct(recommendedReal.testAccuracy), `Correct on original test photographs, ${recommendedModel.name}`],
                [`${recommendedModel.latencyMs.toFixed(1)} ms`, "Per image on a CPU, measured in batches of 32"],
                [num(EVAL_SETS.full.images), `Test images never used in training, ${num(EVAL_SETS.real.images)} of them original photographs`],
              ].map(([v, l]) => (
                <div key={l} className="px-4 py-3 sm:px-6 sm:py-4 lg:[@media(max-height:820px)]:py-3 sm:min-w-[180px] flex items-baseline justify-between gap-3 sm:block">
                  <dd className="font-display text-xl sm:text-[1.75rem] leading-none text-ink tabular shrink-0">{v}</dd>
                  <dt className="text-[11px] sm:text-xs text-ink-3 sm:mt-1.5 leading-snug text-right sm:text-left">{l}</dt>
                </div>
              ))}
            </dl>
          </div>
          <div className="hidden sm:block absolute right-6 top-6 lg:right-10 lg:top-10">
            <PhotoChip kind="healthy">Healthy apple leaf · field capture</PhotoChip>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------ PIPELINE */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 lg:pt-28">
        <div className="max-w-2xl space-y-4">
          <Eyebrow>How a leaf check works</Eyebrow>
          <h2 className="text-3xl sm:text-[2.25rem] leading-tight text-ink">Three checks before you see a result.</h2>
          <p className="text-[15px] text-ink-2 leading-relaxed">
            Every photograph passes two screening checks before the disease model runs, so a result is only given for images that look like
            apple leaves. Each check is a fixed rule applied the same way to every image.
          </p>
        </div>

        <ol className="mt-12 lg:mt-16 space-y-16 lg:space-y-24">
          {STAGE_COPY.map((stage, i) => {
            const v = STAGE_VISUALS[i];
            const flip = i % 2 === 1;
            return (
              <li key={stage.number} className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-center">
                <figure className={cx("relative aspect-[4/3] sm:aspect-[16/10] lg:aspect-[4/3] rounded-3xl overflow-hidden bg-paper-2 shadow-lift lg:col-span-6", flip && "lg:order-2 lg:col-start-7")}>
                  <Image src={v.image} alt={v.alt} fill sizes="(min-width: 1024px) 50vw, 100vw" className="object-cover" />
                  <span className="absolute left-4 top-4 float-card rounded-full px-3 py-1 text-xs font-medium text-ink">Step {stage.number} of 3</span>
                </figure>
                <div className={cx("lg:col-span-5", flip ? "lg:order-1 lg:col-start-1" : "lg:col-start-8")}>
                  <div className="font-display text-6xl sm:text-7xl leading-none text-sage-300 tabular">0{stage.number}</div>
                  <h3 className="mt-5 text-2xl sm:text-[1.75rem] leading-snug text-ink">{stage.title}</h3>
                  <p className="mt-2 text-[15px] font-medium text-forest-800">{stage.what}</p>
                  <p className="mt-4 text-base text-ink-2 leading-relaxed">{stage.why}</p>
                  <p className="mt-3 text-[13px] text-ink-3 leading-relaxed">{stage.tech}</p>
                </div>
              </li>
            );
          })}
        </ol>
      </section>

      {/* ------------------------------------------------------------ IMAGERY */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 lg:pt-28">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-8">
          <div className="space-y-4 max-w-2xl">
            <Eyebrow>What to look for</Eyebrow>
            <h2 className="text-3xl sm:text-[2.25rem] leading-tight text-ink">Healthy apple foliage and Alternaria leaf spot.</h2>
          </div>
          <p className="text-[15px] text-ink-2 leading-relaxed max-w-md sm:text-right">
            Alternaria leaf spot often begins as small, round brown spots. The tissue around them may turn yellowish, and spots can merge as the
            disease progresses. Appearance varies with disease stage, leaf age and photo quality, and other problems can also cause brown
            spots, so a photograph alone cannot confirm the disease.
          </p>
        </div>
        <LeafEditorial />
        <p className="text-[13px] text-ink-3 mt-4">Reference photographs from the project archive, shown to illustrate the symptoms. They are not model predictions.</p>
      </section>

      {/* ------------------------------------------------------------ EVIDENCE */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 lg:pt-28">
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 border-b border-line pb-6">
          <div className="space-y-4 max-w-2xl">
            <Eyebrow>How well the models performed</Eyebrow>
            <h2 className="text-3xl sm:text-[2.1rem] leading-tight text-ink">Four models, tested once on photographs they had never seen.</h2>
            <p className="text-sm text-ink-2 leading-relaxed">
              The test set holds {num(EVAL_SETS.full.images)} images kept apart from training. {num(EVAL_SETS.real.images)} of them are original
              photographs; the rest are flipped or cropped copies of small web images, which are easier for the models. Both views are reported
              separately. The original photographs say most about how a model handles a real photo, but neither view covers every orchard,
              camera, lighting condition or stage of the disease.
            </p>
          </div>
          <Link href="/model-comparison" className="text-sm font-medium text-forest-800 hover:text-forest-900 inline-flex items-center gap-1.5 shrink-0">
            Full results and metric definitions <ArrowUpRight className="w-4 h-4" />
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 divide-y md:divide-y-0 md:divide-x divide-line border-b border-line">
          {COMPARISON_METRICS.map((model) => {
            const real = REAL_SUBSET_METRICS.find((r) => r.id === model.id) || model;
            return (
              <article key={model.id} className="py-8 md:px-7 first:md:pl-0 last:md:pr-0 flex flex-col gap-5">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h3 className="text-2xl text-ink">{model.name}</h3>
                    <div className="text-[13px] text-ink-3 mt-1 tabular">
                      {(model.totalParams / 1e6).toFixed(2)}M params · {model.fileSizeMb.toFixed(1)} MB
                    </div>
                  </div>
                  {model.isRecommended && <Badge tone="forest">Recommended</Badge>}
                </div>

                <div>
                  <div className={cx("font-display text-[2.6rem] leading-none tabular", model.isRecommended ? "text-forest-800" : "text-ink")}>{pct(real.testAccuracy)}</div>
                  <div className="text-[13px] text-ink-3 mt-2">correct on the {num(EVAL_SETS.real.images)} original test photographs</div>
                </div>

                <dl className="grid grid-cols-3 gap-3 text-[13px] border-t border-line pt-4">
                  <div><dt className="text-ink-3">Full test set</dt><dd className="text-ink tabular mt-0.5">{pct(model.testAccuracy)}</dd></div>
                  <div><dt className="text-ink-3">Macro F1, original</dt><dd className="text-ink tabular mt-0.5">{pct(real.f1Macro)}</dd></div>
                  <div><dt className="text-ink-3">Time per image</dt><dd className="text-ink tabular mt-0.5">{model.latencyMs.toFixed(1)} ms</dd></div>
                </dl>

                <p className="mt-auto text-[13px] text-ink-3 leading-snug">{model.architectureDetails}</p>
              </article>
            );
          })}
        </div>

        <Panel tone="paper" className="mt-6 p-5 sm:p-6 grid grid-cols-1 md:grid-cols-12 gap-5 items-center">
          <div className="md:col-span-9 text-sm text-ink-2 leading-relaxed">
            <span className="font-medium text-ink">Which model to use. </span>
            <span className="text-forest-800 font-medium">{recommendedModel.name}</span> is the practical choice: it was correct on{" "}
            {pct(recommendedReal.testAccuracy)} of the original test photographs, runs fastest ({recommendedModel.latencyMs.toFixed(1)} ms per image on a CPU)
            and is the smallest file ({recommendedModel.fileSizeMb.toFixed(1)} MB), which matters on ordinary computers. {fullBest.name} scored highest on the full
            test set ({pct(fullBest.testAccuracy)}), while VGG16 and VGG19 need far more storage and time for similar or lower accuracy. No single model is best in
            every situation, so all four can be run and compared in the workspace.
          </div>
          <div className="md:col-span-3 md:justify-self-end">
            <Button href="/detection" variant="primary">
              Check a leaf with {recommendedModel.name} <ArrowRight className="w-4 h-4" />
            </Button>
          </div>
        </Panel>
        <p className="text-xs text-ink-3 mt-3">
          Timing is per image, averaged over batches of 32 on a CPU, from {recommendedModel.latencyMs.toFixed(1)} ms ({recommendedModel.name}) to{" "}
          {slowest.latencyMs.toFixed(1)} ms ({slowest.name}). A single photograph on your own device may take longer.
        </p>
      </section>

      {/* ------------------------------------------------------------ CTA */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 lg:pt-28">
        <PhotoBand>
          <div className="glass border rounded-2xl shadow-lift p-6 sm:p-8 max-w-xl space-y-4">
            <h2 className="text-3xl sm:text-4xl text-ink leading-tight">Check a leaf in four steps.</h2>
            <ol className="text-[15px] text-ink-2 leading-relaxed space-y-1.5 list-decimal pl-5 marker:text-forest-700 marker:font-medium">
              <li>Upload a clear photograph of a single apple leaf.</li>
              <li>The application checks the image, then analyses it.</li>
              <li>Read the predicted class and the model&apos;s confidence score.</li>
              <li>If you wish, run all four models and compare their predictions.</li>
            </ol>
            <p className="text-[13px] text-ink-3 leading-relaxed">
              A prediction is an initial screening result, not a confirmed field diagnosis. If it suggests disease, confirm with a qualified
              agricultural expert or your local horticulture extension service before deciding on any treatment.
            </p>
            <div className="flex flex-wrap gap-3 pt-1">
              <Button href="/detection" size="lg">
                Start a leaf check <ArrowRight className="w-4 h-4" />
              </Button>
              <Button href="/performance-analysis" variant="secondary" size="lg">
                See where the models err
              </Button>
            </div>
          </div>
        </PhotoBand>
      </section>
    </div>
  );
}
