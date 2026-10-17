import Image from "next/image";
import { ShieldCheck, Cpu, Database, AlertTriangle, Code2 } from "lucide-react";
import { COMPARISON_METRICS, REAL_SUBSET_METRICS, EVAL_SETS, DATASET_SPLITS } from "@/data/projectData";
import { STAGE_COPY } from "@/data/stageCopy";
import { PageHeader, Panel, Eyebrow, Badge, StatTile, pct, num } from "@/components/ui";
import { PhotoChip, LesionCloseup } from "@/components/visuals";

export default function AboutPage() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 lg:py-14 space-y-12">
      <PageHeader
        eyebrow="About the project"
        title="About Apple Leaf Alternaria Detection"
        lede="A final-year B.Tech project that screens photographs of apple leaves for Alternaria leaf spot. It compares four transfer-learning models on a dataset split so that no leaf appears in both training and testing, and reports results in two clearly separated views."
      />

      {/* Objective with specimen plate */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        <div className="lg:col-span-7 space-y-5">
          <Eyebrow>The problem</Eyebrow>
          <h2 className="text-3xl text-ink leading-tight">Telling healthy apple foliage from Alternaria leaf spot.</h2>
          <div className="text-sm text-ink-2 leading-relaxed space-y-3">
            <p>
              <strong className="text-ink">Alternaria leaf spot</strong> is a fungal disease of apple trees caused by <em>Alternaria alternata</em>. It appears as small,
              round brown spots on the leaves, sometimes with yellowish tissue around them, and spots can merge as the disease advances. Heavily affected leaves may drop
              early, which weakens the tree and reduces fruit yield. Spotting it early helps growers decide when to look closer and seek advice.
            </p>
            <p>
              <strong className="text-ink">Apple Leaf Alternaria Detection</strong> frames this as a two-class image classification task: given a photograph, decide between{" "}
              <strong className="text-ink">healthy apple foliage</strong> (class 0) and <strong className="text-ink">Alternaria leaf spot</strong> (class 1). The project
              builds the application, trains and compares four models for that task, and adds two screening checks so that photographs of non-leaves or other crops are
              rejected rather than mis-scored.
            </p>
            <p>
              All four models use <strong className="text-ink">transfer learning</strong>: a network first trained on ImageNet, a very large general image collection, is
              adapted to recognise the two classes in this project. This gives good accuracy from a moderate number of apple-leaf images and far less training time than
              starting from scratch.
            </p>
          </div>
        </div>
        <figure className="lg:col-span-5 relative aspect-[4/3] rounded-2xl overflow-hidden shadow-lift bg-paper-2">
          <Image src="/about-leaf.jpg" alt="Healthy apple leaf, field photograph" fill sizes="(min-width: 1024px) 40vw, 100vw" className="object-cover" />
          <PhotoChip kind="healthy" className="absolute left-4 bottom-4">Healthy apple leaf · field capture · class 0</PhotoChip>
        </figure>
      </section>

      {/* Dataset */}
      <Panel className="p-6 sm:p-8 space-y-6">
        <div className="flex items-center gap-2">
          <Database className="w-5 h-5 text-forest-700 shrink-0" strokeWidth={1.5} />
          <h2 className="text-2xl text-ink">The dataset and how it was split</h2>
        </div>
        <p className="text-sm text-ink-2 leading-relaxed max-w-3xl">
          The models were trained and evaluated on <strong className="text-ink">{num(DATASET_SPLITS.totalIncluded)} images</strong> of apple leaves. All{" "}
          {num(DATASET_SPLITS.totalImages)} images in the collection were audited first, and {DATASET_SPLITS.excluded.images} with labels that could not be verified were set
          aside and never used. The images were then divided into three parts with distinct roles:
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          <StatTile label="Training images" value={num(DATASET_SPLITS.train.images)} sub={`Used to fit the models · ${num(DATASET_SPLITS.train.groups)} leaf groups · includes ${num(DATASET_SPLITS.train.synthetic)} synthetic (CycleGAN) images, used in training only`} />
          <StatTile label="Validation images" value={num(DATASET_SPLITS.val.images)} sub={`Used only to choose the best training epoch · ${num(DATASET_SPLITS.val.groups)} leaf groups · no synthetic images`} />
          <StatTile
            label="Held-out test images"
            value={num(EVAL_SETS.full.images)}
            tone="forest"
            sub={`Seen once, for the final results · ${num(EVAL_SETS.full.healthy)} healthy, ${num(EVAL_SETS.full.alternaria)} Alternaria · ${num(EVAL_SETS.real.images)} are original photographs`}
          />
        </div>
        <div className="text-[13px] text-ink-3 leading-relaxed border-t border-line pt-4 space-y-2">
          <p>
            <span className="font-medium text-ink-2">Why the split matters.</span> Many images in the collection are copies or variants of the same leaf: exact duplicates,
            re-exports, flipped or cropped versions, synthetic images generated from a real one, and repeated photographs of one leaf. All such variants were grouped
            together and each group was assigned to exactly one of the three parts. Without this step a model could be tested on near-copies of images it trained on,
            which would overstate its accuracy. No synthetic image is in the validation or test parts.
          </p>
          <p>
            <span className="font-medium text-ink-2">Validation versus test.</span> Validation accuracy was checked after every training epoch and used only to decide which
            epoch&apos;s weights to keep. The test images were opened once, after all decisions were final, to produce the results shown on this site.
          </p>
          <p>
            <span className="font-medium text-ink-2">Two views of the test set.</span> The full test set ({num(EVAL_SETS.full.images)} images) includes flipped and cropped
            copies of small web images, which are easier for the models. The {num(EVAL_SETS.real.images)} original photographs within it are reported separately because
            they say more about performance on a real photo. Neither view covers every orchard, camera, lighting condition or disease stage.
          </p>
        </div>
      </Panel>

      {/* Pipeline */}
      <section className="space-y-5">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-forest-700 shrink-0" strokeWidth={1.5} />
          <h2 className="text-2xl text-ink">The three-step screening pipeline</h2>
        </div>
        <p className="text-sm text-ink-2 leading-relaxed max-w-3xl">
          A classifier trained on apple leaves will still produce an answer for a photograph of a desk or a tomato plant. Two screening steps run before the disease model
          so that such images are rejected instead of mis-scored. Each step is a fixed rule applied the same way to every image.
        </p>
        <ol className="grid grid-cols-1 md:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-line border-y border-line">
          {STAGE_COPY.map((s) => (
            <li key={s.number} className="py-6 md:px-6 first:md:pl-0 last:md:pr-0 space-y-2">
              <span className="font-display text-2xl text-ink-4">0{s.number}</span>
              <h3 className="text-xl text-ink">{s.title}</h3>
              <div className="text-[15px] text-forest-800 font-medium">{s.what}</div>
              <p className="text-[15px] text-ink-2 leading-relaxed">{s.why}</p>
              <p className="text-[13px] text-ink-3 leading-relaxed">{s.tech}</p>
            </li>
          ))}
        </ol>
      </section>

      {/* Pathology close-up: what the two classes look like in the field */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
        <LesionCloseup className="lg:col-span-5" />
        <div className="lg:col-span-7 space-y-3">
          <Eyebrow>What a result means</Eyebrow>
          <h2 className="text-2xl text-ink">Two classes, one confidence score.</h2>
          <p className="text-[15px] text-ink-2 leading-relaxed max-w-2xl">
            The disease model outputs a probability for each of the two classes, healthy apple foliage (class 0) and Alternaria leaf spot (class 1). The confidence
            score shown in the workspace is the probability of the predicted class for that image. It is not the probability that the plant is truly diseased, nor a count
            of spots, a severity grade or a map of where the spots are. A result is a screening aid; confirmation should come from a qualified agricultural expert.
          </p>
        </div>
      </section>

      {/* Architectures */}
      <section className="space-y-5">
        <div className="flex items-center gap-2">
          <Cpu className="w-5 h-5 text-forest-700 shrink-0" strokeWidth={1.5} />
          <h2 className="text-2xl text-ink">The four models compared</h2>
        </div>
        <p className="text-sm text-ink-2 leading-relaxed max-w-3xl">
          Four well-known convolutional network architectures were fine-tuned on the same training images for five epochs, with the same preprocessing and the same
          validation-based epoch selection. They span a wide range of size and speed, from the compact MobileNetV2 to the much larger VGG networks.
        </p>
        <div className="divide-y divide-line border-y border-line">
          {COMPARISON_METRICS.map((m) => {
            const real = REAL_SUBSET_METRICS.find((r) => r.id === m.id) || m;
            return (
              <article key={m.id} className="py-6 grid grid-cols-1 md:grid-cols-12 gap-4 md:gap-8 items-start">
                <div className="md:col-span-3">
                  <div className="flex items-center gap-2 flex-wrap">
                    <h3 className="text-2xl text-ink">{m.name}</h3>
                    {m.isRecommended && <Badge tone="forest">Recommended</Badge>}
                  </div>
                  <div className="text-[13px] text-ink-3 mt-1 tabular">fine-tuned on {m.trainedOn} · selected epoch {m.bestEpoch} of 5</div>
                </div>
                <p className="md:col-span-5 text-[15px] text-ink-2 leading-relaxed">{m.architectureDetails}</p>
                <dl className="md:col-span-4 grid grid-cols-2 gap-x-6 gap-y-2 text-[13px] tabular">
                  <div><dt className="text-ink-3">Original photographs</dt><dd className="font-display text-xl text-forest-800">{pct(real.testAccuracy)}</dd></div>
                  <div><dt className="text-ink-3">Full test set</dt><dd className="font-display text-xl text-ink">{pct(m.testAccuracy)}</dd></div>
                  <div><dt className="text-ink-3">Parameters · file size</dt><dd className="text-ink">{(m.totalParams / 1e6).toFixed(2)}M · {m.fileSizeMb.toFixed(1)} MB</dd></div>
                  <div><dt className="text-ink-3">Time per image, CPU</dt><dd className="text-ink">{m.latencyMs.toFixed(1)} ms</dd></div>
                </dl>
              </article>
            );
          })}
        </div>
      </section>

      {/* Stack */}
      <Panel className="p-6 sm:p-8 space-y-5">
        <div className="flex items-center gap-2">
          <Code2 className="w-5 h-5 text-forest-700 shrink-0" strokeWidth={1.5} />
          <h2 className="text-2xl text-ink">How it is built</h2>
        </div>
        <dl className="grid grid-cols-2 lg:grid-cols-4 gap-6 text-[13px]">
          {[
            ["Frontend", "Next.js (App Router)", "TypeScript · Tailwind CSS"],
            ["Inference backend", "FastAPI & Uvicorn", "Python REST API"],
            ["Deep learning", "PyTorch & Torchvision", "timm integration"],
            ["Charts", "Recharts", "Confusion matrices as tables"],
          ].map(([k, v, s]) => (
            <div key={k} className="border-t border-line-2 pt-3">
              <dt className="eyebrow">{k}</dt>
              <dd className="text-ink mt-1.5 font-medium">{v}</dd>
              <dd className="text-ink-3 mt-0.5">{s}</dd>
            </div>
          ))}
        </dl>
      </Panel>

      {/* Limitations */}
      <Panel tone="paper" className="p-6 sm:p-8 space-y-4">
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-ochre-500" strokeWidth={1.5} />
          <h2 className="text-2xl text-ink">Limitations</h2>
        </div>
        <ul className="space-y-3 text-[15px] text-ink-2 leading-relaxed">
          <li className="rule-accent pt-3">
            <strong className="text-ink">Only two classes.</strong> The models were trained to separate healthy apple foliage from Alternaria leaf spot. Other apple
            diseases such as scab, rust or powdery mildew, and damage from insects or nutrient deficiency, are outside this scope and may be labelled as one of the two
            classes.
          </li>
          <li className="rule-accent pt-3">
            <strong className="text-ink">A screening result, not a diagnosis.</strong> The confidence score is the model&apos;s probability for its predicted class on one
            photograph. It does not measure how severe the disease is and is not the probability that the tree is infected. Treatment decisions should follow confirmation
            by a qualified agricultural expert or extension service.
          </li>
          <li className="rule-accent pt-3">
            <strong className="text-ink">The crop check knows 36 crops.</strong> Step 2 can only choose among 36 crop categories. A leaf from a plant outside that list is
            assigned to whichever category looks most similar, so the 60% Apple threshold is a safeguard rather than a guarantee.
          </li>
          <li className="rule-accent pt-3">
            <strong className="text-ink">The dataset has limits.</strong> It mixes orchard photographs, bench photographs on plain backgrounds, small web images and, in
            training only, synthetic images. It does not cover every apple variety, region, camera, season or stage of the disease, so performance on new conditions may
            differ from the reported figures.
          </li>
          <li className="rule-accent pt-3">
            <strong className="text-ink">Photo quality affects the first check.</strong> Strong flash, deep shadow, heavy blur or compression can push a genuine leaf
            photograph below the 0.33 similarity threshold in step 1, in which case the application asks for another photograph rather than guessing.
          </li>
        </ul>
      </Panel>
    </div>
  );
}
