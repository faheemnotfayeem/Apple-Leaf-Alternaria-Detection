/* Plain-language description of the three-stage pipeline.
   Each entry pairs an everyday explanation with the exact technical detail an evaluator needs.
   The thresholds and model names mirror the backend configuration and are not changed here. */

export interface StageCopy {
  number: number;
  title: string;
  /** One sentence: what this stage does. */
  what: string;
  /** One sentence: why it matters to the person using the tool. */
  why: string;
  /** Concise technical detail for evaluators. */
  tech: string;
}

export const STAGE_COPY: StageCopy[] = [
  {
    number: 1,
    title: "Does the photo show a leaf?",
    what: "The application first checks whether the uploaded photograph resembles the apple-leaf images it was built to analyse.",
    why: "Photographs of rooms, screens or other objects are stopped here instead of being given a meaningless result.",
    tech: "A MobileNetV2 feature extractor compares the image with an apple-leaf reference embedding; the image passes when the cosine similarity is at least 0.33. This is a similarity screen, not a botanical authentication test.",
  },
  {
    number: 2,
    title: "Is it an apple leaf?",
    what: "Next, the application checks whether the leaf appears to come from an apple tree rather than another crop.",
    why: "The disease models were trained only on apple foliage, so leaves of other plants would produce unreliable results.",
    tech: "A ConvNeXt-Small crop classifier from Hugging Face, trained on 36 crop categories, must rank Apple first with at least 60% confidence. This is an automated screening step, not a guarantee of species identification.",
  },
  {
    number: 3,
    title: "Healthy, or Alternaria leaf spot?",
    what: "Finally, the model you selected decides whether the leaf looks healthy or shows signs of Alternaria leaf spot, and reports a confidence score.",
    why: "This is the result you act on: a prompt to look more closely or seek expert advice, not a confirmed diagnosis.",
    tech: "One of four transfer-learning classifiers (MobileNetV2, ResNet18, VGG16 or VGG19) outputs a probability for each of two classes, healthy apple foliage (class 0) and Alternaria leaf spot (class 1). The confidence score is the probability of the predicted class.",
  },
];
