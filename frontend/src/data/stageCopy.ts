/* Plain-language description of how a photograph becomes a screening result.
   Each entry pairs an everyday explanation with a concise technical note for evaluators. */

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
    title: "Photograph the leaf",
    what: "Capture a single apple leaf in good, even light, filling most of the frame.",
    why: "A sharp, well-lit image lets the model see the colour and texture changes that mark early lesions. Photographs that do not clearly show an apple leaf are declined rather than scored.",
    tech: "Images are resized to 256 px, centre-cropped to 224 × 224 and normalised with ImageNet statistics, matching the training pipeline.",
  },
  {
    number: 2,
    title: "Analyse for Alternaria leaf spot",
    what: "A fine-tuned convolutional network examines the leaf for the lesion patterns characteristic of Alternaria leaf spot.",
    why: "The models learned from more than twenty thousand labelled apple-leaf images, evaluated on photographs they never saw during training.",
    tech: "Four ImageNet-pretrained architectures (MobileNetV2, ResNet18, VGG16, VGG19) fine-tuned for two classes: healthy foliage and Alternaria leaf spot.",
  },
  {
    number: 3,
    title: "Review the screening result",
    what: "The result states whether the leaf appears healthy or affected, with the model's confidence and guidance on next steps.",
    why: "Use it as an early prompt to inspect the tree more closely or seek expert advice, not as a confirmed diagnosis.",
    tech: "Confidence is the softmax probability of the predicted class. Running all four models side by side shows where they agree.",
  },
];
