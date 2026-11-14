/* Compact reference notes shown under a screening result: key signs, context and next steps.
   General guidance only; product choice and rates must follow local registration and extension advice. */

import type { LucideIcon } from "lucide-react";
import { Bug, CalendarCheck, Camera, CircleCheck, CircleDot, Layers, Leaf, Microscope, ScanSearch, Scissors, ShieldCheck, SprayCan, Target } from "lucide-react";

export type InterpretationKey = "alternaria" | "healthy" | "split";

export interface Interpretation {
  signsTitle: string;
  signs: { icon: LucideIcon; text: string }[];
  factorsTitle: string;
  factors: string[];
  /** Conditions that can resemble the predicted class. */
  lookalikes?: string[];
  steps: { icon: LucideIcon; title: string; text: string }[];
  footnote: string;
}

export const INTERPRETATION: Record<InterpretationKey, Interpretation> = {
  alternaria: {
    signsTitle: "Key signs",
    signs: [
      { icon: CircleDot, text: "Round brown spots, 2–5 mm" },
      { icon: Target, text: "Purplish margin, faint rings" },
      { icon: Layers, text: "Spots merge into blotches" },
      { icon: Leaf, text: "Yellowing and early leaf drop" },
    ],
    factorsTitle: "Favoured by",
    factors: ["Warm, 25–30 °C", "Prolonged leaf wetness", "Dense canopy", "Mite damage", "Delicious-type cultivars"],
    lookalikes: ["Marssonina blotch", "Frog-eye leaf spot", "Apple scab", "Mite or spray injury"],
    steps: [
      { icon: Microscope, title: "Confirm", text: "Send leaves to a diagnostic lab or extension service." },
      { icon: ScanSearch, title: "Survey", text: "Check 10–20 leaves on several trees." },
      { icon: Scissors, title: "Sanitise", text: "Remove fallen leaves; prune for airflow." },
      { icon: SprayCan, title: "Protect", text: "Registered fungicide in rotation, e.g. mancozeb." },
    ],
    footnote: "Follow your local extension spray schedule for products, rates and timing.",
  },

  healthy: {
    signsTitle: "Observed",
    signs: [
      { icon: Leaf, text: "Uniform green colour" },
      { icon: CircleCheck, text: "No spots or lesions" },
      { icon: Layers, text: "Normal venation and shape" },
    ],
    factorsTitle: "Keep in mind",
    factors: ["Covers this leaf only", "Early infection may not be visible", "Other disorders out of scope"],
    steps: [
      { icon: CalendarCheck, title: "Scout", text: "Check leaves every 1–2 weeks in season." },
      { icon: Scissors, title: "Hygiene", text: "Clear fallen leaves; keep the canopy open." },
      { icon: ShieldCheck, title: "Prevent", text: "Follow the local preventive spray schedule." },
      { icon: Camera, title: "Re-check", text: "Analyse any new spots that appear." },
    ],
    footnote: "One healthy leaf cannot rule out disease elsewhere on the tree.",
  },

  split: {
    signsTitle: "Possible causes",
    signs: [
      { icon: Camera, text: "Blur, glare or deep shadow" },
      { icon: CircleDot, text: "Early or atypical lesions" },
      { icon: Bug, text: "Another disease or pest" },
    ],
    factorsTitle: "Keep in mind",
    factors: ["Models disagree on this photograph", "Each model's result is shown separately"],
    steps: [
      { icon: Camera, title: "Retake", text: "Even daylight, leaf filling the frame." },
      { icon: Leaf, title: "Sample more", text: "Analyse several leaves from the tree." },
      { icon: Microscope, title: "Consult", text: "Ask a diagnostic lab or extension service." },
    ],
    footnote: "An inconclusive result is not evidence that the leaf is healthy.",
  },
};
