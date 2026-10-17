import Image from "next/image";
import type { ReactNode } from "react";
import { cx } from "@/components/ui";

/* ------------------------------------------------------------------
   Photographic building blocks. Every image is a real apple leaf from
   the project's archive, exported to /public/leaves under generic
   names. Nothing drawn over a photograph is a model output.
   ------------------------------------------------------------------ */

type LeafClass = "healthy" | "alternaria";

export function PhotoChip({ kind, children, className }: { kind?: LeafClass; children: ReactNode; className?: string }) {
  return (
    <span className={cx("float-card rounded-full px-3.5 py-1.5 text-xs font-medium text-ink inline-flex items-center gap-2", className)}>
      {kind && <span className={cx("w-1.5 h-1.5 rounded-full", kind === "healthy" ? "bg-forest-600" : "bg-rust-600")} aria-hidden />}
      {children}
    </span>
  );
}

function Caption({ kind, title, detail }: { kind: LeafClass; title: string; detail: string }) {
  return (
    <figcaption className="absolute inset-x-0 bottom-0 p-4 sm:p-5 bg-gradient-to-t from-ink/60 via-ink/20 to-transparent">
      <div className="flex items-center gap-2 text-white text-sm sm:text-[15px] font-medium">
        <span className={cx("w-1.5 h-1.5 rounded-full", kind === "healthy" ? "bg-forest-500" : "bg-rust-600")} aria-hidden />
        {title}
      </div>
      <div className="text-white/75 text-xs sm:text-[13px] mt-0.5 pl-3.5">{detail}</div>
    </figcaption>
  );
}

/** Editorial comparison: one large healthy photograph beside two stages of Alternaria leaf spot. */
export function LeafEditorial({ className }: { className?: string }) {
  return (
    <div className={cx("grid grid-cols-1 md:grid-cols-12 gap-4 lg:gap-5", className)}>
      <figure className="md:col-span-7 relative aspect-[4/3] md:aspect-auto md:min-h-[460px] rounded-2xl overflow-hidden bg-paper-2 shadow-lift">
        <Image src="/leaves/healthy-orchard.jpg" alt="Healthy apple leaves beside a developing apple in an orchard" fill sizes="(min-width: 768px) 58vw, 100vw" className="object-cover" />
        <Caption kind="healthy" title="Healthy apple foliage" detail="Even green colour, no spots, leaf edge intact · orchard photograph" />
      </figure>
      <div className="md:col-span-5 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-1 gap-4 lg:gap-5">
        <figure className="relative aspect-[4/3] md:aspect-[16/10] rounded-2xl overflow-hidden bg-paper-2 shadow-lift">
          <Image src="/leaves/alternaria-early.jpg" alt="Apple leaf with scattered dark Alternaria lesions" fill sizes="(min-width: 768px) 40vw, 100vw" className="object-cover" />
          <Caption kind="alternaria" title="Alternaria leaf spot, early stage" detail="A few small, round brown spots on the leaf surface" />
        </figure>
        <figure className="relative aspect-[4/3] md:aspect-[16/10] rounded-2xl overflow-hidden bg-paper-2 shadow-lift">
          <Image src="/leaves/alternaria-advanced.jpg" alt="Apple leaf densely covered with Alternaria lesions" fill sizes="(min-width: 768px) 40vw, 100vw" className="object-cover" />
          <Caption kind="alternaria" title="Alternaria leaf spot, advanced stage" detail="Many spots, some merging, with yellowing tissue around them" />
        </figure>
      </div>
    </div>
  );
}

/** Full-width photographic band with a translucent panel for a call to action. */
export function PhotoBand({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cx("relative overflow-hidden rounded-3xl min-h-[340px] flex items-center shadow-lift", className)}>
      <Image src="/leaves/band.jpg" alt="" fill sizes="100vw" className="object-cover" />
      <div className="absolute inset-0 bg-gradient-to-r from-ink/45 via-ink/20 to-transparent" aria-hidden />
      <div className="relative w-full px-5 sm:px-10 py-10">{children}</div>
    </div>
  );
}

/** Close-up of Alternaria lesions used as a quiet visual anchor on information pages. */
export function LesionCloseup({ className }: { className?: string }) {
  return (
    <figure className={cx("relative aspect-[4/3] rounded-2xl overflow-hidden bg-paper-2 shadow-lift", className)}>
      <Image src="/leaves/alternaria-closeup.jpg" alt="Close-up of Alternaria leaf spot lesions on an apple leaf" fill sizes="(min-width: 1024px) 40vw, 100vw" className="object-cover" />
      <Caption kind="alternaria" title="Alternaria leaf spot" detail="Round brown spots with a yellowish ring · field photograph" />
    </figure>
  );
}
