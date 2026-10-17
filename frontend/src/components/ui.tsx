import Link from "next/link";
import type { ReactNode } from "react";
import { Leaf } from "lucide-react";

/* Small, dependency-free primitives shared by every page. */

export function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(" ");
}

/** Sentence-case pill tag with a leaf mark: the section label of this design system. */
export function Eyebrow({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span className={cx("inline-flex items-center gap-1.5 rounded-full bg-sage-100/80 text-forest-800 text-xs font-medium px-3 py-1.5", className)}>
      <Leaf className="w-3 h-3" strokeWidth={2} aria-hidden />
      <span>{children}</span>
    </span>
  );
}

export function PageHeader({
  eyebrow,
  title,
  lede,
  aside,
}: {
  eyebrow: string;
  title: ReactNode;
  lede?: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <header className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-10 items-end border-b border-line pb-8">
      <div className="lg:col-span-8 space-y-4">
        <Eyebrow>{eyebrow}</Eyebrow>
        <h1 className="text-3xl sm:text-[2.6rem] leading-[1.08] text-ink">{title}</h1>
        {lede && <p className="text-base sm:text-lg text-ink-2 leading-relaxed max-w-2xl">{lede}</p>}
      </div>
      {aside && <div className="lg:col-span-4 lg:justify-self-end w-full lg:w-auto">{aside}</div>}
    </header>
  );
}

export function Panel({
  children,
  className,
  tone = "white",
}: {
  children: ReactNode;
  className?: string;
  tone?: "white" | "paper" | "ink";
}) {
  const tones = {
    white: "bg-white border-line shadow-hair",
    paper: "bg-paper-2 border-line",
    ink: "ink-surface border-line",
  } as const;
  return <section className={cx("rounded-xl border", tones[tone], className)}>{children}</section>;
}

export function PanelHeader({
  title,
  meta,
  action,
  className,
}: {
  title: ReactNode;
  meta?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cx("flex flex-col sm:flex-row sm:items-end justify-between gap-3 px-5 sm:px-6 pt-5 pb-4 border-b border-line", className)}>
      <div className="space-y-1">
        <h2 className="text-xl text-ink">{title}</h2>
        {meta && <div className="text-[13px] text-ink-3 tabular">{meta}</div>}
      </div>
      {action}
    </div>
  );
}

type ButtonProps = {
  children: ReactNode;
  href?: string;
  onClick?: () => void;
  type?: "button" | "submit";
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md" | "lg";
  disabled?: boolean;
  className?: string;
  ariaLabel?: string;
};

/** Pill buttons. Primary is the only solid green element on a page. */
export function Button({ children, href, onClick, type = "button", variant = "primary", size = "md", disabled, className, ariaLabel }: ButtonProps) {
  const base = "press inline-flex items-center justify-center gap-2 font-medium tracking-tight transition-colors select-none rounded-full border";
  const sizes = { sm: "text-xs px-3.5 py-2", md: "text-sm px-5 py-2.5", lg: "text-sm sm:text-[15px] px-6 py-3" } as const;
  const variants = {
    primary: "bg-forest-700 border-forest-700 text-white hover:bg-forest-800 hover:border-forest-800 shadow-lift",
    secondary: "bg-white border-line-2 text-ink hover:bg-paper-2",
    ghost: "bg-transparent border-transparent text-ink-2 hover:text-ink hover:bg-paper-2",
    danger: "bg-white border-rust-600/40 text-rust-700 hover:bg-rust-50",
  } as const;
  const cls = cx(base, sizes[size], variants[variant], disabled && "opacity-50 cursor-not-allowed pointer-events-none", className);
  if (href) {
    return (
      <Link href={href} className={cls} aria-label={ariaLabel}>
        {children}
      </Link>
    );
  }
  return (
    <button type={type} onClick={onClick} disabled={disabled} className={cx(cls, !disabled && "cursor-pointer")} aria-label={ariaLabel}>
      {children}
    </button>
  );
}

/** Boxed statistic: large figure, sentence-case label beneath. */
export function StatTile({
  label,
  value,
  sub,
  tone = "default",
  className,
}: {
  label: ReactNode;
  value: ReactNode;
  sub?: ReactNode;
  tone?: "default" | "forest" | "muted";
  className?: string;
}) {
  const tones = {
    default: "bg-white border-line",
    forest: "bg-sage-50 border-sage-100",
    muted: "bg-paper-2 border-line",
  } as const;
  return (
    <div className={cx("rounded-lg border px-4 py-4", tones[tone], className)}>
      <div className={cx("font-display text-2xl sm:text-[1.75rem] leading-none tabular", tone === "forest" ? "text-forest-800" : "text-ink")}>{value}</div>
      <div className="text-[13px] font-medium text-ink mt-2 leading-snug">{label}</div>
      {sub && <div className="text-xs text-ink-3 mt-0.5 leading-snug">{sub}</div>}
    </div>
  );
}

export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: ReactNode;
  tone?: "neutral" | "forest" | "sage" | "ochre" | "rust" | "mono";
  className?: string;
}) {
  const tones = {
    neutral: "bg-paper-2 text-ink-2 border-line",
    forest: "bg-forest-50 text-forest-800 border-forest-100",
    sage: "bg-sage-50 text-forest-800 border-sage-200",
    ochre: "bg-ochre-50 text-ochre-700 border-ochre-100",
    rust: "bg-rust-50 text-rust-700 border-rust-100",
    mono: "bg-white text-ink-3 border-line tabular",
  } as const;
  return (
    <span className={cx("inline-flex items-center gap-1 text-xs font-medium px-2.5 py-0.5 rounded-full border leading-5 whitespace-nowrap", tones[tone], className)}>
      {children}
    </span>
  );
}

/** Pill tabs; the active tab is solid green. */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  ariaLabel,
  className,
}: {
  options: { id: T; label: ReactNode }[];
  value: T;
  onChange: (id: T) => void;
  ariaLabel: string;
  className?: string;
}) {
  return (
    <div role="radiogroup" aria-label={ariaLabel} className={cx("inline-flex flex-wrap sm:flex-nowrap items-stretch bg-paper-2 rounded-2xl sm:rounded-full p-1 gap-1 max-w-full", className)}>
      {options.map((o) => {
        const active = o.id === value;
        return (
          <button
            key={o.id}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(o.id)}
            className={cx(
              "px-3.5 py-1.5 rounded-full text-[13px] font-medium whitespace-nowrap transition-colors cursor-pointer",
              active ? "bg-forest-700 text-white shadow-hair" : "text-ink-2 hover:text-ink hover:bg-white/70"
            )}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

export function MetricBar({ value, tone = "forest", className }: { value: number; tone?: "forest" | "rust" | "sage"; className?: string }) {
  const fill = { forest: "bg-forest-600", rust: "bg-rust-600", sage: "bg-sage-400" }[tone];
  return (
    <div className={cx("h-1.5 w-full bg-paper-3 rounded-full overflow-hidden", className)} aria-hidden>
      <div className={cx("h-full rounded-full", fill)} style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
    </div>
  );
}

/** Quiet caption row (sentence case, tabular figures). */
export function SpecimenLabel({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cx("text-xs text-ink-3 tabular flex flex-wrap items-center gap-x-2.5 gap-y-1", className)}>{children}</div>;
}

export function Dot({ tone = "forest" }: { tone?: "forest" | "rust" | "ochre" | "sage" }) {
  const c = { forest: "bg-forest-600", rust: "bg-rust-600", ochre: "bg-ochre-500", sage: "bg-sage-400" }[tone];
  return <span className={cx("inline-block w-1.5 h-1.5 rounded-full", c)} aria-hidden />;
}

export const pct = (v: number, d = 2) => `${(v * 100).toFixed(d)}%`;
export const num = (v: number) => v.toLocaleString();
