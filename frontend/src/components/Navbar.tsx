"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Leaf, Menu, X, ArrowUpRight } from "lucide-react";
import { useEffect, useState } from "react";
import { cx } from "@/components/ui";

const NAV_LINKS = [
  { href: "/", label: "Home" },
  { href: "/detection", label: "Disease Detection" },
  { href: "/model-comparison", label: "Model Comparison" },
  { href: "/performance-analysis", label: "Performance Analysis" },
  { href: "/about", label: "About Project" },
];

export default function Navbar() {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // close the drawer on Escape (links close it on click)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setMobileMenuOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <header className="sticky top-0 z-50 pt-3 px-3 sm:px-6 lg:px-8">
      {/* floating translucent bar */}
      <div className="max-w-7xl mx-auto glass border rounded-full shadow-lift">
        <div className="flex items-center justify-between h-14 pl-3 pr-2 sm:pl-4 sm:pr-2.5">
          {/* Wordmark */}
          <Link href="/" className="flex items-center gap-2.5" aria-label="Apple Leaf Alternaria Detection, home">
            <span className="w-9 h-9 rounded-full bg-forest-700 text-white flex items-center justify-center">
              <Leaf className="w-4 h-4" strokeWidth={1.75} />
            </span>
            <span className="font-medium text-[15px] text-ink tracking-tight">Apple Leaf Alternaria Detection</span>
          </Link>

          {/* Desktop navigation */}
          <nav className="hidden lg:flex items-center gap-0.5" aria-label="Primary">
            {NAV_LINKS.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  aria-current={isActive ? "page" : undefined}
                  className={cx(
                    "px-3.5 py-2 text-[13px] font-medium rounded-full transition-colors",
                    isActive ? "bg-white text-ink shadow-hair border border-line" : "text-ink-2 hover:text-ink hover:bg-white/60"
                  )}
                >
                  {link.label}
                </Link>
              );
            })}
          </nav>

          <div className="hidden lg:flex items-center">
            <Link
              href="/detection"
              className="press inline-flex items-center gap-1.5 text-[13px] font-medium pl-4 pr-3 py-2 rounded-full bg-forest-700 text-white hover:bg-forest-800 transition-colors"
            >
              Open workspace
              <ArrowUpRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          {/* Mobile toggle */}
          <button
            type="button"
            onClick={() => setMobileMenuOpen((v) => !v)}
            className="lg:hidden w-10 h-10 flex items-center justify-center text-ink-2 hover:text-ink rounded-full hover:bg-white/70"
            aria-label={mobileMenuOpen ? "Close navigation menu" : "Open navigation menu"}
            aria-expanded={mobileMenuOpen}
            aria-controls="mobile-nav"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile drawer */}
      {mobileMenuOpen && (
        <div id="mobile-nav" className="lg:hidden max-w-7xl mx-auto mt-2 glass border rounded-xl shadow-lift px-3 pt-2 pb-3">
          <nav className="flex flex-col gap-0.5" aria-label="Primary mobile">
            {NAV_LINKS.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  aria-current={isActive ? "page" : undefined}
                  onClick={() => setMobileMenuOpen(false)}
                  className={cx(
                    "flex items-center justify-between px-3 py-2.5 rounded-lg text-[15px]",
                    isActive ? "bg-white text-ink font-medium border border-line" : "text-ink-2 hover:bg-white/60"
                  )}
                >
                  {link.label}
                  {isActive && <span className="w-1.5 h-1.5 rounded-full bg-forest-600" aria-hidden />}
                </Link>
              );
            })}
          </nav>
          <Link
            href="/detection"
            onClick={() => setMobileMenuOpen(false)}
            className="mt-2 flex items-center justify-center gap-1.5 text-sm font-medium px-4 py-3 rounded-full bg-forest-700 text-white"
          >
            Open workspace <ArrowUpRight className="w-4 h-4" />
          </Link>
        </div>
      )}
    </header>
  );
}
