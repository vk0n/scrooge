"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { clearAuth, hasSavedAuth } from "../lib/auth";

const primaryLinks: Array<{ href: string; label: string; mobileLabel?: string }> = [
  { href: "/dashboard", label: "Office" },
  { href: "/chart", label: "Market Map", mobileLabel: "Map" },
  { href: "/logs", label: "Ledger" }
];

export default function Nav(): JSX.Element {
  const pathname = usePathname();
  const [isAuthed, setIsAuthed] = useState<boolean>(false);

  const loginActive = pathname === "/login";

  useEffect(() => {
    setIsAuthed(hasSavedAuth());
  }, [pathname]);

  function linkClass(href: string): string {
    const isActive = pathname === href || pathname.startsWith(`${href}/`);
    return `nav-link${isActive ? " active" : ""}`;
  }

  function bottomLinkClass(href: string): string {
    const isActive = pathname === href || pathname.startsWith(`${href}/`);
    return `bottom-nav-link${isActive ? " active" : ""}`;
  }

  return (
    <>
      <nav className="panel top-nav">
        <div className="nav-header">
          <Link href="/dashboard" className="nav-brand">
            <Image
              src="/brand/png/scrooge-mark-light-64.png"
              alt="Scrooge mark"
              width={44}
              height={44}
              className="nav-brand-mark"
              priority
              unoptimized
            />
            <span>Scrooge Control</span>
          </Link>
          <div className="nav-actions">
            {!isAuthed ? (
              <Link href="/login" className={`nav-link${loginActive ? " active" : ""}`}>
                Enter
              </Link>
            ) : null}
            {isAuthed ? (
              <Link
                href="/login"
                className="nav-link"
                onClick={() => {
                  clearAuth();
                  setIsAuthed(false);
                }}
              >
                Step Out
              </Link>
            ) : null}
          </div>
        </div>
        <div className="nav-links">
          {primaryLinks.map((link) => (
            <Link key={link.href} href={link.href} className={linkClass(link.href)}>
              {link.label}
            </Link>
          ))}
        </div>
      </nav>
      <nav className="bottom-nav" aria-label="Mobile navigation">
        {primaryLinks.map((link) => (
          <Link key={link.href} href={link.href} className={bottomLinkClass(link.href)}>
            {link.mobileLabel ?? link.label}
          </Link>
        ))}
      </nav>
    </>
  );
}
