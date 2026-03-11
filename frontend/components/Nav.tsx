"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { clearAuth, hasSavedAuth } from "../lib/auth";

const primaryLinks: Array<{ href: string; label: string }> = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/chart", label: "Chart" },
  { href: "/logs", label: "Logs" }
];

export default function Nav(): JSX.Element {
  const router = useRouter();
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
            Scrooge Control
          </Link>
          <div className="nav-actions">
            <Link href="/login" className={`nav-link${loginActive ? " active" : ""}`}>
              Login
            </Link>
            {isAuthed ? (
              <button
                type="button"
                onClick={() => {
                  clearAuth();
                  setIsAuthed(false);
                  router.push("/login");
                }}
              >
                Logout
              </button>
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
            {link.label}
          </Link>
        ))}
      </nav>
    </>
  );
}
