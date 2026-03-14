"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { clearAuth, hasSavedAuth } from "../lib/auth";
import { PRIMARY_LINKS } from "../lib/navigation";
import PushNotificationControl from "./PushNotificationControl";

export default function Nav(): JSX.Element {
  const pathname = usePathname();
  const [isAuthed, setIsAuthed] = useState<boolean>(false);

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
          <Link href={isAuthed ? "/dashboard" : "/login"} className="nav-brand">
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
          {isAuthed ? (
            <div className="nav-actions">
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
              <PushNotificationControl variant="nav" />
            </div>
          ) : null}
        </div>
        {isAuthed ? (
          <div className="nav-links">
            {PRIMARY_LINKS.map((link) => (
              <Link key={link.href} href={link.href} className={linkClass(link.href)}>
                {link.label}
              </Link>
            ))}
          </div>
        ) : null}
      </nav>
      {isAuthed ? (
        <nav className="bottom-nav" aria-label="Mobile navigation">
          {PRIMARY_LINKS.map((link) => (
            <Link key={link.href} href={link.href} className={bottomLinkClass(link.href)}>
              {link.mobileLabel ?? link.label}
            </Link>
          ))}
        </nav>
      ) : null}
    </>
  );
}
