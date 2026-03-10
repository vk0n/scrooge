"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";

import { clearAuth, hasSavedAuth } from "../lib/auth";

const links: Array<{ href: string; label: string }> = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/chart", label: "Chart" },
  { href: "/logs", label: "Logs" },
  { href: "/config", label: "Config" },
  { href: "/controls", label: "Controls" },
  { href: "/login", label: "Login" }
];

export default function Nav(): JSX.Element {
  const router = useRouter();

  return (
    <nav className="panel" style={{ marginBottom: "1rem", display: "flex", gap: "1rem", alignItems: "center" }}>
      {links.map((link) => (
        <Link key={link.href} href={link.href}>
          {link.label}
        </Link>
      ))}
      {hasSavedAuth() ? (
        <button
          type="button"
          onClick={() => {
            clearAuth();
            router.push("/login");
          }}
          style={{ marginLeft: "auto" }}
        >
          Logout
        </button>
      ) : null}
    </nav>
  );
}
