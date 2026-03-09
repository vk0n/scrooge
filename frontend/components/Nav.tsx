import Link from "next/link";

const links: Array<{ href: string; label: string }> = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/logs", label: "Logs" },
  { href: "/config", label: "Config" },
  { href: "/controls", label: "Controls" }
];

export default function Nav(): JSX.Element {
  return (
    <nav className="panel" style={{ marginBottom: "1rem", display: "flex", gap: "1rem" }}>
      {links.map((link) => (
        <Link key={link.href} href={link.href}>
          {link.label}
        </Link>
      ))}
    </nav>
  );
}
