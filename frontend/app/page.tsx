import Link from "next/link";

export default function HomePage(): JSX.Element {
  return (
    <section className="panel">
      <h1>Scrooge Control Plane</h1>
      <p className="muted">Stage 1 is active. Use placeholder views below.</p>
      <p>
        Open <Link href="/dashboard">Dashboard</Link>.
      </p>
    </section>
  );
}
