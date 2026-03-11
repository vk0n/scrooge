import type { Metadata } from "next";
import type { ReactNode } from "react";

import Nav from "../components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "Scrooge Control Plane",
  description: "Control and monitoring interface for Scrooge bot"
};

type RootLayoutProps = {
  children: ReactNode;
};

export default function RootLayout({ children }: RootLayoutProps): JSX.Element {
  return (
    <html lang="en">
      <body>
        <main className="app-shell">
          <Nav />
          {children}
        </main>
      </body>
    </html>
  );
}
