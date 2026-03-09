import type { Metadata } from "next";
import type { ReactNode } from "react";

import Nav from "../components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "Scrooge Control Plane",
  description: "Stage 1 scaffold"
};

type RootLayoutProps = {
  children: ReactNode;
};

export default function RootLayout({ children }: RootLayoutProps): JSX.Element {
  return (
    <html lang="en">
      <body>
        <main>
          <Nav />
          {children}
        </main>
      </body>
    </html>
  );
}
