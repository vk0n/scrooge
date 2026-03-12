"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";

import { hasSavedAuth } from "../lib/auth";

type AuthGateProps = {
  children: ReactNode;
};

export default function AuthGate({ children }: AuthGateProps): JSX.Element | null {
  const router = useRouter();
  const [checked, setChecked] = useState<boolean>(false);
  const [isAuthed, setIsAuthed] = useState<boolean>(false);

  useEffect(() => {
    const authed = hasSavedAuth();
    setIsAuthed(authed);
    setChecked(true);
    if (!authed) {
      router.replace("/login");
    }
  }, [router]);

  if (!checked || !isAuthed) {
    return null;
  }

  return <>{children}</>;
}
