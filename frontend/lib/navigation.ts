export type PrimaryLink = {
  href: string;
  label: string;
  mobileLabel?: string;
};

export const PRIMARY_LINKS: PrimaryLink[] = [
  { href: "/dashboard", label: "Office" },
  { href: "/chart", label: "Market Map", mobileLabel: "Map" },
  { href: "/logs", label: "Ledger" }
];
