import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FinRisk AI | Risk Intelligence",
  description: "Research workspace for credit risk and fraud analytics",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
