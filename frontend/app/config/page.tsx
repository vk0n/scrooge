import { redirect } from "next/navigation";

export default function ConfigPage(): never {
  redirect("/dashboard");
}
