import { useTypewriter } from "../lib/hooks";
export default function Typed({ text }: { text: string }) { return <>{useTypewriter(text)}</>; }
