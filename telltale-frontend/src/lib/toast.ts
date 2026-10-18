export const toast = (message: string) => window.dispatchEvent(new CustomEvent("toast", { detail: message }));
