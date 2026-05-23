import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merges Tailwind classes safely, resolving conflicts via tailwind-merge.
 * Industry standard pattern for shadcn-based projects.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
