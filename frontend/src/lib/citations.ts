export type Part = { kind: 'text'; text: string } | { kind: 'cite'; n: number }

const MARKER = /\[(\d+(?:\s*,\s*\d+)*)\]/g

function numbersIn(group: string, valid: Set<number>): number[] {
  return group
    .split(',')
    .map((s) => Number(s.trim()))
    .filter((n) => valid.has(n))
}

/** Split answer text into plain text and [n] cites. Numbers with no source are dropped. */
export function splitCitations(text: string, valid: Set<number>): Part[] {
  const parts: Part[] = []
  let last = 0
  for (const match of text.matchAll(MARKER)) {
    if (match.index > last) parts.push({ kind: 'text', text: text.slice(last, match.index) })
    for (const n of numbersIn(match[1], valid)) parts.push({ kind: 'cite', n })
    last = match.index + match[0].length
  }
  if (last < text.length) parts.push({ kind: 'text', text: text.slice(last) })
  return parts
}

/** Source numbers the answer actually cites, unique, in first-appearance order. */
export function citedNumbers(text: string, valid: Set<number>): number[] {
  const seen = new Set<number>()
  for (const part of splitCitations(text, valid)) {
    if (part.kind === 'cite') seen.add(part.n)
  }
  return [...seen]
}

/** DOM id of a source row, so a citation badge can scroll to it. */
export const sourceElementId = (turnId: string, n: number) => `source-${turnId}-${n}`
