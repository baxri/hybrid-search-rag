import { expect, test } from 'vitest'
import { citedNumbers, splitCitations } from './citations'

test('each number in a bracket group becomes its own cite, in order', () => {
  const parts = splitCitations('A [1][2] B [1, 3]', new Set([1, 2, 3]))

  expect(parts.filter((p) => p.kind === 'cite').map((p) => p.n)).toEqual([1, 2, 1, 3])
  expect(parts[0]).toEqual({ kind: 'text', text: 'A ' })
})

test('unknown numbers are dropped and an empty bracket group vanishes', () => {
  expect(splitCitations('A [9].', new Set([1]))).toEqual([
    { kind: 'text', text: 'A ' },
    { kind: 'text', text: '.' },
  ])
})

test('cited numbers are unique, in first-appearance order', () => {
  expect(citedNumbers('x [2] y [1] z [2]', new Set([1, 2]))).toEqual([2, 1])
})
