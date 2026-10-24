import { useEffect, useState } from 'react'
import type { EvalReport } from '../lib/evalTypes'

export type EvalLoad =
  { status: 'loading' } | { status: 'missing' } | { status: 'ok'; report: EvalReport }

/** The published evaluation (/eval/results.json), or 'missing' when none has been published. */
export function useEvalReport(): EvalLoad {
  const [load, setLoad] = useState<EvalLoad>({ status: 'loading' })

  useEffect(() => {
    let cancelled = false
    fetch('/eval/results.json')
      .then((res) => (res.ok ? (res.json() as Promise<EvalReport>) : null))
      .catch(() => null)
      .then((report) => {
        if (!cancelled) setLoad(report?.summary ? { status: 'ok', report } : { status: 'missing' })
      })
    return () => {
      cancelled = true
    }
  }, [])

  return load
}
