/**
 * minieval-verify-adapter / verify-adapter.ts
 *
 * A working implementation of VerifyingMemoryAdapter, backed by a real
 * MiniEval MemoryGate instance over HTTP.
 *
 * This is not a mock. It calls MiniEval's actual /api/check endpoint —
 * the same endpoint used by MiniEval's own dashboard — which runs the
 * real DeBERTa-v3 faithfulness model and returns a real verdict,
 * reason, and evidence_span.
 *
 * Prerequisite to run this for real:
 *   pip install minieval-pro fastapi uvicorn
 *   python dashboard/app.py          (from the minieval-pro repo)
 *   # serves http://localhost:8000
 *
 * Usage:
 *   const adapter = new MiniEvalVerifyAdapter("http://localhost:8000");
 *   const result = await adapter.verify({
 *     scope: { companyId: "acme" },
 *     source: { kind: "issue_comment", companyId: "acme", issueId: "123" },
 *     sourceText: "I am allergic to peanuts.",
 *     candidateText: "The user loves eating peanuts.",
 *   });
 *   // result.verdict === "rejected"
 */

import type {
  VerifyingMemoryAdapter,
  MemoryVerificationRequest,
  MemoryVerificationResult,
  MemoryVerificationVerdict,
} from "./types";

/** The real shape MiniEval's /api/check endpoint actually returns —
 *  confirmed against the live dashboard, not assumed. */
interface MiniEvalCheckResponse {
  verdict: "STORE" | "REJECT" | "REVIEW";
  fact: string;
  source: string;
  faithfulness: number;
  label: string;
  reason: string;
  evidence_span: string | null;
  policy_name: string;
  policy_version: string;
  policy_fingerprint: string;
  error?: string;
}

/** MiniEval's STORE/REJECT/REVIEW maps onto this adapter's
 *  accepted/rejected/uncertain tri-state — same mapping already used
 *  for the Mem0 hook contract, kept consistent here on purpose. */
function toVerdict(raw: MiniEvalCheckResponse["verdict"]): MemoryVerificationVerdict {
  switch (raw) {
    case "STORE":
      return "accepted";
    case "REJECT":
      return "rejected";
    case "REVIEW":
      return "uncertain";
  }
}

export class MiniEvalVerifyAdapter implements VerifyingMemoryAdapter {
  constructor(private readonly baseUrl: string) {}

  async verify(req: MemoryVerificationRequest): Promise<MemoryVerificationResult> {
    const res = await fetch(`${this.baseUrl}/api/check`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: req.sourceText,
        fact: req.candidateText,
        // record: false — this call is a pre-write check, not a
        // decision to persist. The caller decides what happens next;
        // MiniEval shouldn't log a decision about content the caller
        // might reject for reasons unrelated to faithfulness.
        record: false,
      }),
    });

    if (!res.ok) {
      throw new Error(
        `MiniEval verify() call failed: ${res.status} ${res.statusText}`
      );
    }

    const data: MiniEvalCheckResponse = await res.json();

    if (data.error) {
      throw new Error(`MiniEval returned an error: ${data.error}`);
    }

    return {
      verdict: toVerdict(data.verdict),
      reason: data.reason,
      evidenceSpan: data.evidence_span,
      faithfulness: data.faithfulness,
    };
  }
}