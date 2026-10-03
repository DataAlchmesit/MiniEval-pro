/**
 * minieval-verify-adapter / types.ts
 *
 * Mirrors the adapter contract proposed in Paperclip's
 * "Memory Service Surface API" design doc (2026-03-17, PAP-1274),
 * plus one new, additive type: MemoryVerificationResult.
 *
 * These interfaces are copied here, not imported, because this is a
 * standalone proof-of-concept against the PUBLISHED DESIGN DOC, not
 * against Paperclip's real internal SDK (no access to that). If/when
 * `@paperclip/plugins-sdk` exports these types for real, this file's
 * local copies should be deleted in favour of the real import — kept
 * deliberately identical in shape and naming so that swap is
 * mechanical, not a rewrite.
 */

// ---- Copied from the design doc, unchanged ----------------------------

export interface MemoryScope {
  companyId: string;
  agentId?: string;
  projectId?: string;
  issueId?: string;
  runId?: string;
  subjectId?: string;
  sessionKey?: string;
  namespace?: string;
}

export interface MemorySourceRef {
  kind:
    | "issue_comment"
    | "issue_document"
    | "issue"
    | "run"
    | "activity"
    | "manual_note"
    | "external_document";
  companyId: string;
  issueId?: string;
  commentId?: string;
  documentKey?: string;
  runId?: string;
  activityId?: string;
  externalRef?: string;
}

export interface MemorySourcePayload {
  text?: string;
  mimeType?: string;
  metadata?: Record<string, unknown>;
  object?: Record<string, unknown>;
}

export interface MemoryRecordHandle {
  providerKey: string;
  providerRecordId: string;
}

export interface MemoryAdapterCapabilities {
  profile?: boolean;
  correction?: boolean;
  multimodal?: boolean;
  providerManagedExtraction?: boolean;
  asyncExtraction?: boolean;
  providerNativeBrowse?: boolean;
  // New, additive — does this adapter support write-time verification?
  // Capability-gated like every other optional surface in the doc, so
  // adapters that don't implement verify() are unaffected.
  verification?: boolean;
}

// ---- New: the verification surface -------------------------------------

/**
 * The content that would be captured or upserted, if a provider is about
 * to write it. This is intentionally narrower than MemoryCaptureRequest /
 * MemoryRecordWriteRequest from the doc — verification only needs the
 * candidate text and what it's meant to be grounded in, not the full
 * request envelope (binding key, hook context, etc.), which the caller
 * already has and doesn't need MiniEval to see.
 */
export interface MemoryVerificationRequest {
  scope: MemoryScope;
  source: MemorySourceRef;
  /** The raw text the candidate fact should be verified against — e.g.
   *  the original issue comment, run transcript excerpt, or document
   *  text the candidate record was extracted from. */
  sourceText: string;
  /** The candidate fact / record text about to be captured or upserted. */
  candidateText: string;
}

export type MemoryVerificationVerdict = "accepted" | "rejected" | "uncertain";

/**
 * Named to match the "accepted / rejected_with_reason / uncertain"
 * tri-state contract already discussed on the Mem0 verification-hook
 * thread (mem0ai/mem0#7283) — kept consistent across both integrations
 * deliberately, not coincidentally, so a caller integrating against
 * both doesn't have to learn two different vocabularies for the same
 * three outcomes.
 */
export interface MemoryVerificationResult {
  verdict: MemoryVerificationVerdict;
  reason: string;
  /** The specific span of sourceText the verdict is actually based on,
   *  when sourceText is long enough to contain more than one claim.
   *  Null when sourceText is short enough that the whole thing was used
   *  directly — mirrors MiniEval's own evidence_span field. */
  evidenceSpan: string | null;
  /** Raw faithfulness score, 0.0-1.0, for callers that want the number
   *  behind the verdict rather than just the tri-state label. */
  faithfulness: number;
}

/**
 * The optional capability-gated surface itself, in the same style as
 * `correct(handle, patch)` from the design doc's "Optional Adapter
 * Surfaces" section.
 */
export interface VerifyingMemoryAdapter {
  verify(req: MemoryVerificationRequest): Promise<MemoryVerificationResult>;
}