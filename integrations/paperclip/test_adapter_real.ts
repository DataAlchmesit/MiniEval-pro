/**
 * Real end-to-end test against the ACTUAL MiniEval dashboard — real
 * DeBERTa-v3 model, real scoring, no mock. Requires the dashboard
 * running first (see README.md in this folder).
 */
import { MiniEvalVerifyAdapter } from "./verify-adapter";
import type { MemoryVerificationRequest } from "./types";

async function run() {
  const adapter = new MiniEvalVerifyAdapter("http://localhost:8000");

  const cases: Array<[MemoryVerificationRequest, string]> = [
    [
      {
        scope: { companyId: "acme" },
        source: { kind: "issue_comment", companyId: "acme", issueId: "123" },
        sourceText: "I am allergic to peanuts.",
        candidateText: "The user loves eating peanuts.",
      },
      "contradiction case",
    ],
    [
      {
        scope: { companyId: "acme" },
        source: { kind: "run", companyId: "acme", runId: "run-1" },
        sourceText:
          "I moved to Chennai last week. My brother is a lawyer and he helped me with the contract.",
        candidateText: "The user is a lawyer.",
      },
      "misattribution case",
    ],
    [
      {
        scope: { companyId: "acme" },
        source: { kind: "issue_comment", companyId: "acme", issueId: "124" },
        sourceText: "I am allergic to peanuts.",
        candidateText: "The user is allergic to peanuts.",
      },
      "clean accept case",
    ],
  ];

  for (const [req, label] of cases) {
    const result = await adapter.verify(req);
    console.log(`\n[${label}]`);
    console.log(`  candidate: "${req.candidateText}"`);
    console.log(`  verdict:   ${result.verdict}`);
    console.log(`  reason:    ${result.reason}`);
    console.log(`  evidenceSpan: ${result.evidenceSpan}`);
    console.log(`  faithfulness: ${result.faithfulness}`);
  }

  console.log("\nAll calls completed against the REAL MiniEval dashboard — real model, real scores.");
}

run().catch((err) => {
  console.error("FAILED:", err);
});