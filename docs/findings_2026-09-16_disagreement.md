# Findings — can a structural change make the agents disagree?

**Date**: 2026-09-16
**Runs**: 3 arms, 15 cycles each (one stopped at 5), BASELINE, `qwen2.5:7b-instruct`, local Ollama
**Status**: exploratory, `n=1` per arm. A signal to act on, not a result to publish.

---

## Question

Across every run ever collected — three prompt variants, two model sizes — the
mutual-approval gate has never closed. Before these runs the lifetime record was
**93 doctrine proposals, 93 approvals, 0 rejections**. `PLAN.md` §11 makes that
gate the centre of the design, and the April proposal predicted the failure
exactly (F02 Trivial Agreement, detection rule "CS = 1.0 for 5+ cycles") and
named a mitigation: a devil's-advocate phase.

It is not a prompt problem. The system prompts already say *"When you disagree
with Flux, say so directly"* and call the tension between the agents *"a design
feature, not a bug"*. The vote prompt already names three grounds for rejection.
So: can a **structural** change do what the prompts cannot?

## Design

Three arms, identical but for the intervention. Same task, execution settings,
scenario schedule (cycles 5 and 10, both fired), cycle count, model.

| Arm | Intervention |
|---|---|
| `DA_CONTROL` | None. The current system. |
| `DA_TEST` (v1) | Each **voter** writes the case against, then judges it. |
| `DA_TEST2` (v2) | The **proposer** writes the case against its own proposal; the voter must write both cases before voting. |

`n=1` is defensible for the primary question and would not be for most: the
baseline is a *perfect* record, so telling "none" from "several" is not a
question about sampling noise. It would be if the arms were 12% and 18%.

No seeds — neither backend sends one — so none of this is reproducible. Recorded
rather than pretended away.

## Results

| | control | v1 | v2 |
|---|---|---|---|
| cycles | 15 | 5 (stopped) | 15 |
| proposals | 29 | 10 | 23 |
| approvals | 29 | **0** | 23 |
| **rejections** | **0** | **10** | **0** |
| approved revisions actually applied | 15 / 29 (52%) | 0 | **19 / 23 (83%)** |
| discarded as "identical to current document" | **13** | — | **3** |
| mean `case_against` written per vote | — | — | 385 chars |
| mean evaluation score | 30.9/50 | 28.4/50 | 30.1/50 |
| execution clean | 10/14 | — | 8/15 |

## 1. Neither version produced disagreement

The primary question has a clear negative answer. v2 approved **23 of 23**,
taking the lifetime record to **145 proposals, 145 approvals, 0 rejections**.

v1 did produce rejections — all of them — which is the same failure wearing the
opposite costume. The mechanism was legible in the text. Every rejection opened
with the same three words:

> **flux**: "The objection stands. Adding a separate commitment could dilute…"
> **axiom**: "The objection stands. Introducing a new commitment could dilute…"
> **flux**: "The objection stands. While the criteria provide structure, they…"

Asking a model whether the argument it just made holds up has a compliant
answer. v1 moved *what* the agents agreed with and changed nothing about the
agreeing. Both 0/N and N/N mean the prompt is doing the deciding.

## 2. v2 produced real deliberation, and it changed nothing about the verdict

This is the more interesting half. The mechanism worked exactly as designed —
`DeliberatedVote` orders `case_for` before `case_against` before `vote`, and
structured output is generated in schema order, so the model cannot reach a
verdict without writing both sides. It did, at a mean of 385 characters per
objection, and the objections are substantive:

> **case_against**: "While the structured dialogue format is beneficial, it may
> introduce administrative overhead and slow down the decision-making process…"

That is a real cost, correctly identified. And then it approved. Twenty-three
times out of twenty-three.

So the agents can articulate a genuine objection to a proposal and never weigh
one as decisive. **Trivial agreement is not a failure to see the counter-case.**
That rules out the whole class of interventions aimed at making them think
harder, which is most of what a prompt can do, and is the useful thing this run
bought.

## 3. The unlooked-for result: revisions stopped being no-ops

The control arm discarded **13 of 29 approved revisions because the "revised"
document was byte-identical to the existing one**. Nearly half of its apparent
doctrine evolution was agents voting yes on a change that was not a change.

Under v2 that fell to **3 of 23**, and the share of approved revisions that
actually reached disk went from 52% to 83%.

The mechanism is plausible and mechanical: you cannot write the strongest case
against a document identical to the one it replaces. Forcing the proposer to
argue against its own proposal requires the proposal to contain something to
argue about.

Doctrine evolution is the primary dependent variable in `PLAN.md` §10. If this
holds, the self-critique step is worth keeping for a reason unrelated to why it
was built — not because it creates disagreement, which it does not, but because
it stops proposals being empty.

**Treat as provisional.** Single runs, 29 and 23 observations, no seeds, and the
comparison was not designed to test this. It needs a replication aimed at it.

## 4. Costs

v2 runs one extra inference call per proposal (not per voter, unlike v1). Phase
errors rose from 2 to 4 — all JSON parse failures that exhausted three retries,
consistent with asking a 7b model for a four-field structured vote instead of
two. Execution cleanliness fell slightly (10/14 to 8/15), which on these numbers
is noise.

## Recommendations

1. **Stop trying to induce disagreement through deliberation quality.** §2 rules
   that class out: they see the counter-case and approve anyway. What is left is
   giving the agents genuinely conflicting objectives — asymmetric success
   criteria, so they *want* different things — or accepting that two instances
   of one model in a cooperative frame will not disagree and designing around it.
2. **Keep the self-critique step**, provisionally, for §3 rather than §1.
3. **Replicate §3 directly**: control vs self-critique, measuring the
   identical-document discard rate as the primary outcome, ideally at 3 runs
   per arm.
4. **The `trivial_agreement` watchdog rule now fires during a run** (it would
   have tripped at cycle 5 of every run in project history). Any future arm gets
   an alarm rather than a post-hoc discovery.

## Limitations

`n=1` per arm. 15 cycles. One model, not the study model — `qwen2.5:32b` has
never been run. No seeds, so not reproducible. The v1 arm was stopped at 5
cycles once its outcome was unambiguous. Scenario injections fired in both
full arms (cycles 5 and 10) but their effect is not isolated. The evaluation
score is produced by the same model family as the subjects and 20 of its 50
points are known to measure something other than their name, so the eval column
above is reported for completeness and should not be leaned on.
