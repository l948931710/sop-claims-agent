# SOP claims agent

An insurance claims support agent that follows a fixed workflow and still talks like a person.

The workflow is `VERIFY_ID → RESOLVE_INTENT → PROCESS_CASE → POST_PROCESS`. A small state machine (the
harness) owns the phase, the gates, the memory and the tools. The language model reads what the caller said and
writes the reply, with as much freedom as the current phase allows and no more. The model cannot move the
phase, cannot see claim data before the caller is verified, and cannot send anything on its own.

## Run it

### Docker

```bash
docker build -t sop-claims-agent .
```

```bash
docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... sop-claims-agent
```

Then open <http://localhost:8000>.

Three ways to supply the model's API token:

| How | When to use it |
| --- | --- |
| `-e ANTHROPIC_API_KEY=...` on `docker run`, or copy `.env.example` to `.env` and use `docker run --env-file .env ...` or `docker compose up --build` | Normal use. |
| Start with no key, then paste one into **Settings** in the console | Hosted demo where each tester brings a key. It is sent with each message in a request header, kept only in that browser tab, and not stored by the server. |
| No key at all: `-e LLM_PROVIDER=stub`, or pick **Offline stub** in Settings | Checking the harness without a model. Replies are the harness's own scripts and the message reading is keyword rules, so it is stilted, but every gate behaves the same. |

### Without Docker

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

```bash
ANTHROPIC_API_KEY=sk-ant-... .venv/bin/uvicorn app.main:app --port 8000
```

```bash
.venv/bin/python -m pytest
```

Python 3.10 or newer.

## Try it

The console has the chat on the left and the harness's state on the right: the phase, which identity fields
matched, everything remembered and the phase it was heard in, the counters, and a per-turn trace of what was
understood, gated, called and checked. The **Try** row fills the input with sample messages. **Test data**
shows the sample records, so you know what to type.

| Scenario | Type | What should happen |
| --- | --- | --- |
| Demo case | *I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.* | Verified on three fields. The claim hint is stored during `VERIFY_ID` and used straight after: the agent goes to CL-2048 without asking which claim. One message, three phases. |
| Reason before identity | *Hi, this is Margaret Chen. I'm calling about my denied healthcare claim from January.* then *My date of birth is 1985-03-15 and the last four of my SSN are 4472.* | First reply stays in `VERIFY_ID`, says nothing about the claim, and the hint shows in Memory as "heard in VERIFY_ID". Second reply resolves the case from memory. |
| Frustration | *I already told you who I am. This is ridiculous. Just tell me why my claim was denied.* | Acknowledges the frustration first, explains why verification comes first, lists the accepted identifiers. No claim details. Keep refusing and it offers a human on the second refusal and hands off on the third. |
| Alternate fields | Name, then phone and email instead of the SSN | Any three of the five fields verify. |
| Wrong details | A wrong date of birth three times | It says which field did not match, never the value on file, warns before the last attempt, then hands off. |
| Off-topic | *What is RL?* three times | Polite decline and back to the call; the second decline offers a human; the third hands off. An on-topic message in between resets the count. |
| Representative | *I'm David Chen, calling on behalf of my mother Margaret Chen about her denied claim. Her date of birth is 1985-03-15, her phone is 650-521-2836 and her email is margaret@email.com.* then *She says she has approved it.* | Three matching fields are not enough: a consent request goes to the policyholder and the call waits. Approved on the second check. Set the consent simulator to `timeout` in Settings to see it hand off instead. |
| Claim questions | *How do I submit the documents?* · *What if I can't get the pathology report?* · *What about my auto claim?* | Answers come from the claim record and the guidance tools (see the trace). A different claim sends the call back through `RESOLVE_INTENT`. |
| Wrap-up | *That's all, thanks.* then *Yes please.* or *No, skip it.* | The email summary is offered, and sent only on a yes. The sent email appears under Wrap-up. |

## How it works

### One turn

```
caller message
   │
1  understand   the model reads the message into a typed record: identity details, intent,
   │            claim hints, scope, emotion, refusals. No reply, no decisions.
2  remember     everything useful goes to memory, whichever phase the call is in
3  policies     asked for a human? off-topic? upset?                              (code)
4  gate         the current phase's rule runs and may advance the phase           (code)
5  respond      the model writes the reply from a brief, with that phase's tools only
6  guard        claim numbers, amounts, dates and identifiers are checked;
   │            a failing reply is retried once, then replaced by the script      (code)
reply
```

Steps 4 to 6 repeat when the phase changes inside a turn, which is how one message can verify the caller,
resolve the case and answer the question.

The model is used in two narrow roles. The **reader** fills a structured record and never talks to the caller.
The **writer** gets a brief and never decides what happens next. Neither prompt carries the workflow; that is
in `app/sop.py` and `app/engine.py`.

### Different freedom in different phases

Every reply starts from a *directive* the harness builds: the objective, the points to cover, what must not be
said, the only facts the reply may use, the tools on offer, and a plain script the harness wrote itself that
already satisfies the SOP.

| Phase | Latitude | Who decides what happens | Tools the model has | Data the model sees |
| --- | --- | --- | --- | --- |
| `VERIFY_ID` | scripted | The harness. The model chooses wording and tone. | none | Which fields matched. Never the values on file, never claim data. |
| `RESOLVE_INTENT` | guided | The model picks: select a claim and a workflow path, ask a clarifying question, or hand off. | `select_case`, `request_human_handoff` | An index of the verified policyholder's own claims, plus memory. |
| `PROCESS_CASE` | open | The model answers freely. | `get_document_guidance`, `get_followup_guidance`, `switch_case`, `request_human_handoff` | The selected claim record and what the tools return. |
| `POST_PROCESS` | scripted | The harness. Nothing is sent without a yes. | none | The masked email address and the outcome. |

The script is what the caller gets if the model call fails or a reply is rejected, so a conversation never
depends on the model behaving. The offline stub is that fallback made permanent.

### Memory across phases

Step 2 runs before any gate, in every phase. Slots are namespaced (`identity.*`, `intent.*`, `case.*`,
`prefs.*`, notes) and each keeps the turn, the phase it was heard in, and the caller's words. So "my denied
healthcare claim from January" said during verification is stored but not acted on; after verification
`RESOLVE_INTENT` receives it, and a simple matcher also tells the model which claims on file are consistent
with it. An email preference stated early is remembered too: a wish for a summary is confirmed at wrap-up
rather than assumed, and a refusal is honoured without asking again.

### Where each rule is enforced

| Rule | Enforced by | Test |
| --- | --- | --- |
| No claim details before verification | The model is given none: no claim data in the brief, no tools. A guard also rejects any claim number, amount, date or on-file identifier it did not get from the caller. | `test_model_is_given_no_record_data_before_verification`, `test_nothing_from_the_records_is_sent_before_verification`, `test_claim_details_written_before_verification_never_reach_the_caller` |
| Three of five identity fields | `identity.evaluate` compares in code. The model never sees a value on file, only whether each field matched. | `tests/test_identity_and_guards.py` |
| Phase order | `Session.transition` checks a transition table; nothing else changes the phase. | `test_transition_table_is_enforced_for_everyone` |
| Allowed actions | The tool gateway checks the phase on every call, whatever the model was shown. | `test_model_cannot_use_claim_tools_before_verification` |
| Only the caller's own claims | Claims are looked up under the verified party id; another policyholder's claim number is "not on this policy". | `test_model_cannot_select_another_policyholders_claim` |
| Grounded answers | Every claim number, amount and date in a reply must appear in the brief, a tool result from that turn, or the conversation. Text the model wrote into a tool argument does not count. | `test_ungrounded_amount_in_an_open_phase_is_replaced`, `test_text_the_model_wrote_cannot_ground_its_own_reply` |
| Email only with consent | The harness sends, not the model, and only on a recorded yes. The drafter never sees the verification exchange. | `test_unclear_answer_to_the_email_offer_sends_nothing`, `test_email_draft_with_identity_details_is_replaced_by_the_template` |
| Representatives need consent | A second gate after the three fields; a caller who said they are a representative cannot take it back. | `test_representative_cannot_drop_the_consent_gate_by_changing_their_story` |
| Caller text cannot change the rules | Verification is computed; the reader's record has no "verified" field to set. | `test_caller_instructions_cannot_set_verification` |

What is left to the prompt, and so is probabilistic: tone, how well the reader extracts messy language, and
whether an off-topic question is declined without a partial answer. An off-topic turn does get a scripted
brief and no tools, even in the open phase.

### Emotion and recovery

The reader labels emotion and intensity, refusals, "why do you need this", and demands to skip a step. The
harness turns those into the brief:

1. A tone note comes first: acknowledge what upset the caller, specifically, before any procedure.
2. If the caller pushed back, the brief requires the reason for the step (the SOP's own wording, in
   `sop.RATIONALE`) and the alternatives: any three of the five fields, whichever they are comfortable with.
3. On the second refusal in a row with no progress, or after two heated turns in a row, a human representative
   is offered as a real alternative.
4. On the third refusal the harness stops persuading and hands off, with a ticket saying what was and was not
   established. Supplying any identity detail resets the count.

A plain request for a human is honoured immediately in any phase. Three answers that do not match lock
verification for the conversation and hand off.

### Scope

The reader classifies each message as in scope, social, out of scope, or mixed. Out-of-scope turns are declined
and counted; three in a row hand off. A message that also carries something useful ("my DOB is ..., also what
is RL?") is treated as mixed: the detail is kept, the other part is declined, and no strike is counted.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | none | API token for the model. |
| `SOP_MODEL` | `claude-opus-5-5` | Model that writes replies. |
| `SOP_NLU_MODEL` | `claude-haiku-5-5` | Model that reads each message into a structured record. The smaller model is the default because it gave the same workflow outcomes as Opus 5.5 in every scenario tried and was 1.5 to 3 seconds quicker a turn. Set it to the writer's model to use one model for both. |
| `SOP_EFFORT` | `low` | Thinking effort for both calls. |
| `ANTHROPIC_FALLBACKS` | `default` | Server-side refusal fallback on models that support it; `off` to disable. Dropped automatically if the API rejects the parameter. |
| `LLM_PROVIDER` | `anthropic` | `stub` runs with no model. |
| `DEMO_TODAY` | `2026-03-02` | The sample data describes early March 2026; `real` uses the wall clock. |
| `COMPANY_NAME` | `Acme Insurance` | Name used in replies. |
| `EXPOSE_FIXTURES` | `1` | Serves the sample records to the console's Test data dialog. Set to `0` anywhere the records should stay private. |

Limits (three fields, three attempts, three refusals, three off-topic turns, three consent checks) are in
`sop.Limits`.

## Layout

```
app/
  sop.py          the SOP as data: phases, latitude, tools, transitions, limits, wording
  engine.py       the turn loop and the phase gates
  briefs.py       one directive builder per situation, with the harness's own script
  identity.py     normalising and matching identity fields
  tools.py        the tool gateway
  guards.py       output checks
  state.py        session, memory, trace
  data.py         fixture access, scoped to the verified party
  brain/
    base.py       the boundary: understand, respond, draft_email
    claude.py     Claude API implementation
    prompts.py    reader, writer and email prompts
    stub.py       offline keyword implementation
  main.py         HTTP API and static console
  static/         the console
fixtures/         the starter's sample data, unchanged
tests/            92 tests; none calls a model
```

## What has and has not been checked

Checked without a model: 92 automated tests (Python 3.10 and 3.12), the console in a browser with the offline
stub, and the Docker image.

Checked against the real API, by script through the same harness the console uses, in the default
configuration (Claude Haiku 5.5 reading, Claude Opus 5.5 writing, low effort): twelve conversations covering
the demo case, reason-before-identity, a clarifying question during verification, frustration and recovery,
three refusals, a request for a human, off-topic questions including one mixed with identity details, a wrong
date of birth then a correction, a representative with consent, two claims of the same type, moving between
claims, saying yes to an offer of help, a full call ending in a summary email, and two prompt-injection
attempts. Each ended in the expected phase with the expected gate decisions; no claim detail appeared before
verification; the grounding guard rejected nothing and no reply fell back to the script. Eleven of the twelve
were also run with Opus 5.5 as the reader, with the same outcomes. One turn was also made through the Docker
container and one through the console in a browser.

Measured turn time in the default configuration: about 3 to 6 seconds for a one-call reply, 7 to 11 seconds
when a turn also verifies, selects a claim or drafts the email. With Opus 5.5 on both calls: about 4 to 9 and
10 to 17.

That is one pass over hand-picked conversations, not an evaluation. It shows the path works and behaves
sensibly on these inputs; it does not measure how often the reader misreads unusual phrasing.

Known limits:

- The grounding guard checks that a figure is traceable, not that it is used correctly, and a value the caller
  said themselves can be repeated back.
- The verification lockout is per conversation. A real deployment needs a per-policy limit across calls.
- Telling a caller which field did not match is helpful for typos and is also feedback to someone guessing; the
  attempt limit bounds it.
- Sessions are in memory, and email and consent are simulated.
- Two model calls per turn at minimum, so turns take several seconds (figures above). Replies are not
  streamed, because each one is checked before the caller sees it.
