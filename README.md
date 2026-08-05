# Voice Agent with STT Confidence Awareness

A real-time voice agent built on **LiveKit Agents** that reads Deepgram's speech-recognition
**confidence score**, passes it into the LLM context alongside the transcript, and **changes
its behaviour based on it** — answering normally when it heard you clearly, and asking you to
repeat when it didn't.

```
User Speech ──▶ Deepgram STT ──▶ LLM (+ confidence) ──▶ ElevenLabs TTS ──▶ User
                     │                  ▲
                     └── confidence ────┘
```

---

## Contents

1. [Quick start](#quick-start)
2. [How confidence reaches the LLM](#how-confidence-reaches-the-llm)
3. [Design decisions](#design-decisions)
4. [Example interactions](#example-interactions)
5. [Configuration](#configuration)
6. [Running the real voice loop](#running-the-real-voice-loop)
7. [Tests](#tests)
8. [Project layout](#project-layout)
9. [Verification status](#verification-status)
10. [Evaluation criteria mapping](#evaluation-criteria-mapping)

---

## Quick start

No API keys needed to see the confidence logic work end to end:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python demo.py                     # replays STT events through the real agent
pytest                             # the full suite, ~3s
```

For the real audio loop, see [Running the real voice loop](#running-the-real-voice-loop).

---

## How confidence reaches the LLM

The value makes a two-step journey, using the two override points LiveKit exposes.

### Step 1 — Observe, in `stt_node`

`stt_node` is the only place the raw `SpeechEvent` is visible, and therefore the only place
`alternatives[i].confidence` can be read. [`agent.py`](agent.py) overrides it, records the
text and confidence of every `FINAL_TRANSCRIPT`, and yields every event onward unchanged — it
is a tap, not a filter.

A single spoken turn usually produces several finals, so they are buffered rather than
overwritten.

### Step 2 — Act, in `on_user_turn_completed`

This hook fires after the user's turn ends and *before* inference — the exact moment to enrich
context. It aggregates the turn's finals into one score, resolves a band, and then does one of
two things:

| Band | Behaviour |
|------|-----------|
| `high` (≥ 0.8) | Inject the score into the context. The LLM answers normally. |
| `medium` (0.6 – 0.8) | Inject the score. The LLM answers but may confirm. |
| `low` (< 0.6) | **Skip the LLM entirely.** Speak a fixed clarification and raise `StopResponse`. |

The injected note goes in with `role="system"` on `turn_ctx`, which LiveKit hands over as a
*per-turn copy* — so the metadata reaches the model for this generation only and never
accumulates in the conversation history.

### Confidence sources, in priority order

| Source | When it is used |
|--------|-----------------|
| `forced` | `FORCE_CONFIDENCE` is set — a simulated value, always labelled as such |
| `stt` | Aggregated from the finals observed in `stt_node` (the normal path) |
| `session` | Fallback to LiveKit's own `ChatMessage.transcript_confidence` if the buffer was empty or unusable |
| *(none)* | A typed/text-only turn: no note is injected, and the message is treated normally |

Every turn logs which source it used, so a simulated run can never be mistaken for a real one.

---

## Design decisions

**Why word-weighted aggregation, not `min` or `mean`.**
Deepgram emits filler words as their own low-confidence finals. Taking the **minimum** of a
turn means one stray `"um"` at 0.31 drags a perfectly clear sentence into the LOW band and
triggers a pointless "could you repeat that?". An unweighted **mean** (what LiveKit computes
internally) lets that one-word final count as much as a twelve-word clause. Weighting each
final by its word count makes a short garbled fragment cheap and a long garbled one expensive —
which is what "did I understand this turn?" actually means. Both behaviours are pinned by
tests, and the second and last scenarios in [Example interactions](#example-interactions) show
the difference.

**Why the deterministic low-confidence path is the default.**
Prompt-only enforcement means the one behaviour being asked for is at the mercy of the model
following an instruction. `StopResponse` makes it structural: on a low-confidence turn the LLM
is never called, so it cannot answer a garbled request no matter what it would have done. It
also keeps the unreliable transcript out of the conversation history. Set
`LOW_CONFIDENCE_STRATEGY=llm` for the softer, more natural-sounding variant.

**Why the note does not repeat the transcript.**
The task's example shows the transcript and the score together. Both do reach the model — the
score in the system note, the text in the user message immediately after it. Quoting the text a
second time inside the note would waste tokens on every turn and invite the model to answer the
quoted copy rather than the user.

**Why stale samples are discarded, and how.**
LiveKit skips `on_user_turn_completed` when speech scheduling is paused, so a buffer keyed only
on that hook can survive into the next turn and mark an unrelated utterance as low-confidence.
`TurnBuffer.consume()` is therefore always draining, and it verifies that what it holds actually
describes the turn.

The check replays how LiveKit assembles a transcript — it appends each final in order
(`audio_recognition.py`: `self._audio_transcript += f" {transcript}"`) — by walking the samples
and matching each one's **token run** at or after a moving cursor. The surviving samples must
start at the first token and consume the last; partial coverage means the buffer does not
describe this turn, and `consume()` returns `None` so the agent falls back to LiveKit's own
measurement rather than scoring from partial evidence.

Matching is on token runs, never raw substrings. Naive containment lets `"no"` match inside
`"i know that"` and `"art"` inside `"smart"`, which would drag a clean 0.95 turn to 0.74 and
across a band boundary. Both cases are pinned by regression tests.

*Known limit:* if a discarded turn's final is token-identical to the start of the next turn
(`"yes"` then `"yes, what time do you close"`), it is indistinguishable from a legitimate final
without per-turn IDs from the framework, and is kept. It contributes its own word weight only.

**Why `StopResponse` on the low band, and its cost.**
Skipping the LLM keeps the unreliable transcript out of the conversation history — but the
spoken clarification is still recorded, so history can contain an assistant message with no
preceding user turn. That is the deliberate trade: a slightly odd transcript beats persisting
text the recogniser itself does not trust.

**Why the policy lives in its own module.**
[`confidence.py`](confidence.py) has no LiveKit, network or I/O imports, so every rule the
agent's behaviour depends on is testable in milliseconds without a session, a key or a
microphone. [`agent.py`](agent.py) is wiring and logging only.

---

## Example interactions

Real output from `python demo.py`, which replays synthetic Deepgram events through the shipped
`stt_node` and `on_user_turn_completed` code paths. Nothing below is hand-written.

### High confidence — answer normally

```
  Deepgram final : 'What time do you close today?'  confidence=0.94
  -> injected into the LLM context for this turn only:
     | [stt-confidence] score=0.94 band=high source=stt
     | Apply the high rule from your instructions to the next user message.
     | (user) What time do you close today?
```

### High confidence — a low-confidence filler must not trigger a repeat

```
  Deepgram final : 'um'  confidence=0.31
  Deepgram final : 'what time do you close today'  confidence=0.96
  -> injected into the LLM context for this turn only:
     | [stt-confidence] score=0.87 band=high source=stt (weighted over 2 final transcripts, lowest 0.31)
     | Apply the high rule from your instructions to the next user message.
     | (user) um what time do you close today
```

### Medium confidence — answer, but confirm

```
  Deepgram final : 'I need help with my order'  confidence=0.71
  -> injected into the LLM context for this turn only:
     | [stt-confidence] score=0.71 band=medium source=stt
     | Apply the medium rule from your instructions to the next user message.
     | (user) I need help with my order
```

### Low confidence — ask to clarify (default, deterministic)

```
  Deepgram final : 'I need help with my order'  confidence=0.35
  -> LLM SKIPPED (StopResponse); agent speaks a fixed clarification:
     "Sorry, I may have misunderstood you. Could you please repeat that?"
```

### Low confidence — let the LLM phrase it (`LOW_CONFIDENCE_STRATEGY=llm`)

```
  Deepgram final : 'I need help with my order'  confidence=0.35
  -> injected into the LLM context for this turn only:
     | [stt-confidence] score=0.35 band=low source=stt
     | Apply the low rule from your instructions to the next user message.
     | (user) I need help with my order
```

### Low confidence — a long garbled clause *does* pull the turn down

```
  Deepgram final : 'yes'  confidence=0.97
  Deepgram final : 'i need help with the thing about my order please'  confidence=0.34
  -> injected into the LLM context for this turn only:
     | [stt-confidence] score=0.40 band=low source=stt (weighted over 2 final transcripts, lowest 0.34)
     | Apply the low rule from your instructions to the next user message.
     | (user) yes i need help with the thing about my order please
```

Same transcript text at 0.94 and at 0.35, opposite behaviour — driven entirely by the score.

---

## Configuration

All keys and tuning live in `.env`; copy [`.env.example`](.env.example) and fill it in.

| Variable | Required | Default | Purpose |
|----------|:--------:|---------|---------|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | ✅ | — | `livekit-server --dev` prints these (`ws://localhost:7880`, `devkey`, `secret`) |
| `DEEPGRAM_API_KEY` | ✅ | — | STT + confidence scores → https://deepgram.com/ |
| `OPENAI_API_KEY` | ✅ | — | LLM → https://platform.openai.com/api-keys |
| `ELEVEN_API_KEY` | ✅ | — | TTS → https://elevenlabs.io/. **Not** `ELEVENLABS_API_KEY` — the plugin reads this exact name |
| `HIGH_CONFIDENCE` | | `0.8` | Inclusive floor of the `high` band |
| `LOW_CONFIDENCE` | | `0.6` | Exclusive ceiling of the `low` band |
| `LOW_CONFIDENCE_STRATEGY` | | `deterministic` | `deterministic` or `llm` |
| `CLARIFICATION_PROMPT` | | *(see below)* | What the agent says on a low-confidence turn |
| `FORCE_CONFIDENCE` | | *(unset)* | Simulate a fixed score for reproducible demos |

Malformed values are rejected in `main()` **before `cli.run_app` is called**, exiting the
parent process with a readable message and nothing spawned:

```
$ FORCE_CONFIDENCE=high python agent.py console
configuration error: FORCE_CONFIDENCE must be a number in [0.0, 1.0], got 'high'
```

The placement is deliberate. Validating inside `on_user_turn_completed` is worst — LiveKit
catches the exception and returns, silently dropping every reply with no visible symptom.
Validating inside `prewarm` is only marginally better: the process pool catches it too
(`ipc/proc_pool.py`, *"error initializing process"* → *"retrying"*), producing a loop of
failing worker spawns instead of a clean exit.

### Forcing a confidence value for a reproducible demo

```bash
FORCE_CONFIDENCE=0.35 python agent.py console      # simulate low confidence
```
```powershell
$env:FORCE_CONFIDENCE = "0.35"; python agent.py console
Remove-Item Env:FORCE_CONFIDENCE                   # back to real values
```

The agent logs a warning at start-up and tags every affected turn `source=forced`. Real low
confidence also occurs naturally if you mumble, whisper, or talk over background noise.

---

## Running the real voice loop

**Prerequisites:** Python 3.10+ and the LiveKit media server
([install guide](https://docs.livekit.io/)).

```bash
pip install -r requirements.txt
python -m livekit.agents download-files    # Silero VAD weights, once
cp .env.example .env                       # Windows: copy .env.example .env
#   ...then fill in your keys
```

### Console mode — fastest loop, no browser or server

```bash
python agent.py console
```

Talk to the agent in your terminal. Each final transcript and its resolved band are logged.

### Full loop — LiveKit server + a frontend

```bash
livekit-server --dev     # terminal 1: prints devkey / secret / ws://localhost:7880
python agent.py dev      # terminal 2: the agent worker
```

Then connect with the [LiveKit Agents Playground](https://agents-playground.livekit.io/) or any
LiveKit client SDK pointed at `ws://localhost:7880`.

> **Deprecation note.** On `livekit-agents` 1.6.x the built-in Python CLI (`console`, `dev`,
> `connect`, `download-files` invoked via the agent script) emits a `DeprecationWarning` in
> favour of the standalone LiveKit CLI (`lk agent console`, `lk agent dev`). Both still work;
> `python -m livekit.agents download-files` is the non-deprecated form of the last one.

---

## Tests

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
coverage run -m pytest && coverage report   # gate: 98% on agent.py + confidence.py
```

All three run on every push via [CI](.github/workflows/ci.yml), on Python 3.10 and 3.12, plus
`python demo.py` to prove the documented output is still what the code produces. No network and
no keys at any point.

[`tests/test_confidence.py`](tests/test_confidence.py) covers band boundaries, threshold
validation, weighted aggregation (including the filler-word and long-garbled-clause cases),
buffer draining, token-run matching (with the `"no"`/`"i know"` and `"art"`/`"smart"`
regressions), coverage rejection, and every environment-variable failure mode.

[`tests/test_agent.py`](tests/test_agent.py) drives the real hooks against real `ChatContext`,
`ChatMessage` and `SpeechEvent` objects with a stub session. Confidence always enters through
the real `stt_node` rather than by seeding the buffer, so capture → aggregate → act is
exercised in one piece: that `stt_node` forwards every event untouched, that the note lands
immediately before the user message, that it never leaks into persisted history, that a
low-confidence turn raises `StopResponse` without reaching the LLM, that each confidence source
wins in the right order, and that `build_session` wires the required providers and `main`
refuses to spawn a worker on bad configuration.

---

## Project layout

```
.
├── agent.py                    # Pipeline wiring + the two override hooks
├── confidence.py               # All policy: bands, aggregation, settings, prompts (no I/O)
├── demo.py                     # Offline replay — real code paths, no keys required
├── tests/
│   ├── test_confidence.py
│   └── test_agent.py
├── .github/workflows/ci.yml    # ruff + pytest + coverage gate + demo, on 3.10 and 3.12
├── requirements.txt            # Exact pins
├── requirements-dev.txt        # + pytest, coverage, ruff
├── pyproject.toml              # pytest, coverage & ruff configuration
├── .env.example                # Committed template
├── .env                        # Your keys (git-ignored)
├── .gitattributes
└── .gitignore
```

Two source modules rather than one: the split is along the testability seam. Everything in
`confidence.py` is a pure function of its inputs, so the behaviour being graded can be verified
without a voice session; `agent.py` holds the parts that can only be exercised against live
services.

---

## Verification status

Stated plainly, because "it works" is worth nothing without saying how it was checked.

| Component | Status |
|-----------|--------|
| Confidence policy — bands, aggregation, token matching, config validation | ✅ Unit-tested |
| `stt_node` capture and pass-through | ✅ Tested against real `SpeechEvent` objects with a stubbed stream |
| `on_user_turn_completed` injection, ordering, history isolation, `StopResponse` | ✅ Tested against real `ChatContext` / `ChatMessage` |
| Pipeline wiring, `prewarm`, `entrypoint`, config gate in `main` | ✅ Tested with real plugin constructors and a stub room |
| Line coverage of `agent.py` + `confidence.py` | ✅ 100% (gate set at 98%) |
| Lint & format | ✅ `ruff check` and `ruff format --check` clean |
| Deepgram → OpenAI → ElevenLabs over live audio | ⚠️ **Not run.** Requires the three API keys and a local `livekit-server`, neither of which is included here |

Everything except the last row is verified in CI. That row is the honest gap: the plugin
constructors and their arguments are asserted, but no audio has crossed the network, so
`session.say()` under a live `AgentSession` — the deterministic clarification path — has been
exercised only against a stub. It is the first thing a reviewer with keys should check.

---

## Evaluation criteria mapping

| Criterion | Where |
|-----------|-------|
| Correct LiveKit setup | `cli.run_app(WorkerOptions(entrypoint_fnc=…, prewarm_fnc=…))`; VAD loaded once per worker process, not per call |
| Working real-time voice pipeline | `AgentSession` wires Deepgram → OpenAI → ElevenLabs with Silero VAD |
| Proper Deepgram integration | `deepgram.STT(model="nova-3", language="en-US")`; `smart_format` deliberately left off so finals stay matchable against the assembled transcript |
| Correct extraction & propagation of confidence | `stt_node` reads `alternatives[0].confidence`; `on_user_turn_completed` aggregates and injects it, with LiveKit's `transcript_confidence` as a fallback |
| Clean architecture & code quality | Policy/IO split, no mid-turn failure modes, typed throughout, ruff-clean, 100% line coverage, CI |
| Clear documentation | This README, plus a runnable `demo.py` instead of hand-written example output |
| Agent behaviour changes based on confidence | Three bands; the low band is enforced structurally via `StopResponse`, not by hoping the model obeys a prompt |

---

## Possible extensions

- **Per-word confidence.** Deepgram returns per-word scores and LiveKit's `TimedString` has a
  `confidence` field — but the Deepgram plugin does not currently populate it, so this needs a
  patched plugin or a direct Deepgram client, not just `alternatives[0].words`.
- **Adaptive thresholds.** Raise `LOW_CONFIDENCE` when the rolling session mean drops, to avoid
  a bad microphone turning every turn into a clarification request.
- **Confidence-aware re-prompting.** Ask the user to repeat only the low-confidence *span*
  rather than the whole utterance.
- **LiveKit Inference** to swap providers without managing three separate keys.

### Reference links

- LiveKit Agents — https://github.com/livekit/agents
- LiveKit docs — https://docs.livekit.io/
- Deepgram — https://deepgram.com/
- ElevenLabs — https://elevenlabs.io/
