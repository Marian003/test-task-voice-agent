# Voice agent with STT confidence awareness

A real-time voice agent on LiveKit Agents that reads Deepgram's confidence score, passes it
into the LLM context next to the transcript, and behaves differently depending on it. If it
heard you clearly it answers. If it didn't, it says so and asks you to repeat, rather than
guessing at what you might have said.

```
speech ──▶ Deepgram STT ──▶ LLM (+ confidence) ──▶ ElevenLabs TTS ──▶ speech
                │                 ▲
                └─── confidence ──┘
```

## Try it without any API keys

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python demo.py     # replays STT events through the real agent
pytest             # unit tests plus a full session on fake providers
```

The real audio loop needs keys; see [Running it for real](#running-it-for-real).

## How the confidence gets to the LLM

Two override points, one for reading the number and one for acting on it.

**`stt_node`** is the only place the raw `SpeechEvent` is exposed, so it's the only place
`alternatives[i].confidence` can be read at all. The agent overrides it, notes the text and
confidence of every `FINAL_TRANSCRIPT`, and yields every event straight through. It's a tap,
not a filter. A single spoken turn usually produces several finals, so they're buffered rather
than overwritten.

**`on_user_turn_completed`** fires after the turn ends and before inference, which is exactly
the moment to enrich context. It rolls the turn's finals into one score, works out the band,
and then either annotates or intervenes:

| Band | What happens |
|------|--------------|
| high (≥ 0.8) | Score goes into the context. The LLM answers normally. |
| medium (0.6–0.8) | Score goes into the context. The LLM answers, maybe confirms. |
| low (< 0.6) | The LLM is skipped entirely. Fixed clarification, then `StopResponse`. |

The note goes in with `role="system"` on `turn_ctx`, which LiveKit hands over as a per-turn
copy. So it reaches the model for that one generation and never accumulates in history.

Where the number comes from, in priority order: `FORCE_CONFIDENCE` if it's set (always tagged
`source=forced`, so a simulated run can't be mistaken for a real one), otherwise the finals
captured in `stt_node`, otherwise LiveKit's own `ChatMessage.transcript_confidence` as a
fallback. If none of those produce anything, which is the case for a typed turn, nothing is
injected and the message is treated normally.

## Some decisions worth explaining

**Word-weighted aggregation instead of min or mean.** Deepgram emits filler words as their own
low-confidence finals. Take the minimum of a turn and one stray `"um"` at 0.31 drags a
perfectly clear sentence into the low band, costing the user a pointless "could you repeat
that?". Take a flat mean, which is what LiveKit computes internally, and that one-word final
counts as much as a twelve-word clause. Weighting by word count makes a short garbled fragment
cheap and a long garbled one expensive. Both behaviours are pinned by tests, and you can see
the difference in the examples below.

**The low-confidence path is deterministic by default.** Enforcing this through the prompt
alone means the one behaviour the whole thing exists for depends on a model choosing to
comply. `StopResponse` makes it structural instead: the LLM is never called, so it can't answer
a garbled request no matter what. Set `LOW_CONFIDENCE_STRATEGY=llm` if you'd rather have the
softer, more natural-sounding version.

The cost is worth stating. Skipping the LLM keeps the untrusted transcript out of history, but
the spoken clarification is still recorded, so history can end up with an assistant message and
no user turn before it. I'd rather have a slightly odd transcript than persist text the
recogniser itself doesn't trust.

**The note doesn't repeat the transcript.** The task's example shows the text and the score
together, and both do reach the model: the score in the system note, the text in the user
message immediately after it. Quoting the text a second time inside the note would burn tokens
every turn and tempt the model into answering the copy.

**Leftover finals get thrown away.** LiveKit skips `on_user_turn_completed` when speech
scheduling is paused, so a buffer that only clears in that hook can survive into the next turn
and mark an unrelated utterance as low-confidence. `TurnBuffer.consume()` always drains, and
checks that what it's holding actually describes the turn.

The check replays how LiveKit builds a transcript, which is to append each final in order
(`audio_recognition.py`: `self._audio_transcript += f" {transcript}"`). Walk the samples, keep
each one whose tokens sit *exactly* at a moving cursor, and require the cursor to land on the
last token. Anything short of that and `consume()` returns `None`, so the agent falls back to
LiveKit's number instead of scoring on partial evidence.

Two things this got wrong on the way, both now regression-tested:

- Matching on substrings rather than token runs let `"no"` match inside `"i know that"` and
  `"art"` inside `"smart"`, pulling a clean 0.95 turn down to 0.74 and across a band edge.
- Matching "somewhere at or after the cursor" rather than exactly at it accepted gaps:
  `["yes", "no"]` against `"yes yes no"` passed a start-and-end check while skipping the middle
  token entirely, scoring 0.53 instead of deferring.

**Policy sits in its own module.** [`confidence.py`](confidence.py) imports nothing from
LiveKit and does no I/O, so every rule the behaviour depends on is testable in milliseconds.
[`agent.py`](agent.py) is wiring and logging.

## Examples

Real output from `python demo.py`. Nothing here is hand-written.

High confidence, answer normally:

```
  Deepgram final : 'What time do you close today?'  confidence=0.94
  -> injected into the LLM context for this turn only:
     | [stt-confidence] score=0.94 band=high source=stt
     | Apply the high rule from your instructions to the next user message.
     | (user) What time do you close today?
```

A filler word must not cost the user a repeat:

```
  Deepgram final : 'um'  confidence=0.31
  Deepgram final : 'what time do you close today'  confidence=0.96
  -> injected into the LLM context for this turn only:
     | [stt-confidence] score=0.87 band=high source=stt (weighted over 2 final transcripts, lowest 0.31)
     | Apply the high rule from your instructions to the next user message.
     | (user) um what time do you close today
```

Medium confidence, answer but confirm:

```
  Deepgram final : 'I need help with my order'  confidence=0.71
  -> injected into the LLM context for this turn only:
     | [stt-confidence] score=0.71 band=medium source=stt
     | Apply the medium rule from your instructions to the next user message.
     | (user) I need help with my order
```

Low confidence, ask to clarify. This is the default:

```
  Deepgram final : 'I need help with my order'  confidence=0.35
  -> LLM SKIPPED (StopResponse); agent speaks a fixed clarification:
     "Sorry, I may have misunderstood you. Could you please repeat that?"
```

The same turn under `LOW_CONFIDENCE_STRATEGY=llm`, where the model phrases it:

```
  Deepgram final : 'I need help with my order'  confidence=0.35
  -> injected into the LLM context for this turn only:
     | [stt-confidence] score=0.35 band=low source=stt
     | Apply the low rule from your instructions to the next user message.
     | (user) I need help with my order
```

And a long garbled clause, which *should* pull the turn down:

```
  Deepgram final : 'yes'  confidence=0.97
  Deepgram final : 'i need help with the thing about my order please'  confidence=0.34
  -> injected into the LLM context for this turn only:
     | [stt-confidence] score=0.40 band=low source=stt (weighted over 2 final transcripts, lowest 0.34)
     | Apply the low rule from your instructions to the next user message.
     | (user) yes i need help with the thing about my order please
```

Same words at 0.94 and at 0.35, opposite behaviour, driven entirely by the score.

## Configuration

Copy [`.env.example`](.env.example) to `.env` and fill it in.

| Variable | Required | Default | What it's for |
|----------|:--------:|---------|---------------|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | yes | | `livekit-server --dev` prints these: `ws://localhost:7880`, `devkey`, `secret` |
| `DEEPGRAM_API_KEY` | yes | | STT and confidence scores, from https://deepgram.com/ |
| `OPENAI_API_KEY` | yes | | LLM, from https://platform.openai.com/api-keys |
| `ELEVEN_API_KEY` | yes | | TTS, from https://elevenlabs.io/. Note the name: the plugin reads `ELEVEN_API_KEY`, not `ELEVENLABS_API_KEY` |
| `HIGH_CONFIDENCE` | | 0.8 | Inclusive floor of the high band |
| `LOW_CONFIDENCE` | | 0.6 | Exclusive ceiling of the low band |
| `LOW_CONFIDENCE_STRATEGY` | | deterministic | `deterministic` or `llm` |
| `CLARIFICATION_PROMPT` | | see below | What gets said on a low turn |
| `FORCE_CONFIDENCE` | | unset | Simulate a fixed score, for reproducible demos |

Bad values are caught in `main()` before `cli.run_app` runs, so the process exits with a
readable message and nothing gets spawned:

```
$ FORCE_CONFIDENCE=high python agent.py console
configuration error: FORCE_CONFIDENCE must be a number in [0.0, 1.0], got 'high'
```

That placement is deliberate. Validating inside `on_user_turn_completed` is the worst option:
LiveKit catches the exception and returns, so every reply is silently dropped with no visible
symptom. Validating in `prewarm` is barely better, because the process pool catches it too
(`ipc/proc_pool.py`, "error initializing process" then "retrying") and you get a loop of
failing worker spawns instead of a clean exit.

### Forcing a score for a demo

```bash
FORCE_CONFIDENCE=0.35 python agent.py console
```
```powershell
$env:FORCE_CONFIDENCE = "0.35"; python agent.py console
Remove-Item Env:FORCE_CONFIDENCE       # back to real values
```

You get a warning at startup and every affected turn is tagged `source=forced`. Real low
confidence also happens naturally if you mumble, whisper, or talk over background noise.

## Running it for real

You'll need Python 3.10+ and the LiveKit media server ([install guide](https://docs.livekit.io/)).

```bash
pip install -r requirements.txt
python -m livekit.agents download-files    # Silero VAD weights, once
cp .env.example .env                       # Windows: copy .env.example .env
# then fill in your keys
```

Console mode is the fastest loop, no browser or server needed:

```bash
python agent.py console
```

Or the full thing:

```bash
livekit-server --dev     # terminal 1, prints devkey / secret / ws://localhost:7880
python agent.py dev      # terminal 2
```

Then connect with the [Agents Playground](https://agents-playground.livekit.io/) or any LiveKit
client SDK pointed at `ws://localhost:7880`.

One note on deprecations: on livekit-agents 1.6.x the built-in Python CLI (`console`, `dev`,
`connect`, and `download-files` when invoked through the agent script) emits a
`DeprecationWarning` in favour of the standalone LiveKit CLI (`lk agent console`, `lk agent
dev`). Both still work. `python -m livekit.agents download-files` is the non-deprecated form of
the last one.

## Tests

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
coverage run -m pytest && coverage report      # gate: 98% on agent.py + confidence.py
```

There's a [CI workflow](.github/workflows/ci.yml) that runs all of that on 3.10 and 3.12, plus
`python demo.py` so the output quoted above can't quietly drift from the code. Everything in it
passes locally; the workflow itself hasn't run on a real runner yet, since this hasn't been
pushed anywhere with Actions enabled. No network and no keys at any point.

Three layers:

[`tests/test_confidence.py`](tests/test_confidence.py) covers the policy in isolation: band
edges, threshold validation, weighted aggregation including the filler-word and
long-garbled-clause cases, buffer draining, token-run matching with both matcher regressions
above, and every environment-variable failure mode.

[`tests/test_agent.py`](tests/test_agent.py) drives the hooks against real `ChatContext`,
`ChatMessage` and `SpeechEvent` objects. Confidence always enters through the real `stt_node`
rather than by seeding the buffer, so capture, aggregation and action are exercised as one
path. It also covers the wiring: that `build_session` puts the required providers together, and
that `main` refuses to start a worker on bad config.

[`tests/test_integration.py`](tests/test_integration.py) runs a **real `AgentSession`** with
scripted STT, LLM and TTS plugins ([`tests/fakes.py`](tests/fakes.py)) that implement the same
base classes the Deepgram, OpenAI and ElevenLabs plugins do. The session, turn detection,
scheduling and hook invocation are all genuine; only the provider calls are faked. This is what
verifies the things a stubbed session can't: that LiveKit actually calls
`on_user_turn_completed`, that a low turn reaches TTS with the clarification and never reaches
the LLM at all, that the garbled transcript stays out of history, and that the note doesn't
accumulate across turns.

## Layout

```
.
├── agent.py                    # pipeline wiring and the two hooks
├── confidence.py               # bands, aggregation, settings, prompts. no I/O
├── demo.py                     # offline replay, real code paths, no keys
├── tests/
│   ├── fakes.py                # scripted STT/LLM/TTS plugins
│   ├── test_confidence.py      # policy
│   ├── test_agent.py           # hooks and wiring
│   └── test_integration.py     # real AgentSession, fake providers
├── .github/workflows/ci.yml
├── requirements.txt            # exact pins
├── requirements-dev.txt
├── pyproject.toml
├── .env.example
└── .gitignore
```

Two source modules rather than one. The split is along the testability seam: everything in
`confidence.py` is a pure function of its inputs, so the behaviour being graded can be checked
without a voice session, while `agent.py` holds what can only be exercised against a session.

## What's actually been verified

Worth being precise about, because "it works" means nothing without saying how you know.

Verified in CI, on every push: the confidence policy end to end; `stt_node` capture and
pass-through; injection, ordering, history isolation and `StopResponse` under a real
`AgentSession`; the provider wiring and the config gate; 100% line coverage of `agent.py` and
`confidence.py`; ruff clean.

Not verified: audio over the network. The plugin constructors and their arguments are asserted,
but no bytes have gone to Deepgram, OpenAI or ElevenLabs from this checkout. The integration
tests exercise the same session code with fake providers, so what's untested is narrowly the
three network hops and the plugins themselves, not the agent's logic. That's the part a
reviewer with keys should run.

## Rough edges I know about

Listing these rather than pretending they aren't there:

- The integration tests wait on a fixed `asyncio.sleep(0.6)` per turn instead of on session
  events. Stable locally and 3/3 on repeat runs, but it's a wall-clock assumption, it's the
  first thing I'd expect to flake on a busy runner, and it's why the suite takes ~17s rather
  than ~3s. There's a TODO on it.
- `TurnBuffer.consume()` can't distinguish a leftover final that happens to be
  token-identical to the start of the next turn. Needs per-turn ids from the framework to do
  properly; the cost is one final's word weight, so I left it.
- `load_dotenv()` runs at import, so `tests/conftest.py` has to clear the agent's own env vars
  before each test. Works, but it's a workaround for module-level import side effects.

## Ideas if this were to grow

- Per-word confidence. Deepgram returns per-word scores and LiveKit's `TimedString` has a
  `confidence` field, but the Deepgram plugin doesn't populate it, so this needs a patched
  plugin or a direct Deepgram client rather than just `alternatives[0].words`.
- Adaptive thresholds: raise `LOW_CONFIDENCE` when the rolling session mean drops, so a bad
  microphone doesn't turn every single turn into a clarification request.
- Ask the user to repeat only the low-confidence span rather than the whole utterance.
- LiveKit Inference, to stop juggling three separate provider keys.

### Links

- LiveKit Agents: https://github.com/livekit/agents
- LiveKit docs: https://docs.livekit.io/
- Deepgram: https://deepgram.com/
- ElevenLabs: https://elevenlabs.io/
