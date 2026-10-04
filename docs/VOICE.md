# CHALKDUST — Voice

How narration becomes audio, and how one spec runs on more than one machine.
Pipeline position and caching are in `ARCHITECTURE.md` §2–3; this page is the backends.

---

## Backends

| `backend` | Platform | Install | Raw output | Default `voice_id` |
|---|---|---|---|---|
| `macos_say` | macOS | none | AIFF | `Daniel` |
| `windows_sapi` | Windows | none (System.Speech via Windows PowerShell 5.1) | WAV | `Microsoft David Desktop` |
| `kokoro` | any | `pip install -e ".[kokoro]"` (torch; Python <3.13) | WAV | `af_heart` |

`macos_say` and `windows_sapi` are development voices: zero install, real measured
durations. `kokoro` is the intended production voice (`ARCHITECTURE.md` §10).

Each backend writes its own native format and declares it (`raw_format`); the speech
stage then normalises everything to 24kHz mono 16-bit WAV and trims edge silence.
Nothing downstream ever sees a backend's raw file.

Listing voices:
- macOS: `say -v '?'`
- Windows: `powershell -c "Add-Type -A System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices().VoiceInfo.Name"`
- Kokoro: <https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md>. The first
  letter of a voice is its language (`a` American, `b` British English).

`rate` is a speed multiplier everywhere. SAPI only speaks between 1/3× and 3×, in steps
of about 12%; outside that range it raises rather than clamping.

## Resolution: what a spec that says nothing gets

Only fields the spec actually sets count. For the rest:

1. **No `backend`** → the platform's zero-install voice: `macos_say` on macOS,
   `windows_sapi` on Windows. Anywhere else the speech stage raises and asks for one.
2. **No `voice_id`** → the resolved backend's default voice (table above). Voice ids
   mean nothing across backends, so a `voice_id` without a `backend` only makes sense
   on the platform it was written for.
3. **`rate`** is kept as given (default 1.0).

So a spec with no `voice` block, or one that sets only `rate`, runs unchanged on macOS
and Windows. It will not *sound* the same on both. A spec that must — anything headed
for publication — names `backend` and `voice_id` explicitly, usually `kokoro`.

The TTS cache is keyed on the **resolved** voice (D-004), so the same spec never shares
cached audio across platforms. On macOS resolution yields exactly the old defaults
(`macos_say`/`Daniel`), so audio cached before this rule stays valid.

## Python 3.13

The speech stage shells out to ffmpeg/ffprobe and uses no `audioop` (removed in 3.13)
or `pydub`. Manim itself depends on `pydub`, and manim 0.21.0 already declares
`audioop-lts; python_full_version >= '3.13'` for it, so nothing extra is needed here.

Kokoro 0.9.4 and its `misaki` dependency declare `Requires-Python <3.13`; on 3.13 the
`kokoro` extra fails to install with pip's Requires-Python error. Use a 3.12
environment for Kokoro until upstream publishes 3.13 support.
