"use strict";

// Voice-over: reads QUEEN's explanation aloud through the local server
// (ElevenLabs) and highlights each move and square as it is spoken. Character
// timings from the speech line up each token with the moment it is said.

const speechCache = new Map(); // Spoken script → Promise of speech.
let narration = null; // { id, text, parts, part, cue, audio, url, frame }
let narrationId = 0;

// How long a highlight lingers after its words end, in seconds.
const CUE_LINGER = 1.5;

function spokenToken(token, paragraph) {
  const { move } = token;
  if (move) {
    if (isCastle(move))
      return `${move.color} castles ${move.to[0] === "g" ? "kingside" : "queenside"}`;
    return `${move.color} ${move.piece} ${move.capture ? "takes" : "to"} ${move.to}`;
  }
  if (token.range) return `${token.range[0]} to ${token.range[1]}`;
  if (token.file) return `${token.file} file`;
  return paragraph.slice(token.start, token.end);
}

// Rewrites a paragraph for speech ("12.white knight f1-g3" → "white knight
// to g3") and records where each token lands in the spoken text.
function spokenScript({ text, tokens }) {
  let script = "";
  let last = 0;
  const cues = tokens.map((token, index) => {
    let gap = text.slice(last, token.start);
    // Back-to-back moves in a line get a pause between them.
    if (index > 0 && !gap.trim() && token.move && tokens[index - 1].move)
      gap = ", ";
    script += gap;
    const start = script.length;
    script += spokenToken(token, text);
    last = token.end;
    return { index, start, end: script.length };
  });
  script += text.slice(last);
  return { script, cues };
}

function fetchSpeech(script) {
  if (!speechCache.has(script)) {
    const request = fetch("/api/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: script }),
    }).then(async (response) => {
      const speech = await response.json();
      if (!response.ok) throw new Error(speech.error || "Voice-over failed.");
      return speech;
    });
    // Failed requests can be retried.
    request.catch(() => speechCache.delete(script));
    speechCache.set(script, request);
  }
  return speechCache.get(script);
}

// Converts script offsets to seconds. The timings are per character of the
// text sent; scale defensively in case the service normalized it.
function timeCues(part, speech) {
  const scale = speech.starts.length / part.script.length || 1;
  const at = (offset, times) =>
    times[Math.min(times.length - 1, Math.floor(offset * scale))] ?? 0;
  for (const cue of part.cues) {
    cue.time = at(cue.start, speech.starts);
    cue.until = at(cue.end - 1, speech.ends) + CUE_LINGER;
  }
}

function audioUrl(base64) {
  const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
  return URL.createObjectURL(new Blob([bytes], { type: "audio/mpeg" }));
}

async function startNarration() {
  const id = ++narrationId;
  const parts = explanationParagraphs
    .map((paragraph, index) => ({ index, ...spokenScript(paragraph) }))
    .filter((part) => /[a-z]/i.test(part.script));
  if (!parts.length) return;
  narration = { id, text: narratedText(), parts, part: null, cue: null };
  // Rest on the position QUEEN analyzed, before its reply was played.
  const fen = previewBases[0];
  restingPreview = fen && {
    position: pieces(fen),
    moved: new Set(),
    key: "rest " + fen,
    fen,
  };
  clearLine();
  renderVoiceButton();
  try {
    for (const [i, part] of parts.entries()) {
      // Fetch one paragraph ahead so playback doesn't stall between them.
      if (parts[i + 1]) fetchSpeech(parts[i + 1].script).catch(() => {});
      const speech = await fetchSpeech(part.script);
      if (narration?.id !== id) return;
      timeCues(part, speech);
      await playPart(part, speech, id);
      if (narration?.id !== id) return;
    }
  } catch (error) {
    if (narration?.id === id) {
      $("notice").textContent = error.message;
      $("notice").hidden = false;
    }
  }
  if (narration?.id === id) stopNarration();
}

function playPart(part, speech, id) {
  return new Promise((resolve, reject) => {
    const url = audioUrl(speech.audio);
    const audio = new Audio(url);
    Object.assign(narration, { part, cue: null, audio, url });
    clearLine(); // Each paragraph starts from the analyzed position.
    applyCue();
    const finish = () => {
      cancelAnimationFrame(narration?.frame);
      URL.revokeObjectURL(url);
      resolve();
    };
    audio.addEventListener("ended", finish);
    audio.addEventListener("pause", () => narration?.id !== id && finish());
    audio.addEventListener("error", () =>
      reject(new Error("The voice-over couldn't be played.")),
    );
    const tick = () => {
      if (narration?.id !== id) return;
      const t = audio.currentTime;
      const cue = part.cues.findLast((c) => c.time <= t && t < c.until);
      if (cue !== narration.cue) {
        narration.cue = cue;
        applyCue();
      }
      narration.frame = requestAnimationFrame(tick);
    };
    audio.play().then(tick, reject);
    renderVoiceButton();
  });
}

// Highlights the spoken paragraph and token in the text and on the board.
function applyCue() {
  const { part, cue } = narration ?? {};
  for (const element of $("explanation").querySelectorAll(".speaking"))
    element.classList.remove("speaking");
  $("explanation").classList.toggle("narrating", Boolean(part));
  if (part) {
    const paragraph = $("explanation").querySelector(
      `[data-paragraph="${part.index}"]`,
    );
    paragraph?.classList.add("speaking");
    if (cue)
      $("explanation")
        .querySelector(`[data-token="${part.index}:${cue.index}"]`)
        ?.classList.add("speaking");
    if (paragraph && narration.scrolled !== part.index) {
      narration.scrolled = part.index;
      paragraph.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }
  const token = cue && explanationParagraphs[part.index]?.tokens[cue.index];
  let squares = null;
  if (token?.move) {
    if (!showLine(token.line)) clearLine();
    squares = new Map([
      [token.move.from, "from"],
      [token.move.to, "to"],
    ]);
  } else if (token) {
    const kind = token.squares.length > 1 ? "path" : "focus";
    squares = new Map(token.squares.map((square) => [square, kind]));
  }
  marks = squares && { squares, key: [...squares].join() };
  renderBoard();
}

function stopNarration() {
  if (!narration) return;
  const { audio, url } = narration;
  cancelAnimationFrame(narration.frame);
  narration = null;
  audio?.pause();
  if (url) URL.revokeObjectURL(url);
  applyCue();
  restingPreview = null;
  clearLine();
  renderVoiceButton();
}

function narratedText() {
  return explanationParagraphs.map((paragraph) => paragraph.text).join("\n\n");
}

function renderVoiceButton() {
  const button = $("listen");
  const loading = narration && !narration.audio;
  button.textContent = loading ? "Preparing…" : narration ? "Stop" : "Listen";
  button.setAttribute("aria-pressed", String(Boolean(narration)));
}

// After the explanation re-renders, stop if the text changed (a new move or a
// different explanation); otherwise restore the highlights.
$("explanation").addEventListener("rendered", () => {
  if (!narration) return;
  if (narration.text !== narratedText()) stopNarration();
  else applyCue();
});
$("listen").addEventListener("click", () => {
  if (narration) stopNarration();
  else startNarration();
});

fetch("/api/voice")
  .then((response) => response.json())
  .then(({ enabled }) => {
    $("listen").hidden = !enabled;
  })
  .catch(() => {});
