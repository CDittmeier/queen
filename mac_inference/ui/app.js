"use strict";

const $ = (id) => document.getElementById(id);
const pieceNames = {
  p: "pawn",
  n: "knight",
  b: "bishop",
  r: "rook",
  q: "queen",
  k: "king",
};
let game = null;
let orientation = "white";
let selected = null;
let dragging = null;
let pending = false;
let requestEpoch = 0;
let boardKey = "";
let movesKey = "";
let explanationKey = "";
let promotionMoves = [];

function pieces(fen) {
  const map = {};
  fen
    .split(" ")[0]
    .split("/")
    .forEach((row, rank) => {
      let file = 0;
      for (const item of row) {
        if (/\d/.test(item)) file += Number(item);
        else map[String.fromCharCode(97 + file++) + (8 - rank)] = item;
      }
    });
  return map;
}

function colorOf(piece) {
  return piece === piece.toUpperCase() ? "white" : "black";
}
function canMove() {
  return (
    game && !pending && game.phase === "playing" && game.turn === game.human
  );
}
function candidates(from, to) {
  return game.legal_moves.filter((move) => move.startsWith(from + (to || "")));
}

function renderBoard() {
  if (!game) return;
  const key = [
    game.fen,
    game.phase,
    game.human,
    game.last_move,
    orientation,
    selected,
    pending,
  ].join("|");
  if (key === boardKey) return;
  boardKey = key;
  const position = pieces(game.fen);
  const files = orientation === "white" ? "abcdefgh" : "hgfedcba";
  const ranks =
    orientation === "white"
      ? [8, 7, 6, 5, 4, 3, 2, 1]
      : [1, 2, 3, 4, 5, 6, 7, 8];
  const fragment = document.createDocumentFragment();
  ranks.forEach((rank, row) =>
    [...files].forEach((file, col) => {
      const square = file + rank;
      const piece = position[square];
      const button = document.createElement("button");
      button.type = "button";
      button.className = "square";
      button.dataset.square = square;
      const name = piece
        ? `${colorOf(piece)} ${pieceNames[piece.toLowerCase()]}`
        : "empty";
      button.setAttribute("aria-label", `${square}, ${name}`);
      button.setAttribute("aria-pressed", String(square === selected));
      if ((file.charCodeAt(0) - 97 + rank) % 2 === 1)
        button.classList.add("dark");
      if (piece) button.classList.add("occupied");
      if (square === selected) button.classList.add("selected");
      if (
        game.last_move &&
        [game.last_move.slice(0, 2), game.last_move.slice(2, 4)].includes(
          square,
        )
      )
        button.classList.add("last-move");
      if (selected && canMove() && candidates(selected, square).length)
        button.classList.add("available");
      if (
        game.in_check &&
        piece &&
        piece.toLowerCase() === "k" &&
        colorOf(piece) === game.turn
      )
        button.classList.add("in-check");
      if (piece) {
        const image = document.createElement("img");
        image.src = `/pieces/${colorOf(piece) === "white" ? "w" : "b"}${piece.toUpperCase()}.svg`;
        image.alt = "";
        image.draggable = false;
        button.append(image);
      }
      if (col === 0) {
        const label = document.createElement("span");
        label.className = "rank";
        label.textContent = rank;
        label.setAttribute("aria-hidden", "true");
        button.append(label);
      }
      if (row === 7) {
        const label = document.createElement("span");
        label.className = "file";
        label.textContent = file;
        label.setAttribute("aria-hidden", "true");
        button.append(label);
      }
      button.addEventListener("click", () => clickSquare(square));
      button.draggable = Boolean(
        canMove() && piece && colorOf(piece) === game.human,
      );
      button.addEventListener("dragstart", (event) => {
        const image = button.querySelector("img");
        if (!image?.complete || !image.naturalWidth) {
          event.preventDefault();
          return;
        }
        const bounds = image.getBoundingClientRect();
        const preview = document.createElement("canvas");
        preview.width = Math.max(1, Math.round(bounds.width));
        preview.height = Math.max(1, Math.round(bounds.height));
        const context = preview.getContext("2d");
        if (!context) {
          event.preventDefault();
          return;
        }
        // Render only the SVG at its displayed size, with a transparent background.
        context.drawImage(image, 0, 0, preview.width, preview.height);
        event.dataTransfer.setDragImage(
          preview,
          preview.width / 2,
          preview.height / 2,
        );
        dragging = square;
        event.dataTransfer.setData("text/plain", square);
        event.dataTransfer.effectAllowed = "move";
      });
      button.addEventListener("dragover", (event) => {
        if (canMove() && dragging && candidates(dragging, square).length)
          event.preventDefault();
      });
      button.addEventListener("drop", (event) => {
        event.preventDefault();
        const from = dragging;
        dragging = null;
        if (from && canMove()) chooseMove(from, square);
      });
      button.addEventListener("dragend", () => {
        dragging = null;
      });
      button.addEventListener("keydown", (event) => {
        const offsets = {
          ArrowLeft: -1,
          ArrowRight: 1,
          ArrowUp: -8,
          ArrowDown: 8,
        };
        if (event.key in offsets) {
          event.preventDefault();
          const target = row * 8 + col + offsets[event.key];
          const squares = $("board").children;
          if (target >= 0 && target < 64) squares[target].focus();
        }
        if (event.key === "Escape") {
          selected = null;
          renderBoard();
        }
      });
      fragment.append(button);
    }),
  );
  $("board").replaceChildren(fragment);
}

function clickSquare(square) {
  if (!canMove()) return;
  if (selected && candidates(selected, square).length) {
    chooseMove(selected, square);
    return;
  }
  const piece = pieces(game.fen)[square];
  selected =
    piece && colorOf(piece) === game.human && selected !== square
      ? square
      : null;
  renderBoard();
}

function chooseMove(from, to) {
  const options = candidates(from, to);
  if (!options.length) return;
  if (options.length > 1) {
    promotionMoves = options;
    $("promotion-dialog").showModal();
    return;
  }
  action("/api/move", { uci: options[0] });
}

function renderMoves() {
  const key = JSON.stringify(game.moves);
  if (key === movesKey) return;
  movesKey = key;
  $("move-count").textContent = `${game.moves.length} played`;
  if (!game.moves.length) {
    const empty = document.createElement("span");
    empty.className = "muted";
    empty.textContent = "The story of your game starts here.";
    $("moves").replaceChildren(empty);
    return;
  }
  const fragment = document.createDocumentFragment();
  for (let i = 0; i < game.moves.length; i += 2) {
    const first = game.moves[i];
    const number = document.createElement("span");
    number.className = "move-number";
    number.textContent = first.number + ".";
    fragment.append(number);
    for (let j = 0; j < 2; j++) {
      const item = game.moves[i + j];
      const cell = document.createElement("span");
      cell.className = "move-san";
      cell.textContent = item ? item.san : "—";
      if (i + j === game.moves.length - 1) cell.classList.add("latest");
      fragment.append(cell);
    }
  }
  $("moves").replaceChildren(fragment);
  $("moves").scrollTop = $("moves").scrollHeight;
}

function renderExplanation() {
  const thinking = game.phase === "thinking";
  const text = thinking ? game.thinking_text : game.analysis?.text || "";
  const key = game.phase + "|" + text;
  if (key !== explanationKey) {
    const wasThinking = explanationKey.startsWith("thinking|");
    explanationKey = key;
    if (text) {
      const prose = text
        .replace(/^ANALYSIS:\s*/, "")
        .split(
          /\n(?:BEST_MOVE|CRITICAL_LINE|PRINCIPAL_VARIATION|PROMISING_MOVES|EVALUATION):/,
        )[0];
      const fragment = document.createDocumentFragment();
      prose.split(/\n\s*\n/).forEach((paragraph) => {
        const p = document.createElement("p");
        p.textContent = paragraph;
        fragment.append(p);
      });
      $("explanation").replaceChildren(fragment);
      if (thinking) $("explanation").scrollTop = $("explanation").scrollHeight;
      else if (wasThinking) $("explanation").scrollTop = 0;
    } else if (thinking) {
      const p = document.createElement("p");
      p.className = "thinking-placeholder";
      p.textContent = game.engine_loading
        ? "QUEEN is warming up. Its first reply is on the way…"
        : "QUEEN is considering its reply…";
      $("explanation").replaceChildren(p);
    } else {
      $("explanation").replaceChildren(emptyExplanation());
    }
  }
  $("analysis-summary").hidden = thinking || !game.analysis?.best_move_san;
  if (game.analysis?.best_move_san) {
    const a = game.analysis;
    $("analysis-move").textContent =
      `QUEEN played ${a.move_number}${a.color === "black" ? "…" : "."} ${a.best_move_san}`;
    $("analysis-timing").textContent = `${a.generation_seconds.toFixed(1)}s`;
  }
  $("retry-box").hidden = game.phase !== "error";
  $("ai-error").textContent = game.error || "";
}

function emptyExplanation() {
  const box = document.createElement("div");
  box.className = "empty-explanation";
  const icon = document.createElement("div");
  icon.className = "empty-piece";
  icon.textContent = "♞";
  icon.setAttribute("aria-hidden", "true");
  const title = document.createElement("h3");
  title.textContent = "Strong moves. Clear intentions.";
  const prose = document.createElement("p");
  prose.textContent =
    "Make your first move. QUEEN will play its reply and explain the ideas behind it, right here.";
  const note = document.createElement("span");
  note.className = "small-note";
  note.textContent = "A real game, running entirely on your Mac.";
  box.append(icon, title, prose, note);
  return box;
}

function render() {
  if (!game) return;
  $("connection").textContent = game.engine_loading
    ? "Warming up"
    : game.error && !game.analysis
      ? "Local demo"
      : "Local · Ready";
  $("connection-dot").classList.toggle("loading", game.engine_loading);
  const topColor = orientation === "white" ? "black" : "white";
  const bottomColor = orientation;
  for (const [where, color] of [
    ["top", topColor],
    ["bottom", bottomColor],
  ]) {
    const avatar = $(where + "-avatar");
    avatar.textContent = color === game.human ? "Y" : "♛";
    avatar.className = `avatar ${color === game.human ? "human-avatar" : "queen-avatar"}`;
    $(where + "-player").textContent = color === game.human ? "You" : "QUEEN";
    $(where + "-color").textContent =
      (color === "white" ? "White" : "Black") + " pieces";
    $(where + "-turn").textContent =
      game.phase !== "finished" && game.turn === color
        ? color === game.human
          ? "Your turn"
          : "Thinking"
        : "";
  }
  let status = "Your turn";
  if (game.phase === "thinking")
    status = game.engine_loading ? "QUEEN is warming up" : "QUEEN is thinking";
  if (game.phase === "error") status = "QUEEN needs another try";
  if (game.phase === "finished") {
    const o = game.outcome;
    status = o.winner
      ? `${o.winner === game.human ? "You win" : "QUEEN wins"} · ${o.reason}`
      : `Draw · ${o.reason}`;
  } else if (game.in_check && canMove()) status = "Your turn · check";
  $("status").textContent = status;
  $("status-dot").classList.toggle("loading", game.phase === "thinking");
  $("elapsed").textContent =
    game.phase === "thinking" ? `${Math.floor(game.thinking_seconds)}s` : "";
  $("hint").textContent = canMove()
    ? "Click a piece, then a highlighted square. Or drag to move."
    : game.phase === "finished"
      ? "Start a new game, or take back your last turn."
      : "Your next move will be available after QUEEN replies.";
  $("undo").disabled = pending || !game.can_take_back;
  $("resign").disabled = pending || game.phase === "finished";
  $("claim-draw").hidden = !game.can_claim_draw;
  $("claim-draw").disabled = pending;
  $("new-game").disabled = pending;
  $("retry").disabled = pending;
  renderBoard();
  renderMoves();
  renderExplanation();
}

function acceptState(next) {
  if (game && next.game_id === game.game_id && next.version < game.version)
    return;
  if (!game || next.game_id !== game.game_id) {
    orientation = next.human;
    selected = null;
    boardKey = "";
    movesKey = "";
    explanationKey = "";
  }
  game = next;
  render();
}

async function action(path, body = {}) {
  if (pending || !game) return;
  const epoch = ++requestEpoch;
  pending = true;
  selected = null;
  $("notice").hidden = true;
  render();
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        game_id: game.game_id,
        version: game.version,
        ...body,
      }),
    });
    const next = await response.json();
    if (!response.ok)
      throw new Error(next.error || "That action couldn't be completed.");
    if (epoch === requestEpoch) acceptState(next);
  } catch (error) {
    $("notice").textContent = error.message;
    $("notice").hidden = false;
  } finally {
    pending = false;
    render();
  }
}

async function poll() {
  const epoch = requestEpoch;
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error("Connection lost");
    const next = await response.json();
    if (epoch === requestEpoch && !pending) acceptState(next);
  } catch {
    $("connection").textContent = "Reconnecting…";
    $("connection-dot").classList.add("loading");
  }
  setTimeout(
    poll,
    game?.phase === "thinking" || game?.engine_loading ? 250 : 1200,
  );
}

$("flip").addEventListener("click", () => {
  orientation = orientation === "white" ? "black" : "white";
  render();
});
$("undo").addEventListener("click", () => action("/api/undo"));
$("resign").addEventListener("click", () => action("/api/resign"));
$("claim-draw").addEventListener("click", () => action("/api/draw"));
$("retry").addEventListener("click", () => action("/api/retry"));
$("new-game").addEventListener("click", () => {
  $("color").value = game?.human || "white";
  $("new-game-dialog").showModal();
});
$("cancel-new").addEventListener("click", () => $("new-game-dialog").close());
$("new-game-form").addEventListener("submit", (event) => {
  event.preventDefault();
  $("new-game-dialog").close();
  action("/api/new", { color: $("color").value });
});
$("cancel-promotion").addEventListener("click", () =>
  $("promotion-dialog").close(),
);
document.querySelectorAll("[data-promotion]").forEach((button) =>
  button.addEventListener("click", () => {
    const move = promotionMoves.find((uci) =>
      uci.endsWith(button.dataset.promotion),
    );
    $("promotion-dialog").close();
    if (move) action("/api/move", { uci: move });
  }),
);
poll();
