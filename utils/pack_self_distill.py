"""Filter durable consolidation outputs, rewrite evaluations, and pack POV SFT data.

The filtering and numerical-first rewrite follow the final Sol/Hero recipe.
No fresh engine searches are performed: scores come from the accepted roots.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import tempfile
from datetime import datetime, timezone

import chess
import yaml

from datagen.self_distill.analysis import ANALYSIS_PROMPT as PROMPT, MATE, expected_winrate
from datagen.self_distill.consolidation import audit, input_record_identity, output_record_identity
from datagen.self_distill.io import atomic_json

FIELD_ORDER = [
    "ANALYSIS", "BEST_MOVE", "CRITICAL_LINE", "PROMISING_MOVES", "EVALUATION"
]

FIELD_HEADING = re.compile(
    r"(?m)^(ANALYSIS|BEST_MOVE|CRITICAL_LINE|PROMISING_MOVES|EVALUATION):"
)

TAG = re.compile(r"<([^<>]+)>")

ABS_SQUARE = re.compile(r"<SQUARE_([A-H][1-8])>")

ABS_PIECE = re.compile(
    r"<(WHITE|BLACK)_(PAWN|KNIGHT|BISHOP|ROOK|QUEEN|KING)>"
)

ABS_PLAYER = re.compile(r"<(WHITE|BLACK)>")

FILE_TOKEN = re.compile(r"<FILE_([A-H])>")

RANK_TOKEN = re.compile(r"<RANK_([1-8])>")

DIAGONAL_TOKEN = re.compile(r"<DIAGONAL_([A-H][1-8])_([A-H][1-8])>")

POV_PIECE_LETTER = {
    "PAWN": "P",
    "KNIGHT": "N",
    "BISHOP": "B",
    "ROOK": "R",
    "QUEEN": "Q",
    "KING": "K",
}

KNOWN_TAGS = [
    re.compile(r"SQUARE_[A-H][1-8]"),
    re.compile(r"(?:WHITE|BLACK)_(?:PAWN|KNIGHT|BISHOP|ROOK|QUEEN|KING)"),
    re.compile(r"(?:WHITE|BLACK)"),
    re.compile(r"FILE_[A-H]"),
    re.compile(r"RANK_[1-8]"),
    re.compile(r"DIAGONAL_[A-H][1-8]_[A-H][1-8]"),
]

KNOWN_INTERMEDIATE_TOKEN = re.compile(
    r"<(?:"
    r"SQUARE_[A-H][1-8]|"
    r"(?:WHITE|BLACK)_(?:PAWN|KNIGHT|BISHOP|ROOK|QUEEN|KING)|"
    r"(?:WHITE|BLACK)|FILE_[A-H]|RANK_[1-8]|"
    r"DIAGONAL_[A-H][1-8]_[A-H][1-8]"
    r")>"
)

POV_TAGS = [
    re.compile(r"SQUARE_(?:[1-9]|[1-5][0-9]|6[0-4])"),
    re.compile(r"PIECE_[MO][PNBRQK]"),
    re.compile(r"EMPTY"),
]

KNOWN_POV_TOKEN = re.compile(
    r"<(?:SQUARE_(?:[1-9]|[1-5][0-9]|6[0-4])|PIECE_[MO][PNBRQK]|EMPTY)>"
)

TERMINAL_RESPONSES = {
    "This position is checkmate.",
    "This position is drawn.",
}

def unexpected_tags(text: str) -> list[str]:
    return sorted({
        tag for tag in TAG.findall(text)
        if not any(pattern.fullmatch(tag) for pattern in KNOWN_TAGS)
    })

def malformed_token_fragments(text: str, known_token: re.Pattern[str]) -> list[str]:
    """Return compact contexts around unmatched token delimiters.

    This catches malformed forms such as ``<SQUARE_H5}`` which do not have a
    closing ``>`` and are therefore invisible to a conventional ``<...>``
    token regex.
    """
    residue = known_token.sub("", text)
    contexts = []
    for match in re.finditer(r"[<>{}]", residue):
        start, end = max(0, match.start() - 24), min(len(residue), match.end() + 24)
        contexts.append(residue[start:end].replace("\n", "\\n"))
    return sorted(set(contexts))[:8]

def rejection_reasons(record: dict) -> list[str]:
    text = (record.get("consolidated_explanation") or "").strip()
    terminal = record.get("mode") == "terminal_passthrough"
    reasons = []
    if terminal:
        if record.get("finish_reason") != "terminal":
            reasons.append("terminal record has a nonterminal finish reason")
        if text not in TERMINAL_RESPONSES:
            reasons.append("malformed terminal response")
        if record.get("audit_issues"):
            reasons.extend(record["audit_issues"])
    else:
        if record.get("mode") != "qwen_consolidation":
            reasons.append(f"unknown mode: {record.get('mode')}")
        if record.get("finish_reason") != "stop":
            reasons.append(f"finish reason: {record.get('finish_reason')}")
        if FIELD_HEADING.findall(text) != FIELD_ORDER:
            reasons.append("fields are missing, duplicated, or out of order")
        saved = record.get("audit_issues") or []
        recomputed = audit(record, text)
        reasons.extend(saved)
        reasons.extend(issue for issue in recomputed if issue not in saved)
    tags = unexpected_tags(text)
    if tags:
        reasons.append("unexpected vocabulary tokens: " + ", ".join(tags))
    fragments = malformed_token_fragments(text, KNOWN_INTERMEDIATE_TOKEN)
    if fragments:
        reasons.append("malformed token delimiters: " + " | ".join(fragments))
    return list(dict.fromkeys(reasons))

def rejection_category(reason: str) -> str:
    """Collapse record-specific diagnostics into stable manifest counters."""
    if reason.startswith("finish reason:"):
        return "truncated_generation"
    if reason.startswith("fields are ") or reason.startswith("missing fields:"):
        return "malformed_field_structure"
    if reason.startswith("unexpected vocabulary tokens:"):
        return "unexpected_vocabulary_token"
    if reason.startswith("malformed token delimiters:"):
        return "malformed_token_delimiter"
    if reason.startswith("promising moves:") or reason.startswith("PROMISING_MOVES"):
        return "invalid_promising_moves"
    if reason.startswith("critical line"):
        return "invalid_critical_line"
    if reason.startswith("BEST_MOVE"):
        return "invalid_best_move"
    if reason == "response contains POV vocabulary":
        return "unexpected_pov_vocabulary"
    if reason.startswith("terminal record") or reason == "malformed terminal response":
        return "malformed_terminal_response"
    return "other"

def normalized_fen(fen: str) -> str:
    return " ".join(fen.split()[:4])

def geometry_to_endpoints(text: str) -> str:
    text = DIAGONAL_TOKEN.sub(
        lambda match: f"<SQUARE_{match.group(1)}>-<SQUARE_{match.group(2)}>", text
    )
    text = FILE_TOKEN.sub(
        lambda match: f"<SQUARE_{match.group(1)}1>-<SQUARE_{match.group(1)}8>", text
    )
    return RANK_TOKEN.sub(
        lambda match: f"<SQUARE_A{match.group(1)}>-<SQUARE_H{match.group(1)}>", text
    )

def absolute_to_pov(text: str, fen: str) -> str:
    pov = chess.Board(fen).turn
    text = geometry_to_endpoints(text)

    def square(match: re.Match[str]) -> str:
        absolute = chess.parse_square(match.group(1).lower())
        relative = absolute if pov == chess.WHITE else absolute ^ 56
        return f"<SQUARE_{relative + 1}>"

    def piece(match: re.Match[str]) -> str:
        colour = chess.WHITE if match.group(1) == "WHITE" else chess.BLACK
        side = "M" if colour == pov else "O"
        return f"<PIECE_{side}{POV_PIECE_LETTER[match.group(2)]}>"

    def player(match: re.Match[str]) -> str:
        colour = chess.WHITE if match.group(1) == "WHITE" else chess.BLACK
        return "the player" if colour == pov else "the opponent"

    text = ABS_SQUARE.sub(square, text)
    text = ABS_PIECE.sub(piece, text)
    text = ABS_PLAYER.sub(player, text)

    # Match the established prose convention at sentence and paragraph starts.
    text = re.sub(
        r"(^|[.!?]\s+|\n+)(the player|the opponent)\b",
        lambda match: match.group(1) + match.group(2).capitalize(),
        text,
    )
    leftovers = sorted({
        tag for tag in TAG.findall(text)
        if not any(pattern.fullmatch(tag) for pattern in POV_TAGS)
    })
    if leftovers:
        raise ValueError(f"absolute-to-POV conversion left tokens: {leftovers}")
    fragments = malformed_token_fragments(text, KNOWN_POV_TOKEN)
    if fragments:
        raise ValueError(f"POV response contains malformed token delimiters: {fragments}")
    return text

def write_arrow(path: Path, dataset, split: str, task_name: str) -> None:
    dataset.save_to_disk(str(path))
    atomic_json(path / "dataset_config.json", {
        "pov": True,
        "split": split,
        "task": task_name,
        "n_records": len(dataset),
        "prompt": PROMPT,
    })

EVAL = re.compile(r"(?ms)^EVALUATION:.*\Z")

def evaluation_field(text):
    match = EVAL.search(text)
    return match.group().split(":", 1)[1].strip() if match else ""

def read_rows(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def source_info(row):
    step = row["trajectory"][-1]
    history = step.get("history")
    if history is None:
        history = ((row.get("source", {}).get("history") or [])
                   + [s['fen'] for s in row['trajectory'][:-1]])[-7:]
    if step.get('terminal'):
        mated = chess.Board(step['fen']).is_checkmate()
        return {'history':history, 'true_cp':-100000 if mated else 0,
                'true_winrate':0.0 if mated else 0.5}
    result = {"history": history, "true_cp": step["true_root_cp"],
              "true_winrate": step["true_root_winrate"]}
    label = step.get("decision", {}).get("selected_label")
    child = next((c for c in step.get("children", []) if c["label"] == label), None)
    if child is not None:
        result.update(predicted_winrate=child["predicted_root_winrate"],
                      move_winrate=child["true_root_winrate"],
                      selected_move=child["move_uci"])
    return result

def evaluation(cp):
    if abs(cp) >= 99000:
        distance = 100000 - abs(int(cp))
        sign = "+" if cp > 0 else "-"
        return f"EVALUATION: {sign}M{distance} from the player's perspective."
    return f"EVALUATION: {cp / 100:+.2f} pawns from the player's perspective."

def parsed_final_evaluation(text, side):
    """Read the consolidated numerical claim, independently of the SF answer.

    Explicit numeric POV takes priority. Otherwise the local evaluation clause
    identifies the advantaged/disadvantaged side; later tactical prose must not
    override it. Square/move digits are never evaluation numbers. Mate prose is
    an exact expected-score endpoint, even when its distance is not supplied.
    """
    value = evaluation_field(text)
    player, opponent = ('<WHITE>', '<BLACK>') if side else ('<BLACK>', '<WHITE>')
    value = value.replace('<PLAYER>', player).replace('<OPPONENT>', opponent)
    value = re.sub(r'\b(?:the\s+)?player\b', player, value, flags=re.I)
    value = re.sub(r'\b(?:the\s+)?opponent\b', opponent, value, flags=re.I)
    value = re.sub(r'(?i)(?<![<_])\b(white|black)\b(?![_>])',
                   lambda m:'<'+m.group().upper()+'>', value)
    masked = re.sub(r'<[^>]+>', lambda m: m.group() if m.group() in ('<WHITE>', '<BLACK>')
                    else ' '*len(m.group()), value)
    explicit = re.search(r"(?i)(?:from\s+)?<(WHITE|BLACK)>(?:['’]s)?\s+(?:perspective|point of view|pov)",masked)
    mate = MATE.search(masked)
    mate_prose = re.search(r'(?i)\b(?:checkmate|mate|mated)\b',masked)
    atom = mate or re.search(r'(?<![\w.])[+-]?\d+\.\d+(?!\w)',masked)
    if atom is None:
        atom = re.search(r'(?<![\w.])[+-]?\d+(?=\s*pawns?\b)',masked,re.I)
    if atom is None:
        atom = re.match(r'\s*([+-]?\d+)(?![\w.])',masked)
    # Determine the side from the evaluation clause, not a later explanation.
    end=atom.end() if atom else 180
    tail=re.split(r'(?i)[,;]|\b(?:as|with|after|though|because)\b',masked[end:],maxsplit=1)[0] if atom else ''
    clause = masked[:end]+tail
    positive = r'(?:up|ahead|wins?|winning|better|favou?red|advantage|edge|initiative|delivers?\s+(?:immediate\s+|forced\s+)?(?:checkmate|mate)|has\s+delivered\s+checkmate|forces?\s+(?:a\s+)?(?:checkmate|mate)|has\s+(?:a\s+)?(?:forced\s+)?(?:checkmate|mate|win))'
    negative = r'(?:down|behind|loses?|losing|lost|worse|disadvantage|deficit|concession|mated|getting\s+mated|faces?\s+(?:an?\s+)?(?:immediate\s+|forced\s+)?checkmate)'
    claims=[]
    for m in re.finditer(r'<(WHITE|BLACK)>([^<>;!?]*)',clause,re.I):
        description=m.group(2)
        neg=re.search(r'\b'+negative+r'\b',description,re.I)
        pos=re.search(r'\b'+positive+r'\b',description,re.I)
        if neg or pos:
            bad=bool(neg and (not pos or neg.start()<pos.start()))
            claims.append((m.start(),m.group(1).upper(),-1 if bad else 1))
    for m in re.finditer(r'(?i)\b(winning|better|advantage|edge|losing|lost|worse|disadvantage)\s+(?:for|to)\s+<(WHITE|BLACK)>',clause):
        claims.append((m.start(),m.group(2).upper(),-1 if m.group(1).lower() in ('losing','lost','worse','disadvantage') else 1))
    claim=min(claims) if claims else None
    if claim is None and mate_prose:
        m=re.search(r'(?i)(?:checkmate|mate)(?:\s+in\s+\w+(?:\s+moves?)?)?\s+for\s+<(WHITE|BLACK)>',masked)
        if m:claim=(m.start(),m.group(1).upper(),1)
    if claim is None:
        m=re.match(r'(?i)\s*(?:decisively\s+)?(losing|winning|lost|better|worse)\b',masked)
        if m:claim=(0,'WHITE' if side else 'BLACK',-1 if m.group(1).lower() in ('losing','lost','worse') else 1)
    kind='mate' if mate or (mate_prose and (atom is None or
        (mate_prose.start()<atom.start() and not re.search(r'\bpawns?\b',masked[:atom.end()+8],re.I)))) else 'pawn'
    if kind=='pawn' and atom is None:
        raise ValueError('no numerical evaluation or definite mate claim')
    raw_sign=1
    magnitude=None
    if mate:
        raw_sign=1 if mate.group(1)=='+' else -1
        magnitude=int(mate.group(2))
    elif kind=='pawn':
        number=float(atom.group().strip());magnitude=abs(number)
        raw_sign=(number>0)-(number<0)
    if explicit:
        white_sign=raw_sign*(1 if explicit.group(1).upper()=='WHITE' else -1)
        method='explicit_perspective'
    elif claim:
        white_sign=claim[2]*(1 if claim[1]=='WHITE' else -1)
        if kind=='pawn' and magnitude==0:white_sign=0
        method='local_side_claim'
    elif kind=='mate' and not mate:
        raise ValueError('mate claim has no identifiable winning side')
    else:
        subject=re.match(r'\s*<(WHITE|BLACK)>',clause)
        numeric_for=re.search(r'(?i)\bpawns?\s+for\s+<(WHITE|BLACK)>',clause)
        numeric_pov=subject or numeric_for
        white_sign=raw_sign*(1 if not numeric_pov or numeric_pov.group(1)=='WHITE' else -1)
        method='named_numeric_subject' if numeric_pov else 'white_pov_fallback'
    root_sign=white_sign if side else -white_sign
    if kind=='mate':
        winrate=1. if root_sign>0 else 0.
        display=('+' if root_sign>0 else '-')+'M'+(str(magnitude) if magnitude is not None else '?')
    else:
        winrate=expected_winrate(round(root_sign*magnitude*100))
        display=f'{root_sign*magnitude:+.2f}'
    result={'raw':value,'kind':kind,'method':method,'root_pov':display,'winrate':winrate}
    if atom:result['number_span']=atom.span()
    if mate:result['mate_distance_span']=mate.span(2)
    return result

def corrected_evaluation(text, fen, cp):
    """Renumber only when the POV parser agrees before and after the change."""
    side = chess.Board(fen).turn
    colour = '<WHITE>' if side else '<BLACK>'
    blanket = evaluation(cp).replace("the player", colour)
    try:
        parsed = parsed_final_evaluation(text, side)
    except ValueError:
        return blanket, 'full_unparseable', None
    predicted_sign = (parsed['winrate'] > .5) - (parsed['winrate'] < .5)
    true_sign = (cp > 0) - (cp < 0)
    if predicted_sign != true_sign:
        return blanket, 'full_direction_change', parsed
    target_kind = 'mate' if abs(cp) >= 99000 else 'pawn'
    if parsed['kind'] != target_kind:
        return blanket, 'full_pawn_mate_change', parsed
    if target_kind == 'mate' and 'mate_distance_span' not in parsed:
        return blanket, 'full_prose_mate', parsed
    value = parsed['raw']  # Also canonicalizes player/opponent to color tokens.
    start, stop = parsed['number_span']
    old = value[start:stop]
    if target_kind == 'mate':
        # Preserve spelling and sign; change only the mate-distance digits.
        start, stop = parsed['mate_distance_span']
        replacement = str(100000 - abs(int(cp)))
    else:
        sign = '-' if old.startswith('-') else '+' if old.startswith('+') else ''
        replacement = f'{sign}{abs(cp)/100:.2f}'
    candidate = 'EVALUATION: ' + value[:start] + replacement + value[stop:]
    if target_kind == 'mate':
        # A field can state the same distance twice: "mate in 3 (+M3)".
        # Renumber the explicit prose distance too, leaving all prose intact.
        candidate = re.sub(r'(?i)(\bmate\s+in\s+)' + re.escape(value[start:stop]) + r'\b',
                           lambda m:m.group(1)+replacement, candidate)
    try:
        check = parsed_final_evaluation(candidate, side)
    except ValueError:
        return blanket, 'full_verification_failed', parsed
    expected = evaluation(cp).split(': ',1)[1].split(' ',1)[0]
    if check['root_pov'] != expected:
        return blanket, 'full_verification_failed', parsed
    return candidate, 'renumbered', parsed


def inventory(config):
    """Freeze published chunks only; neither inputs nor running writers are modified."""
    tasks = []
    seen = set()
    for directory in config['consolidated_roots']:
        directory = Path(directory).resolve()
        roots = [directory] if (directory / 'manifest.json').is_file() else sorted(directory.iterdir())
        for root in roots:
            manifest = root / 'manifest.json'
            if not manifest.is_file() or root in seen:
                continue
            seen.add(root)
            run = json.loads(manifest.read_text())
            if run.get('mode') != 'consolidate':
                raise ValueError(f'Not a consolidation manifest: {manifest}')
            source = Path(run['input']['path'])
            chunks = sorted((root / 'chunks').glob('chunk_*.jsonl'))
            if not chunks:
                continue
            tasks.append(dict(
                name=str(root), run_identity=run['run_identity'],
                source=file_identity(source), chunks=[file_identity(p) for p in chunks],
            ))
    if not tasks:
        raise ValueError('No published consolidation chunks found')
    return tasks


def file_identity(path):
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def verified_rows(spec):
    path = Path(spec['path'])
    if file_identity(path) != spec:
        raise ValueError(f'Input changed since snapshot: {path}')
    yield from read_rows(path)
    if file_identity(path) != spec:
        raise ValueError(f'Input changed while packing: {path}')


def pack(config):
    from datasets import Dataset, Features, Sequence, Value, load_from_disk

    output = Path(config['output'])
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}; use a new output directory')
    threshold = float(config.get('max_eval_error_exclusive', 0.075))
    validation_size = int(config.get('validation_size', 100))
    if not 0 < threshold <= 1 or validation_size < 1:
        raise ValueError('Evaluation tolerance must be in (0, 1]; validation size must be positive')
    tasks = inventory(config)
    validation_path = config.get('validation_arrow')
    validation = load_from_disk(validation_path) if validation_path else None
    validation_fens = {normalized_fen(f) for f in validation['fen']} if validation is not None else set()
    output.mkdir(parents=True)
    atomic_json(output / 'snapshot.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(), config=config, tasks=tasks,
    ))

    # filter: exclude malformed/terminal targets, duplicates, validation overlaps and bad evals.
    counts = Counter()
    by_source = defaultdict(Counter)
    record_ids, fens = set(), set()
    with (output / 'eligible.jsonl').open('w') as kept, \
            (output / 'heldout.jsonl').open('w') as held, \
            (output / 'evaluation_rewrites.jsonl').open('w') as rewrites:
        for task in tasks:
            metadata = {}
            for source in verified_rows(task['source']):
                identity = input_record_identity(source)
                if identity in metadata:
                    raise ValueError(f'Duplicate accepted identity: {identity}')
                metadata[identity] = source_info(source)
            for chunk in task['chunks']:
                for row in verified_rows(chunk):
                    identity = output_record_identity(row)
                    info = metadata[identity]
                    kind = {'game': 'general', 'self_play': 'play'}.get(row['source_kind'], row['source_kind'])
                    key = normalized_fen(row['fen'])
                    counts['input'] += 1
                    by_source[kind]['input'] += 1
                    reason, details, error = None, [], None
                    if identity in record_ids:
                        reason = 'duplicate_record'
                    elif row.get('terminal') or chess.Board(row['fen']).is_game_over():
                        reason = 'terminal'
                    elif key in validation_fens:
                        reason = 'validation_overlap'
                    else:
                        details = rejection_reasons(row)
                        if details:
                            reason = 'malformed'
                    record_ids.add(identity)
                    if reason is None:
                        text = row['consolidated_explanation']
                        replacement, mode, parsed = corrected_evaluation(text, row['fen'], info['true_cp'])
                        error = abs(parsed['winrate'] - info['true_winrate']) if parsed else None
                        if error is None:
                            reason = 'unparseable_evaluation'
                        elif error >= threshold:
                            reason = 'evaluation_error'
                        elif key in fens:
                            reason = 'duplicate_fen'
                    if reason:
                        counts[reason] += 1
                        by_source[kind][reason] += 1
                        held.write(json.dumps(dict(record_id=identity, fen=row['fen'], source_kind=kind,
                                                   reason=reason, details=details, eval_error=error)) + '\n')
                        continue
                    # rewrite: preserve evaluation prose when a numerical edit is consistent with its POV.
                    rewritten = EVAL.sub(lambda _: replacement, text)
                    response = absolute_to_pov(rewritten, row['fen'])
                    fens.add(key)
                    rewrites.write(json.dumps(dict(record_id=identity, fen=row['fen'], true_cp=info['true_cp'],
                        original=evaluation_field(text), replacement=replacement, mode=mode, eval_error=error)) + '\n')
                    kept.write(json.dumps(dict(fen=row['fen'], history=info['history'], prompt=PROMPT,
                        response=response, extra=dict(record_id=identity, source_kind=kind, terminal=False,
                        source_file=chunk['path'], task=config['task_name']))) + '\n')
                    counts['eligible'] += 1
                    counts[mode] += 1
                    by_source[kind]['eligible'] += 1
            print(json.dumps(dict(task=task['name'], counts=dict(counts))), flush=True)

    # split: reuse the prior heldout set, or draw a deterministic first-round validation split.
    count = counts['eligible']
    if count <= (validation_size if validation is None else 0):
        raise ValueError(f'Only {count} eligible examples; cannot form nonempty training/validation sets')
    held_indices = set(random.Random(config.get('seed', 20260828)).sample(range(count), validation_size)) \
        if validation is None else set()
    with (output / 'train.jsonl').open('w') as train, (output / 'validation.jsonl').open('w') as val:
        for index, row in enumerate(read_rows(output / 'eligible.jsonl')):
            split = 'validation' if index in held_indices else 'train'
            (val if split == 'validation' else train).write(json.dumps(row) + '\n')
            counts[split] += 1
            by_source[row['extra']['source_kind']][split] += 1
        if validation is not None:
            for row in validation:
                val.write(json.dumps(row) + '\n')
            counts['validation'] = len(validation)

    # pack: write disk-backed Arrow directories with the trainer's standard instance schema.
    features = Features(dict(fen=Value('string'), history=Sequence(Value('string')),
        prompt=Value('string'), response=Value('string'), extra=dict(record_id=Value('string'),
        source_kind=Value('string'), terminal=Value('bool'), source_file=Value('string'), task=Value('string'))))
    for split in ('train', 'validation'):
        if split == 'validation' and validation is not None:
            dataset = validation
        else:
            with tempfile.TemporaryDirectory(prefix='.arrow-cache-', dir=output) as cache:
                dataset = Dataset.from_generator(read_rows, gen_kwargs={'path': output / f'{split}.jsonl'},
                                                 features=features, cache_dir=cache)
                write_arrow(output / f'{split}.arrow', dataset, split, config['task_name'])
            continue
        write_arrow(output / f'{split}.arrow', dataset, split, config['task_name'])
    (output / 'eligible.jsonl').unlink()
    report = dict(config=config, counts=dict(counts), by_source={k: dict(v) for k, v in by_source.items()},
                  note='No additional source rebalancing or gameplay downsampling at packing time.')
    atomic_json(output / 'manifest.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    pack(yaml.safe_load(args.config.read_text()))


if __name__ == '__main__':
    main()
