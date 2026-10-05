"""Resumable tactical/general PV benchmarks using the historical scoring convention."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import chess
import chess.engine
import yaml

from datagen.self_distill.analysis import ANALYSIS_PROMPT
from datagen.self_distill.io import atomic_json
from eval.critical_line import parse_critical_line, win_rate, _prefix_suffix


def prepare(config, index):
    dataset = config['datasets'][index]
    source = Path(dataset['source'])
    sample = json.loads((source / 'sample.json').read_text())
    if not sample or len({r['sample_id'] for r in sample}) != len(sample):
        raise ValueError('Benchmark sample must be nonempty with unique sample IDs')
    for row in sample:
        for key in ('fen', 'prompt', 'correct_move_uci', 'solution_uci'):
            if key not in row:
                raise ValueError(f'Missing {key} in benchmark sample')
    output = Path(dataset['output'])
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(config=config, dataset=dataset['name'], sample_n=len(sample),
                    sample_sha256=hashlib.sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest())
    path = output / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError(f'Existing evaluation has different inputs/settings: {output}')
    atomic_json(path, manifest)
    atomic_json(output / 'sample.json', sample)
    return sample, output


def generate(config, sample, output):
    path = output / 'generations.json'
    rows = json.loads(path.read_text()) if path.exists() else []
    if [r['sample_id'] for r in rows] != [r['sample_id'] for r in sample[:len(rows)]]:
        raise ValueError('Saved generations are not a prefix of the benchmark sample')
    if len(rows) == len(sample):
        return
    from models.vllm.flamingo_generate import ChessFlamingoGenerator
    settings = config['generation']
    generator = ChessFlamingoGenerator(
        config['model'], config['encoder'], max_model_len=settings['max_tokens'] + 512,
        max_num_seqs=settings['max_num_seqs'], gpu_memory_utilization=settings['gpu_memory_utilization'],
        seed=settings['seed'], enforce_eager=True, use_v1_vllm=True,
    )
    # generation: the dataset's neutral prompt is preserved unless training mode is explicitly selected.
    for start in range(len(rows), len(sample), settings['chunk_size']):
        chunk = sample[start:start + settings['chunk_size']]
        prompts = [ANALYSIS_PROMPT if config['prompt_mode'] == 'training' else r['prompt'] for r in chunk]
        generated = generator.generate(
            [r['fen'] for r in chunk], prompts, [r.get('history') or [] for r in chunk],
            temperature=settings['temperature'], top_k=settings['top_k'], top_p=settings['top_p'],
            max_tokens=settings['max_tokens'], seed=settings['seed'],
        )
        for row, result in zip(chunk, generated, strict=True):
            rows.append(dict(sample_id=row['sample_id'], generation=result['text'],
                token_count=len(result['token_ids']), hit_token_cap=len(result['token_ids']) >= settings['max_tokens'],
                critical_line=parse_critical_line(result['text'], row['fen'])))
        atomic_json(path, rows)
        print(f'[generation] {len(rows)}/{len(sample)}', flush=True)


def score_moves(tasks, config, worker):
    results = {}
    # scoring: one persistent, single-threaded Stockfish per worker; best and played searches share a root.
    with chess.engine.SimpleEngine.popen_uci(config['stockfish']) as engine:
        engine.configure({'Threads': 1, 'Hash': 32})
        limit = chess.engine.Limit(nodes=config['sf_nodes'])
        for index, (fen, uci) in enumerate(tasks, 1):
            board = chess.Board(fen)
            move = chess.Move.from_uci(uci)
            best = engine.analyse(board, limit)['score'].pov(board.turn).score(mate_score=10000)
            played = engine.analyse(board, limit, root_moves=[move])['score'].pov(board.turn).score(mate_score=10000)
            if best is None or played is None:
                raise ValueError(f'Stockfish returned no score: {fen} {uci}')
            drop = win_rate(best) - win_rate(played)
            results[(fen, uci)] = dict(best_cp=best, played_cp=played, win_rate_drop=round(drop, 4),
                                       mistake=drop >= config['mistake_wr_drop'])
            if index % 10 == 0 or index == len(tasks):
                print(f'[stockfish worker {worker}] {index}/{len(tasks)}', flush=True)
        engine.quit()
    return results


def score(config, sample, output):
    generated = json.loads((output / 'generations.json').read_text())
    if [r['sample_id'] for r in generated] != [r['sample_id'] for r in sample]:
        raise ValueError('Generation is incomplete or has mismatched sample IDs')
    tasks = set()
    for row, result in zip(sample, generated, strict=True):
        line = result['critical_line']
        board = chess.Board(row['fen'])
        side = board.turn
        # The extra first-move metric can be scored even if a later ply is illegal.
        for ply, uci in enumerate(line['uci']):
            if board.turn == side and (line['valid'] or ply == 0):
                tasks.add((board.fen(), uci))
            board.push_uci(uci)
    path = output / 'sf_scores.json'
    cached = json.loads(path.read_text()) if path.exists() else []
    scores = {(r['fen'], r['move']): {k: v for k, v in r.items() if k not in ('fen', 'move')} for r in cached}
    missing = sorted(tasks - scores.keys())
    workers = min(config['sf_workers'], len(missing))
    if workers:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(score_moves, missing[w::workers], config, w) for w in range(workers)]
            for future in as_completed(futures):
                scores.update(future.result())
                atomic_json(path, [dict(fen=k[0], move=k[1], **v) for k, v in sorted(scores.items())])

    # metrics: opponent mistakes do not fail STM accuracy; exact match also accepts prefixes/suffixes.
    details = []
    for row, result in zip(sample, generated, strict=True):
        line = result['critical_line']
        moves = line['uci']
        valid = bool(line['valid'] and moves)
        board = chess.Board(row['fen'])
        side = board.turn
        move_scores = []
        if valid:
            for ply, uci in enumerate(moves):
                if board.turn == side:
                    move_scores.append(dict(ply=ply, fen=board.fen(), move=uci, **scores[(board.fen(), uci)]))
                board.push_uci(uci)
        first_safe = bool(moves and not scores[(chess.Board(row['fen']).fen(), moves[0])]['mistake'])
        details.append(dict(**row, critical_line_uci=moves, critical_line_valid=valid,
            critical_line_error=line['error'], first_move_correct=bool(moves and moves[0] == row['correct_move_uci']),
            first_move_no_mistake=first_safe, side_to_move_correct=valid and not any(s['mistake'] for s in move_scores),
            exact_prefix_suffix_correct=valid and any(_prefix_suffix(moves, row['solution_uci'])),
            puzzle_side_move_scores=move_scores))
    n = len(sample)
    metrics = dict(n=n, model=config['model'], label=config['label'],
        first_move_accuracy=sum(r['first_move_correct'] for r in details) / n,
        first_move_no_mistake_rate=sum(r['first_move_no_mistake'] for r in details) / n,
        side_to_move_accuracy=sum(r['side_to_move_correct'] for r in details) / n,
        exact_prefix_suffix_accuracy=sum(r['exact_prefix_suffix_correct'] for r in details) / n,
        valid_line_rate=sum(r['critical_line_valid'] for r in details) / n,
        token_cap_rate=sum(r['hit_token_cap'] for r in generated) / n,
        mean_generated_tokens=sum(r['token_count'] for r in generated) / n)
    with (output / 'per_position.jsonl').open('w') as handle:
        for row in details:
            handle.write(json.dumps(row) + '\n')
    atomic_json(output / 'metrics.json', metrics)
    columns = [('First move', 'first_move_accuracy'), ('First move no mistake', 'first_move_no_mistake_rate'),
               ('STM no mistake', 'side_to_move_accuracy'), ('Exact prefix/suffix', 'exact_prefix_suffix_accuracy'),
               ('Valid line', 'valid_line_rate')]
    (output / 'results.md').write_text(
        '| Model | ' + ' | '.join(c[0] for c in columns) + ' |\n|---|' + '---:|' * len(columns) + '\n'
        + '| ' + config['label'] + ' | ' + ' | '.join(f'{100 * metrics[k]:.2f}%' for _, k in columns) + ' |\n')
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset-index', type=int, default=0)
    parser.add_argument('--phase', choices=('all', 'generate', 'score'), default='all')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config['prompt_mode'] not in ('sample', 'training') or config['sf_workers'] < 1:
        raise ValueError('Invalid prompt mode or worker count')
    if not 0 <= args.dataset_index < len(config['datasets']):
        raise ValueError('Invalid dataset index')
    sample, output = prepare(config, args.dataset_index)
    if args.phase == 'all':
        # A separate process releases vLLM/GPU memory before CPU scoring begins.
        subprocess.run([sys.executable, '-m', 'eval.benchmark', '--config', str(args.config),
                        '--dataset-index', str(args.dataset_index), '--phase', 'generate'], check=True)
    elif args.phase == 'generate':
        generate(config, sample, output)
        return
    score(config, sample, output)


if __name__ == '__main__':
    main()
