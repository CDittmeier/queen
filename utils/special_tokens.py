"""Chess special-token support (all the token-adding machinery).

Whether tokens are added is decided automatically: the answer-token set matching
``args.pov`` is probed for its first token. If the tokenizer already contains it,
the tokenizer is assumed to be a chess-resized one that already has every answer
token and nothing is added; otherwise the whole set is added and embeddings are
trained for them.

  * maybe_add_special_tokens     — adds the answer-token set to the tokenizer
                                   *before* the model is built (so n_new_tokens
                                   is known), unless they are already present.
  * maybe_init_special_token_embeddings — semantically initializes the new
                                   embeddings *after* the model is built.

Which token set is used (POV vs board-absolute) is selected by ``args.pov``,
which the trainer sources from the dataset's ``dataset_config.json``.
"""
import chess
import torch

from models.base import unwrap_decoder
from utils.utils import (
    ANSWER_SPECIAL_TOKENS,
    EMPTY_TOKEN,
    POV_ANSWER_SPECIAL_TOKENS,
    POV_SQUARE_TOKENS,
    SQUARE_TOKENS,
    _PIECE_TO_POV_TOKEN,
    _PIECE_TO_TOKEN,
)

_COLOR_WORDS = {chess.WHITE: "white", chess.BLACK: "black"}
_POV_SIDE_WORDS = {True: "mine", False: "opp"}
_PIECE_WORDS = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}


def maybe_add_special_tokens(tokenizer, args) -> int:
    """Add the chess answer-token set matching ``args.pov`` to ``tokenizer``,
    unless the tokenizer already contains them.

    The decision probes the first token of the set: if it is already in the
    vocabulary the tokenizer is assumed to already hold the full chess answer
    vocabulary (a resized model), so nothing is added; otherwise the whole set
    is added so embeddings can be trained for them.

    Returns the number of tokens added (0 when already present). Call this BEFORE
    building the model so the model can size its new-token embeddings.
    """
    tok_set = POV_ANSWER_SPECIAL_TOKENS if args.pov else ANSWER_SPECIAL_TOKENS
    if tok_set[0] in tokenizer.get_vocab():
        return 0
    before = len(tokenizer)
    tokenizer.add_tokens(tok_set, special_tokens=True)
    return len(tokenizer) - before


def _mean_embedding(embed_weight: torch.Tensor, tokenizer, text: str) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return embed_weight[ids].float().mean(dim=0)


def maybe_init_special_token_embeddings(model, tokenizer, args) -> None:
    """Semantically initialize the newly added token embeddings.

    No-op when no tokens were added (already present) or embed_init='random'.
    Reads from the frozen pretrained embeddings; never modifies them.

    semantic init (POV mode, args.pov=True):
      POV SQUARE tokens ← mean(file_char, rank_char) of board square i (POV
        index i maps to board square i, the white-POV correspondence — not
        strictly meaningful for black-to-move but adequate as initialization)
      POV piece tokens  ← mean(side_word, piece_word) where side_word ∈
        {"mine", "opp"}
      EMPTY token       ← embedding of 'empty'

    semantic init (absolute mode, args.pov=False):
      SQUARE tokens     ← mean(file_char, rank_char) of the literal board sq
      piece tokens      ← mean(color_word, piece_word) with color ∈
        {"white", "black"}
      EMPTY token       ← embedding of 'empty'
    """
    if (model.n_new_tokens == 0
            or getattr(args, "embed_init", "semantic") == "random"):
        return

    base_decoder = unwrap_decoder(model.decoder)
    frozen_w = base_decoder.model.embed_tokens.weight.data
    new_emb_w = model.new_embed.weight.data
    frozen_vocab = base_decoder.config.vocab_size

    sq_semantic = {}
    for sq in chess.SQUARES:
        name = chess.square_name(sq)
        sq_semantic[sq] = (
            _mean_embedding(frozen_w, tokenizer, name[0])
            + _mean_embedding(frozen_w, tokenizer, name[1])
        ) / 2.0

    # chess.WHITE == True and chess.BLACK == False, so _PIECE_TO_POV_TOKEN's
    # (is_mine, ptype) keys and _PIECE_TO_TOKEN's (color, ptype) keys share
    # the same shape — same for _POV_SIDE_WORDS vs _COLOR_WORDS. One loop body
    # handles both modes after selecting the matching trio.
    sq_tokens, piece_map, side_words = (
        (POV_SQUARE_TOKENS, _PIECE_TO_POV_TOKEN, _POV_SIDE_WORDS) if args.pov
        else (SQUARE_TOKENS, _PIECE_TO_TOKEN, _COLOR_WORDS)
    )
    for sq, tok in zip(chess.SQUARES, sq_tokens):
        idx = tokenizer.convert_tokens_to_ids(tok) - frozen_vocab
        new_emb_w[idx] = sq_semantic[sq].to(new_emb_w.dtype)

    for (side_key, ptype), tok in piece_map.items():
        avg = (
            _mean_embedding(frozen_w, tokenizer, side_words[side_key])
            + _mean_embedding(frozen_w, tokenizer, _PIECE_WORDS[ptype])
        ) / 2.0
        idx = tokenizer.convert_tokens_to_ids(tok) - frozen_vocab
        new_emb_w[idx] = avg.to(new_emb_w.dtype)

    empty_idx = tokenizer.convert_tokens_to_ids(EMPTY_TOKEN) - frozen_vocab
    new_emb_w[empty_idx] = _mean_embedding(frozen_w, tokenizer, "empty").to(new_emb_w.dtype)
