"""Shared, weight-free structural precompute for one position.

This is the genuinely-global half of Stockfish's evaluation: the attack tables,
king ring, mobility areas, pawn spans, and per-piece / per-pawn RAW structural
records. It mirrors evaluate.cpp's initialize() and the attack-accumulating half
of pieces(), plus pawns.cpp's per-pawn flagging -- but applies NO scoring weights.
Individual evaluators read this context and apply their own constant tables.

Square indexing matches python-chess and SF (a1=0, h8=63).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess

from datagen.tree.primitives.core import bitboard as bb
from datagen.tree.primitives.core.score import non_pawn_material

PT_ORDER = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


@dataclass
class PawnRec:
    """Raw structural flags for one pawn (weights applied in pawns.py)."""
    sq: int
    color: chess.Color
    r: int                     # relative rank 0..7
    opposed: bool
    blocked: bool
    doubled: bool
    neighbours: bool
    phalanx: bool
    support: bool              # any supporting pawn
    support_count: int
    backward: bool
    lever_more_than_one: bool
    passed: bool


@dataclass
class PawnStruct:
    passed_pawns: int
    pawn_attacks: int
    pawn_attacks_span: int
    pawns: list = field(default_factory=list)


@dataclass
class PieceRec:
    """Raw structural record for one (non-pawn, non-king) piece."""
    color: chess.Color
    pt: int
    sq: int
    attacks: int               # attack bb (after king-blocker ray restriction)
    mob_count: int             # # mobility-area squares attacked


class Context:
    def __init__(self, board: chess.Board):
        self.board = board
        self.occ = int(board.occupied)
        # attack tables: att[c][pt] / att[c]['ALL'] / att2[c]
        self.att = {chess.WHITE: {}, chess.BLACK: {}}
        self.att2 = {}
        self.mobility_area = {}
        self.king_ring = {}
        self.kac = {chess.WHITE: 0, chess.BLACK: 0}            # king attackers count
        self.katt = {chess.WHITE: 0, chess.BLACK: 0}           # king-zone attacks count
        self.king_attacker_pts = {chess.WHITE: [], chess.BLACK: []}  # piece types (for weight)
        self.pawn = {c: self.pawns_mod(c) for c in (chess.WHITE, chess.BLACK)}
        self.pieces = []
        for c in (chess.WHITE, chess.BLACK):
            self.initialize(c)
        for c in (chess.WHITE, chess.BLACK):
            for pt in PT_ORDER:
                self.pieces_mod(c, pt)

    # -- pawn structure (pawns.cpp evaluate<>), flags only -------------------
    def pawns_mod(self, color: chess.Color) -> PawnStruct:
        board = self.board
        them = not color
        up = 8 if color == chess.WHITE else -8
        our = int(board.pawns & board.occupied_co[color])
        their = int(board.pawns & board.occupied_co[them])
        dbl_them = bb.pawn_double_attacks_bb(them, their)

        passed = 0
        attacks = bb.pawn_attacks_bb(color, our)
        span = attacks
        recs = []
        for s in chess.scan_forward(our):
            r = bb.rel_rank(color, s)
            opposed    = their & bb.forward_file(color, s)
            blocked    = their & chess.BB_SQUARES[s + up]
            stoppers   = their & bb.passed_pawn_span(color, s)
            lever      = their & chess.BB_PAWN_ATTACKS[color][s]
            leverPush  = their & chess.BB_PAWN_ATTACKS[color][s + up]
            doubled    = our & chess.BB_SQUARES[s - up]
            neighbours = our & bb.adjacent_files(s)
            phalanx    = neighbours & chess.BB_RANKS[s >> 3]
            support    = neighbours & chess.BB_RANKS[(s - up) >> 3]

            backward = (not (neighbours & bb.forward_ranks(them, s + up))) and bool(leverPush | blocked)
            if not backward and not blocked:
                span |= bb.pawn_attack_span(color, s)

            # deviation from SF11: a pawn whose stoppers currently lever it is NOT
            # passed (SF11's (stoppers ^ lever) == 0 counted it; pushing to stand
            # side-by-side with the lever would make it passed, but not yet)
            passed_flag = (
                stoppers == 0
                or ((stoppers ^ leverPush) == 0 and bb.popcount(phalanx) >= bb.popcount(leverPush))
                or (stoppers == blocked and r >= 4
                    and (bb.shift_up(color, support) & bb.bb_not(their | dbl_them)))
            )
            if passed_flag:
                passed |= chess.BB_SQUARES[s]

            recs.append(PawnRec(
                sq=s, color=color, r=r,
                opposed=bool(opposed), blocked=bool(blocked), doubled=bool(doubled),
                neighbours=bool(neighbours), phalanx=bool(phalanx),
                support=bool(support), support_count=bb.popcount(support),
                backward=bool(backward),
                lever_more_than_one=bb.popcount(lever) > 1,
                passed=bool(passed_flag)))
        return PawnStruct(passed, attacks, span, recs)

    # -- initialize() : attack seeds, mobility area, king ring ----------------
    def initialize(self, us: chess.Color):
        board, them = self.board, not us
        ksq = board.king(us)
        our_pawns = int(board.pawns & board.occupied_co[us])
        dbl_by_pawn = bb.pawn_double_attacks_bb(us, our_pawns)
        low = (chess.BB_RANKS[1] | chess.BB_RANKS[2]) if us == chess.WHITE \
            else (chess.BB_RANKS[6] | chess.BB_RANKS[5])
        b = our_pawns & (bb.shift_down(us, self.occ) | low)
        kq = (board.kings | board.queens) & board.occupied_co[us]
        self.mobility_area[us] = bb.bb_not(
            b | kq | bb.blockers_for_king(board, us) | self.pawn[them].pawn_attacks)

        self.att[us][chess.KING] = chess.BB_KING_ATTACKS[ksq]
        self.att[us][chess.PAWN] = self.pawn[us].pawn_attacks
        self.att[us]["ALL"] = self.att[us][chess.KING] | self.att[us][chess.PAWN]
        self.att2[us] = dbl_by_pawn | (self.att[us][chess.KING] & self.att[us][chess.PAWN])

        cf = min(max(ksq & 7, 1), 6)
        cr = min(max(ksq >> 3, 1), 6)
        cs = cr * 8 + cf
        kr = chess.BB_KING_ATTACKS[cs] | chess.BB_SQUARES[cs]
        self.kac[them] = bb.popcount(kr & self.pawn[them].pawn_attacks)
        self.king_ring[us] = kr & bb.bb_not(dbl_by_pawn)

    # -- pieces() : attack accumulation + per-piece records (no scoring) ------
    def pieces_mod(self, us: chess.Color, pt: int):
        board, them = self.board, not us
        ksq = board.king(us)
        occ = self.occ
        queens = int(board.queens)
        own_rooks = int(board.rooks & board.occupied_co[us])
        blockers = bb.blockers_for_king(board, us)
        self.att[us][pt] = 0

        for s in chess.scan_forward(board.pieces_mask(pt, us)):
            if pt == chess.BISHOP:
                b = bb.bishop_attacks(s, occ ^ queens)
            elif pt == chess.ROOK:
                b = bb.rook_attacks(s, occ ^ queens ^ own_rooks)
            elif pt == chess.QUEEN:
                b = bb.rook_attacks(s, occ) | bb.bishop_attacks(s, occ)
            else:  # knight
                b = chess.BB_KNIGHT_ATTACKS[s]
            if blockers & chess.BB_SQUARES[s]:
                b &= chess.ray(ksq, s)

            self.att2[us] |= self.att[us]["ALL"] & b
            self.att[us][pt] |= b
            self.att[us]["ALL"] |= b

            if b & self.king_ring[them]:
                self.kac[us] += 1
                self.king_attacker_pts[us].append(pt)
                self.katt[us] += bb.popcount(b & self.att[them][chess.KING])

            mob = bb.popcount(b & self.mobility_area[us])
            self.pieces.append(PieceRec(us, pt, s, b, mob))

    # -- structural helpers used by evaluators -------------------------------
    def piece_recs(self, color: chess.Color, pt: int):
        return [p for p in self.pieces if p.color == color and p.pt == pt]

    def piece_recs_ordered(self, pt: int):
        return [p for p in self.pieces if p.pt == pt]

    def non_pawn_material(self, color: chess.Color) -> int:
        return non_pawn_material(self.board, color)

    def pawn_passed(self, color: chess.Color, sq: int) -> bool:
        enemy = int(self.board.pawns & self.board.occupied_co[not color])
        return not (enemy & bb.passed_pawn_span(color, sq))

    def weak_queen(self, color: chess.Color, qsq: int) -> bool:
        """True if an enemy rook/bishop relatively pins or x-rays the queen."""
        board, them = self.board, not color
        occ = self.occ
        qr, qf = qsq >> 3, qsq & 7
        snipers = 0
        for sq in chess.scan_forward(board.rooks & board.occupied_co[them]):
            if (sq >> 3) == qr or (sq & 7) == qf:
                snipers |= chess.BB_SQUARES[sq]
        for sq in chess.scan_forward(board.bishops & board.occupied_co[them]):
            if abs((sq >> 3) - qr) == abs((sq & 7) - qf):
                snipers |= chess.BB_SQUARES[sq]
        occupancy = occ ^ snipers
        for sq in chess.scan_forward(snipers):
            b = chess.between(qsq, sq) & occupancy
            if b and (b & (b - 1)) == 0:
                return True
        return False


CONTEXT_CACHE: dict[str, Context] = {}


def get_context(board: chess.Board) -> Context:
    key = board.fen()
    ctx = CONTEXT_CACHE.get(key)
    if ctx is None:
        ctx = Context(board)
        CONTEXT_CACHE[key] = ctx
    return ctx
