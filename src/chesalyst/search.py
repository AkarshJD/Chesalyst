import math
import random
import time

from chesalyst.evaluation import Evaluator

# ---------------------------------------------------------------------- #
#  Transposition table                                                     #
# ---------------------------------------------------------------------- #

TT_EXACT = 0
TT_LOWER = 1  # fail-high: stored score is a lower bound on the true value
TT_UPPER = 2  # fail-low:  stored score is an upper bound on the true value

# Piece values used only for MVV-LVA capture ordering (not evaluation).
_MVV_VAL = {'P':1,'N':3,'B':3,'R':5,'Q':9,'K':0,'p':1,'n':3,'b':3,'r':5,'q':9,'k':0}


def _mvv_lva(move, bb):
    """Score a capture for ordering: higher = search first (best winning trade first)."""
    if move.promotion:
        return 90            # promotions before all captures
    if move.en_passant:
        return 9             # pawn x pawn
    victim = 0
    cap_mask = 1 << move.to_square
    for p in ('Q', 'q', 'R', 'r', 'B', 'b', 'N', 'n', 'P', 'p'):
        if int(bb[p]) & cap_mask:
            victim = _MVV_VAL[p]
            break
    attacker = 1
    from_mask = 1 << move.from_square
    for p in ('P', 'p', 'N', 'n', 'B', 'b', 'R', 'r', 'Q', 'q', 'K', 'k'):
        if int(bb[p]) & from_mask:
            attacker = _MVV_VAL[p] or 1
            break
    return 10 * victim - attacker


class TranspositionTable:
    def __init__(self, size=1 << 20):
        self._mask = size - 1
        self._table = [None] * size

    def probe(self, key, depth, alpha, beta):
        """Return a usable score if there's a hit, else None."""
        entry = self._table[key & self._mask]
        if entry is None or entry[0] != key or entry[1] < depth:
            return None
        score, flag = entry[2], entry[3]
        if flag == TT_EXACT:
            return score
        if flag == TT_LOWER and score >= beta:
            return score
        if flag == TT_UPPER and score <= alpha:
            return score
        return None

    def probe_move(self, key):
        """Return the best move recorded for this position, or None."""
        entry = self._table[key & self._mask]
        if entry is not None and entry[0] == key:
            return entry[4]
        return None

    def store(self, key, depth, score, flag, best_move=None):
        idx = key & self._mask
        entry = self._table[idx]
        if entry is None or depth >= entry[1]:
            self._table[idx] = (key, depth, score, flag, best_move)

    def clear(self):
        self._table = [None] * (self._mask + 1)


class Searcher:
    def __init__(self, evaluator=None, base_time=300, increment=0,
                 show_thinking=False, depth_limit=4, tt_size=1 << 20):
        self.evaluator = evaluator or Evaluator()
        self.base_time = base_time        # total clock time (seconds)
        self.increment = increment        # per-move increment (seconds)
        self.show_thinking = show_thinking
        self.depth_limit = depth_limit
        self.tt = TranspositionTable(tt_size)
        self._killers = [[None, None] for _ in range(64)]

        self.nodes = 0
        self.best_move = None
        self.root_scores = []

        self._start_time = 0.0
        self._move_time = 0.0            # budget for the current move
        self._last_root_move = None
        self._last_side = None

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def search(self, board, move_number=1):
        self._start_time = time.time()
        self.nodes = 0
        self.best_move = None
        self.root_scores = []
        self._killers = [[None, None] for _ in range(64)]

        # Reset repetition guard when the side to move changes
        if self._last_side is not None and self._last_side != board.white_to_move:
            self._last_root_move = None
        self._last_side = board.white_to_move

        # Allocate time for this move without mutating base_time
        self._move_time = (self.base_time + self.increment * (move_number - 1)) / 40

        prev_score = 0  # score from the last completed depth
        best = None
        for depth in range(1, self.depth_limit + 1):
            try:
                if depth <= 2:
                    # Scores too unstable at shallow depths — use full window
                    move, score = self._search_at_depth(board, depth)
                    if move is not None:
                        best = move
                        prev_score = score
                else:
                    # Aspiration window: start narrow, widen exponentially on failure
                    delta = 25
                    alpha = prev_score - delta
                    beta  = prev_score + delta
                    while True:
                        move, score = self._search_at_depth(board, depth, alpha, beta)
                        if move is None:
                            break
                        if score <= alpha:
                            # Fail-low: true score is below our window — widen downward
                            alpha = max(score - delta, -math.inf)
                            delta *= 2
                        elif score >= beta:
                            # Fail-high: true score is above our window — widen upward
                            beta = min(score + delta, math.inf)
                            delta *= 2
                        else:
                            best = move
                            prev_score = score
                            break
            except TimeoutError:
                break

        self.best_move = best

        if self.show_thinking and self.best_move:
            elapsed = time.time() - self._start_time
            nps = int(self.nodes / elapsed) if elapsed else 0
            print(f"Searched {self.nodes} nodes in {elapsed:.2f}s ({nps} nps)")

        # Fallback: if iterative deepening timed out before any move was picked
        if self.best_move is None and self.root_scores:
            self.root_scores.sort(key=lambda x: x[0], reverse=board.white_to_move)
            self.best_move = self.root_scores[0][1]

        # Avoid immediate repetition (e.g. knight bouncing back and forth)
        if self.best_move == self._last_root_move and len(self.root_scores) > 1:
            alt = sorted(self.root_scores, key=lambda x: x[0], reverse=board.white_to_move)
            for _, m in alt:
                if m != self.best_move:
                    self.best_move = m
                    break

        self._last_root_move = self.best_move
        return self.best_move

    # ------------------------------------------------------------------ #
    #  Iterative deepening root                                            #
    # ------------------------------------------------------------------ #

    def _search_at_depth(self, board, max_depth, alpha=-math.inf, beta=math.inf):
        moves = board.generate_legal_moves()
        move_scores = []

        for move in moves:
            board.make_move(move)
            try:
                score = self._minimax(board, max_depth - 1,
                                      alpha, beta,
                                      board.white_to_move, ply=1)
            except TimeoutError:
                board.undo_move(move)
                raise
            board.undo_move(move)
            move_scores.append((score, move))
            if self._out_of_time():
                raise TimeoutError

        self.root_scores = move_scores.copy()

        if not move_scores:
            return None, 0

        move_scores.sort(key=lambda x: x[0], reverse=board.white_to_move)
        best_score = move_scores[0][0]
        # Add a small random flair: pick from moves within 25cp of the best
        threshold = best_score - 25 if board.white_to_move else best_score + 25
        candidates = [m for s, m in move_scores
                      if (s >= threshold if board.white_to_move else s <= threshold)]
        return random.choice(candidates), best_score

    # ------------------------------------------------------------------ #
    #  Alpha-beta minimax                                                  #
    # ------------------------------------------------------------------ #

    def _minimax(self, board, depth, alpha, beta, maximizing, ply=0, null_move_ok=True):
        self.nodes += 1
        if self._out_of_time():
            raise TimeoutError

        # Draw by 50-move rule or repetition (second occurrence = draw available)
        if board.halfmove_clock >= 100:
            return 0
        if board.zobrist_hash in board._hash_history:
            return 0

        if depth == 0:
            return self._quiescence(board, alpha, beta, maximizing)

        tt_score = self.tt.probe(board.zobrist_hash, depth, alpha, beta)
        if tt_score is not None:
            return tt_score

        tt_move = self.tt.probe_move(board.zobrist_hash)

        # Null move pruning — skip our turn; if even that gives score >= beta, prune.
        # Guards: not in check, has non-pawn material (avoids zugzwang), no consecutive nulls.
        if (null_move_ok
                and depth >= 3
                and not board.is_check()
                and self._has_non_pawn_material(board)):
            R = 3 if depth >= 6 else 2
            board.make_null_move()
            try:
                null_score = self._minimax(board, depth - 1 - R, alpha, beta,
                                           not maximizing, ply + 1, null_move_ok=False)
            finally:
                board.undo_null_move()
            if maximizing and null_score >= beta:
                return null_score
            if not maximizing and null_score <= alpha:
                return null_score

        moves = board.generate_legal_moves()

        if not moves:
            if board.is_check():
                return (-50000 + depth) if board.white_to_move else (50000 - depth)
            return 0

        moves = self._order_moves(board, moves, ply, tt_move)
        original_alpha = alpha
        original_beta = beta
        best_move = None

        if maximizing:
            best = -math.inf
            for i, move in enumerate(moves):
                is_quiet = not (move.capture or move.en_passant or move.promotion)
                board.make_move(move)
                try:
                    if i >= 3 and depth >= 3 and is_quiet:
                        R = max(1, int(math.log(depth) * math.log(i + 1) / 2.5))
                        score = self._minimax(board, depth - 1 - R, alpha, beta, False, ply + 1)
                        if score > alpha:
                            score = self._minimax(board, depth - 1, alpha, beta, False, ply + 1)
                    else:
                        score = self._minimax(board, depth - 1, alpha, beta, False, ply + 1)
                finally:
                    board.undo_move(move)
                if score > best:
                    best = score
                    best_move = move
                if score > alpha:
                    alpha = score
                if beta <= alpha:
                    if is_quiet:
                        self._update_killers(move, ply)
                    break
            flag = TT_LOWER if best >= beta else (TT_EXACT if best > original_alpha else TT_UPPER)
            self.tt.store(board.zobrist_hash, depth, best, flag, best_move)
            return best
        else:
            best = math.inf
            for i, move in enumerate(moves):
                is_quiet = not (move.capture or move.en_passant or move.promotion)
                board.make_move(move)
                try:
                    if i >= 3 and depth >= 3 and is_quiet:
                        R = max(1, int(math.log(depth) * math.log(i + 1) / 2.5))
                        score = self._minimax(board, depth - 1 - R, alpha, beta, True, ply + 1)
                        if score < beta:
                            score = self._minimax(board, depth - 1, alpha, beta, True, ply + 1)
                    else:
                        score = self._minimax(board, depth - 1, alpha, beta, True, ply + 1)
                finally:
                    board.undo_move(move)
                if score < best:
                    best = score
                    best_move = move
                if score < beta:
                    beta = score
                if beta <= alpha:
                    if is_quiet:
                        self._update_killers(move, ply)
                    break
            flag = TT_UPPER if best <= alpha else (TT_EXACT if best < original_beta else TT_LOWER)
            self.tt.store(board.zobrist_hash, depth, best, flag, best_move)
            return best

    def _order_moves(self, board, moves, ply, tt_move):
        """TT move → captures/promotions (MVV-LVA) → killers → quiet."""
        tt_list, captures, killers, quiet = [], [], [], []
        k = self._killers[ply] if ply < 64 else (None, None)
        k1, k2 = k[0], k[1]

        for move in moves:
            if tt_move is not None and move == tt_move:
                tt_list.append(move)
            elif move.capture or move.en_passant or move.promotion:
                captures.append(move)
            elif move == k1 or move == k2:
                killers.append(move)
            else:
                quiet.append(move)

        captures.sort(key=lambda m: _mvv_lva(m, board.bitboards), reverse=True)
        return tt_list + captures + killers + quiet

    def _update_killers(self, move, ply):
        if ply >= 64:
            return
        k = self._killers[ply]
        if k[0] != move:
            k[1] = k[0]
            k[0] = move

    def _quiescence(self, board, alpha, beta, maximizing):
        self.nodes += 1
        if self._out_of_time():
            raise TimeoutError

        in_check = board.is_check()
        stand_pat = self.evaluator.evaluate(board)

        if not in_check:
            if maximizing:
                if stand_pat >= beta:
                    return stand_pat
                if stand_pat > alpha:
                    alpha = stand_pat
            else:
                if stand_pat <= alpha:
                    return stand_pat
                if stand_pat < beta:
                    beta = stand_pat

        moves = board.generate_legal_moves() if in_check else board.generate_legal_captures()

        if not moves:
            if in_check:
                return -50000 if maximizing else 50000
            return stand_pat

        if maximizing:
            best = -math.inf if in_check else stand_pat
            for move in moves:
                board.make_move(move)
                try:
                    score = self._quiescence(board, alpha, beta, False)
                finally:
                    board.undo_move(move)
                best = max(best, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return best
        else:
            best = math.inf if in_check else stand_pat
            for move in moves:
                board.make_move(move)
                try:
                    score = self._quiescence(board, alpha, beta, True)
                finally:
                    board.undo_move(move)
                best = min(best, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return best

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _has_non_pawn_material(self, board):
        """True if the side to move has at least one piece beyond king and pawns."""
        bb = board.bitboards
        if board.white_to_move:
            return bool(int(bb['N']) | int(bb['B']) | int(bb['R']) | int(bb['Q']))
        else:
            return bool(int(bb['n']) | int(bb['b']) | int(bb['r']) | int(bb['q']))

    def _out_of_time(self):
        return time.time() - self._start_time > self._move_time
