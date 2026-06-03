import math
import random
import time

from chesalyst.evaluation import Evaluator


class Searcher:
    def __init__(self, evaluator=None, base_time=300, increment=0,
                 show_thinking=False, depth_limit=4):
        self.evaluator = evaluator or Evaluator()
        self.base_time = base_time        # total clock time (seconds)
        self.increment = increment        # per-move increment (seconds)
        self.show_thinking = show_thinking
        self.depth_limit = depth_limit

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

        # Reset repetition guard when the side to move changes
        if self._last_side is not None and self._last_side != board.white_to_move:
            self._last_root_move = None
        self._last_side = board.white_to_move

        # Allocate time for this move without mutating base_time
        self._move_time = (self.base_time + self.increment * (move_number - 1)) / 40

        best = None
        for depth in range(1, self.depth_limit + 1):
            try:
                move = self._search_at_depth(board, depth)
                if move is not None:
                    best = move
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

    def _search_at_depth(self, board, max_depth):
        moves = board.generate_legal_moves()
        move_scores = []

        for move in moves:
            board.make_move(move)
            try:
                score = self._minimax(board, max_depth - 1,
                                      -math.inf, math.inf,
                                      board.white_to_move)
            except TimeoutError:
                board.undo_move(move)
                raise
            board.undo_move(move)
            move_scores.append((score, move))
            if self._out_of_time():
                raise TimeoutError

        self.root_scores = move_scores.copy()

        if not move_scores:
            return None

        move_scores.sort(key=lambda x: x[0], reverse=board.white_to_move)
        best_score = move_scores[0][0]
        # Add a small random flair: pick from moves within 25cp of the best
        threshold = best_score - 25 if board.white_to_move else best_score + 25
        candidates = [m for s, m in move_scores
                      if (s >= threshold if board.white_to_move else s <= threshold)]
        return random.choice(candidates)

    # ------------------------------------------------------------------ #
    #  Alpha-beta minimax                                                  #
    # ------------------------------------------------------------------ #

    def _minimax(self, board, depth, alpha, beta, maximizing):
        self.nodes += 1
        if self._out_of_time():
            raise TimeoutError

        if depth == 0:
            return self._quiescence(board, alpha, beta, maximizing)

        moves = board.generate_legal_moves()

        if not moves:
            if board.is_check():
                return (-50000 + depth) if board.white_to_move else (50000 - depth)
            return 0

        if maximizing:
            best = -math.inf
            for move in moves:
                board.make_move(move)
                try:
                    score = self._minimax(board, depth - 1, alpha, beta, False)
                finally:
                    board.undo_move(move)
                best = max(best, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return best
        else:
            best = math.inf
            for move in moves:
                board.make_move(move)
                try:
                    score = self._minimax(board, depth - 1, alpha, beta, True)
                finally:
                    board.undo_move(move)
                best = min(best, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return best

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

    def _out_of_time(self):
        return time.time() - self._start_time > self._move_time
