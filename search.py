import time
import math
import random
from evaluation import Evaluator

class Searcher:
    def __init__(self, evaluator=None, base_time=5*60, increment=0, show_thinking=False, depth_limit=4):
        self.evaluator = evaluator if evaluator else Evaluator()
        self.start_time = 0
        self.time_limit = base_time
        self.increment = increment
        self.show_thinking = show_thinking
        self.best_move = None
        self.nodes = 0
        self.root_scores = []
        self.depth_limit = depth_limit
        self._last_root_move = None

    def search(self, board, move_number=1):
        self.start_time = time.time()
        self.nodes = 0
        self.best_move = None
        self.root_scores = []
        # reset repetition guard if side to move changed
        if hasattr(self, '_last_side') and self._last_side != board.white_to_move:
            self._last_root_move = None
        self._last_side = board.white_to_move
        depth = 1

        time_per_move = (self.time_limit + self.increment * (move_number - 1)) / 40  # Assume ~40 moves
        self.time_limit = time_per_move

        while depth <= self.depth_limit:
            try:
                move = self.iterative_deepening(board, depth)
                if move is not None:
                    self.best_move = move
                depth += 1
            except TimeoutError:
                break

        if self.show_thinking and self.best_move:
            elapsed = time.time() - self.start_time
            nps = int(self.nodes / elapsed) if elapsed else 0
            print(f"Searched {self.nodes} nodes in {elapsed:.2f}s  ({nps} nps)")

        # --- fallback: ensure we always have a legal move ---
        if self.best_move is None and self.root_scores:
            # Pick the top‑scoring move if, for some reason, none was chosen
            self.root_scores.sort(key=lambda x: x[0], reverse=board.white_to_move)
            self.best_move = self.root_scores[0][1]
        # ----------------------------------------------------

        # ─── avoid immediate repetition (e.g. knight bouncing) ──
        if self.best_move == self._last_root_move and len(self.root_scores) > 1:
            # pick the next‑best move
            alt_scores = sorted(self.root_scores,
                                key=lambda x: x[0],
                                reverse=board.white_to_move)
            for s, m in alt_scores:
                if m != self.best_move:
                    self.best_move = m
                    break
        # remember for the next call
        self._last_root_move = self.best_move

        return self.best_move

    def iterative_deepening(self, board, max_depth):
        moves = board.generate_legal_moves()
        move_scores = []
        for move in moves:
            try:
                board.make_move(move)
            except ValueError:
                # Defensive: a malformed move slipped through generation – skip it
                continue
            try:
                score = self.minimax(board, max_depth - 1,
                                     -math.inf, math.inf,
                                     not board.white_to_move)
            except TimeoutError:
                # undo only if the move was actually pushed
                if board._state_stack:
                    board.undo_move(move)
                raise
            except ValueError:
                # Malformed move encountered deeper in the tree – skip it
                board.undo_move(move)
                continue
            board.undo_move(move)
            move_scores.append((score, move))
            if self.out_of_time():
                raise TimeoutError

        # Save root move scores for external access (e.g., for display)
        self.root_scores = move_scores.copy()

        best_move = None
        if move_scores:
            move_scores.sort(key=lambda x: x[0], reverse=board.white_to_move)
            best_cp = move_scores[0][0]
            threshold = best_cp - 25 if board.white_to_move else best_cp + 25
            near = [m for s, m in move_scores if (s >= threshold if board.white_to_move else s <= threshold)]
            best_move = random.choice(near)

        # Remove depth-by-depth progress spam (no print here)

        return best_move

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        self.nodes += 1
        if self.out_of_time():
            raise TimeoutError

        if depth == 0 or board.is_checkmate() or board.is_stalemate():
            return self.evaluator.evaluate(board)

        moves = board.generate_legal_moves()

        if maximizing_player:
            max_eval = -math.inf
            for move in moves:
                try:
                    board.make_move(move)
                except ValueError:
                    # malformed move – just skip it
                    continue
                try:
                    eval = self.minimax(board, depth - 1, alpha, beta, False)
                except TimeoutError:
                    # ensure board state is restored before propagating
                    board.undo_move(move)
                    raise
                finally:
                    # normal path: always restore the position
                    if board._state_stack:
                        board.undo_move(move)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in moves:
                try:
                    board.make_move(move)
                except ValueError:
                    # malformed move – just skip it
                    continue
                try:
                    eval = self.minimax(board, depth - 1, alpha, beta, True)
                except TimeoutError:
                    board.undo_move(move)
                    raise
                finally:
                    if board._state_stack:
                        board.undo_move(move)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def out_of_time(self):
        return time.time() - self.start_time > self.time_limit

## if __name__ == "__main__":  # [Commented out Searcher test block]
##     # from board import Board
##
##     board = Board()
##     searcher = Searcher(base_time=5*60, increment=3, show_thinking=True)  # Example: 5 minutes base, 3 seconds increment
##     move_number = 1
##
##     while True:
##         move = searcher.search(board, move_number)
##         if move:
##             print("Best Move Found:", move)
##             board.make_move(move)
##             board.print_board()
##             move_number += 1
##         else:
##             print("No move found.")
##             break


