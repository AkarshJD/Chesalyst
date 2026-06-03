import numpy as np


class Evaluator:
    def __init__(self):
        self.piece_values = {
            'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
            'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000,
        }

        # Piece-square tables oriented for white (rank 0 = rank 8 in chess)
        self.pst = {
            'P': np.array([
                0,   0,   0,   0,   0,   0,   0,   0,
                5,  10,  10, -20, -20,  10,  10,   5,
                5,  -5, -10,   0,   0, -10,  -5,   5,
                0,   0,   0,  20,  20,   0,   0,   0,
                5,   5,  10,  25,  25,  10,   5,   5,
               10,  10,  20,  30,  30,  20,  10,  10,
               50,  50,  50,  50,  50,  50,  50,  50,
                0,   0,   0,   0,   0,   0,   0,   0,
            ]),
            'N': np.array([
               -50, -40, -30, -30, -30, -30, -40, -50,
               -40, -20,   0,   5,   5,   0, -20, -40,
               -30,   5,  10,  15,  15,  10,   5, -30,
               -30,   0,  15,  20,  20,  15,   0, -30,
               -30,   5,  15,  20,  20,  15,   5, -30,
               -30,   0,  10,  15,  15,  10,   0, -30,
               -40, -20,   0,   0,   0,   0, -20, -40,
               -50, -40, -30, -30, -30, -30, -40, -50,
            ]),
            'B': np.array([
               -20, -10, -10, -10, -10, -10, -10, -20,
               -10,   5,   0,   0,   0,   0,   5, -10,
               -10,  10,  10,  10,  10,  10,  10, -10,
               -10,   0,  10,  10,  10,  10,   0, -10,
               -10,   5,   5,  10,  10,   5,   5, -10,
               -10,   0,   5,  10,  10,   5,   0, -10,
               -10,   0,   0,   0,   0,   0,   0, -10,
               -20, -10, -10, -10, -10, -10, -10, -20,
            ]),
            'R': np.zeros(64, dtype=int),
            'Q': np.zeros(64, dtype=int),
            'K': np.array([
                20,  30,  10,   0,   0,  10,  30,  20,
                20,  20,   0,   0,   0,   0,  20,  20,
               -10, -20, -20, -20, -20, -20, -20, -10,
               -20, -30, -30, -40, -40, -30, -30, -20,
               -30, -40, -40, -50, -50, -40, -40, -30,
               -30, -40, -40, -50, -50, -40, -40, -30,
               -30, -40, -40, -50, -50, -40, -40, -30,
               -30, -40, -40, -50, -50, -40, -40, -30,
            ]),
        }

    def evaluate(self, board):
        score = 0
        for piece, bb in board.bitboards.items():
            sign = 1 if piece.isupper() else -1
            for i in range(64):
                if (bb >> np.uint64(i)) & np.uint64(1):
                    score += self.piece_values[piece]
                    # Mirror the PST index vertically for black so white-oriented
                    # tables apply correctly (XOR 56 flips rank within same file)
                    pst_idx = i if piece.isupper() else (i ^ 56)
                    score += int(self.pst.get(piece.upper(), np.zeros(64))[pst_idx]) * sign

        score += self._pawn_structure(board)
        score += 10 if board.white_to_move else -10
        return score

    def _pawn_structure(self, board):
        score = 0
        white_pawns = board.bitboards['P']
        black_pawns = board.bitboards['p']
        file_balance = [0] * 8

        for i in range(64):
            f = i % 8
            if (white_pawns >> np.uint64(i)) & np.uint64(1):
                file_balance[f] += 1
                if self._is_passed_pawn(i, white_pawns, black_pawns, True):
                    score += 30
            if (black_pawns >> np.uint64(i)) & np.uint64(1):
                file_balance[f] -= 1
                if self._is_passed_pawn(i, black_pawns, white_pawns, False):
                    score -= 30

        for f in range(8):
            count = file_balance[f]
            if count > 1:
                score -= 20 * (count - 1)
            elif count < -1:
                score += 20 * (count + 1)
            # Isolated pawn penalty
            left_empty = (f == 0 or file_balance[f - 1] == 0)
            right_empty = (f == 7 or file_balance[f + 1] == 0)
            if count == 1 and left_empty and right_empty:
                score -= 15
            if count == -1 and left_empty and right_empty:
                score += 15

        return score

    def _is_passed_pawn(self, square, own_pawns, enemy_pawns, white):
        file = square % 8
        rank = square // 8
        files = [f for f in [file - 1, file, file + 1] if 0 <= f <= 7]
        rank_range = range(rank + 1, 8) if white else range(rank - 1, -1, -1)
        for r in rank_range:
            for f in files:
                if (enemy_pawns >> np.uint64(r * 8 + f)) & np.uint64(1):
                    return False
        return True
