import numpy as np
from chesalyst.board import _lsb_iter


def _popcount(bb):
    """Count set bits in a numpy uint64 or Python int."""
    return bin(int(bb)).count('1')


class Evaluator:
    def __init__(self):
        self.piece_values = {
            'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
            'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000,
        }

        # Piece-square tables as plain Python tuples (faster scalar lookup than numpy)
        self.pst = {
            'P': (
                0,   0,   0,   0,   0,   0,   0,   0,
                5,  10,  10, -20, -20,  10,  10,   5,
                5,  -5, -10,   0,   0, -10,  -5,   5,
                0,   0,   0,  20,  20,   0,   0,   0,
                5,   5,  10,  25,  25,  10,   5,   5,
               10,  10,  20,  30,  30,  20,  10,  10,
               50,  50,  50,  50,  50,  50,  50,  50,
                0,   0,   0,   0,   0,   0,   0,   0,
            ),
            'N': (
               -50, -40, -30, -30, -30, -30, -40, -50,
               -40, -20,   0,   5,   5,   0, -20, -40,
               -30,   5,  10,  15,  15,  10,   5, -30,
               -30,   0,  15,  20,  20,  15,   0, -30,
               -30,   5,  15,  20,  20,  15,   5, -30,
               -30,   0,  10,  15,  15,  10,   0, -30,
               -40, -20,   0,   0,   0,   0, -20, -40,
               -50, -40, -30, -30, -30, -30, -40, -50,
            ),
            'B': (
               -20, -10, -10, -10, -10, -10, -10, -20,
               -10,   5,   0,   0,   0,   0,   5, -10,
               -10,  10,  10,  10,  10,  10,  10, -10,
               -10,   0,  10,  10,  10,  10,   0, -10,
               -10,   5,   5,  10,  10,   5,   5, -10,
               -10,   0,   5,  10,  10,   5,   0, -10,
               -10,   0,   0,   0,   0,   0,   0, -10,
               -20, -10, -10, -10, -10, -10, -10, -20,
            ),
            'R': (0,) * 64,
            'Q': (0,) * 64,
            'K': (
                20,  30,  10,   0,   0,  10,  30,  20,
                20,  20,   0,   0,   0,   0,  20,  20,
               -10, -20, -20, -20, -20, -20, -20, -10,
               -20, -30, -30, -40, -40, -30, -30, -20,
               -30, -40, -40, -50, -50, -40, -40, -30,
               -30, -40, -40, -50, -50, -40, -40, -30,
               -30, -40, -40, -50, -50, -40, -40, -30,
               -30, -40, -40, -50, -50, -40, -40, -30,
            ),
        }

    def evaluate(self, board):
        score = 0
        bb = board.bitboards
        pst = self.pst
        pv = self.piece_values

        for piece in ('P', 'N', 'B', 'R', 'Q', 'K'):
            white_bb = bb[piece]
            black_bb = bb[piece.lower()]
            if white_bb:
                score += pv[piece] * _popcount(white_bb)
                pt = pst[piece]
                for i in _lsb_iter(white_bb):
                    score += pt[i]
            if black_bb:
                score += pv[piece.lower()] * _popcount(black_bb)
                pt = pst[piece]
                for i in _lsb_iter(black_bb):
                    score -= pt[i ^ 56]

        score += self._pawn_structure(board)
        score += 10 if board.white_to_move else -10
        return score

    def _pawn_structure(self, board):
        score = 0
        white_pawns = int(board.bitboards['P'])
        black_pawns = int(board.bitboards['p'])
        file_count = [0] * 8

        n = white_pawns
        while n:
            lsb = n & (-n)
            i = lsb.bit_length() - 1
            f = i & 7
            file_count[f] += 1
            if self._is_passed_pawn(i, white_pawns, black_pawns, True):
                score += 30
            n ^= lsb

        n = black_pawns
        while n:
            lsb = n & (-n)
            i = lsb.bit_length() - 1
            f = i & 7
            file_count[f] -= 1
            if self._is_passed_pawn(i, black_pawns, white_pawns, False):
                score -= 30
            n ^= lsb

        for f in range(8):
            count = file_count[f]
            if count > 1:
                score -= 20 * (count - 1)
            elif count < -1:
                score += 20 * (count + 1)
            left_empty = (f == 0 or file_count[f - 1] == 0)
            right_empty = (f == 7 or file_count[f + 1] == 0)
            if count == 1 and left_empty and right_empty:
                score -= 15
            if count == -1 and left_empty and right_empty:
                score += 15

        return score

    def _is_passed_pawn(self, square, own_pawns, enemy_pawns, white):
        file = square & 7
        rank = square >> 3
        rank_range = range(rank + 1, 8) if white else range(rank - 1, -1, -1)
        for r in rank_range:
            base = r << 3
            for f in range(max(0, file - 1), min(8, file + 2)):
                if enemy_pawns & (1 << (base + f)):
                    return False
        return True
