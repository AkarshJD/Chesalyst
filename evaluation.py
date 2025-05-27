import numpy as np

class Evaluator:
    def __init__(self):
        self.piece_values = {
            'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
            'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000
        }

        # Piece-square tables (higher values toward center or activity zones)
        self.pst = {
            'P': np.array([
                0, 0, 0, 0, 0, 0, 0, 0,
                5, 10, 10, -20, -20, 10, 10, 5,
                5, -5, -10, 0, 0, -10, -5, 5,
                0, 0, 0, 20, 20, 0, 0, 0,
                5, 5, 10, 25, 25, 10, 5, 5,
                10, 10, 20, 30, 30, 20, 10, 10,
                50, 50, 50, 50, 50, 50, 50, 50,
                0, 0, 0, 0, 0, 0, 0, 0
            ]),
            'N': np.array([
                -50, -40, -30, -30, -30, -30, -40, -50,
                -40, -20, 0, 5, 5, 0, -20, -40,
                -30, 5, 10, 15, 15, 10, 5, -30,
                -30, 0, 15, 20, 20, 15, 0, -30,
                -30, 5, 15, 20, 20, 15, 5, -30,
                -30, 0, 10, 15, 15, 10, 0, -30,
                -40, -20, 0, 0, 0, 0, -20, -40,
                -50, -40, -30, -30, -30, -30, -40, -50
            ]),
            'B': np.array([
                -20, -10, -10, -10, -10, -10, -10, -20,
                -10, 5, 0, 0, 0, 0, 5, -10,
                -10, 10, 10, 10, 10, 10, 10, -10,
                -10, 0, 10, 10, 10, 10, 0, -10,
                -10, 5, 5, 10, 10, 5, 5, -10,
                -10, 0, 5, 10, 10, 5, 0, -10,
                -10, 0, 0, 0, 0, 0, 0, -10,
                -20, -10, -10, -10, -10, -10, -10, -20
            ]),
            'R': np.zeros(64),
            'Q': np.zeros(64),
            'K': np.array([
                20, 30, 10, 0, 0, 10, 30, 20,
                20, 20, 0, 0, 0, 0, 20, 20,
                -10, -20, -20, -20, -20, -20, -20, -10,
                -20, -30, -30, -40, -40, -30, -30, -20,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30
            ])
        }

    def evaluate(self, board):
        score = 0
        for piece, bb in board.bitboards.items():
            piece_sign = 1 if piece.isupper() else -1
            for i in range(64):
                if (bb >> np.uint64(i)) & np.uint64(1):
                    material = self.piece_values[piece]
                    # Flip the rank for Black so the white‑oriented PST is used correctly
                    pst_index = i if piece.isupper() else (i ^ 56)  # XOR 56 mirrors vertically
                    pst_bonus = self.pst.get(piece.upper(), np.zeros(64))[pst_index] * piece_sign
                    score += material + pst_bonus

        score += self.evaluate_pawn_structure(board)

        if board.white_to_move:
            score += 10
        else:
            score -= 10
        return score

    def evaluate_pawn_structure(self, board):
        score = 0
        white_pawns = board.bitboards['P']
        black_pawns = board.bitboards['p']
        files = [0] * 8

        for i in range(64):
            if (white_pawns >> np.uint64(i)) & np.uint64(1):
                file = i % 8
                files[file] += 1
                if self.is_passed_pawn(i, white_pawns, black_pawns, True):
                    score += 30
            if (black_pawns >> np.uint64(i)) & np.uint64(1):
                file = i % 8
                files[file] -= 1
                if self.is_passed_pawn(i, black_pawns, white_pawns, False):
                    score -= 30

        for f in range(8):
            if files[f] > 1:
                score -= 20 * (files[f] - 1)  # Doubled or tripled pawns penalty
            if files[f] == 1 and (f == 0 or f == 7 or files[f-1] == 0 and (f == 0 or files[f-1] == 0) and (f == 7 or files[f+1] == 0)):
                score -= 15  # Isolated pawn

        return score

    def is_passed_pawn(self, square, own_pawns, enemy_pawns, white):
        file = square % 8
        rank = square // 8
        if white:
            for r in range(rank + 1, 8):
                for f in [file-1, file, file+1]:
                    if 0 <= f <= 7 and ((enemy_pawns >> np.uint64(r*8 + f)) & np.uint64(1)):
                        return False
        else:
            for r in range(rank - 1, -1, -1):
                for f in [file-1, file, file+1]:
                    if 0 <= f <= 7 and ((enemy_pawns >> np.uint64(r*8 + f)) & np.uint64(1)):
                        return False
        return True

## if __name__ == "__main__":  # [Commented out Evaluator test block]
##     # from board import Board
##
##     board = Board()
##     evaluator = Evaluator()
##     print("Evaluation Score:", evaluator.evaluate(board))


