import numpy as np
from move import Move  
from evaluation import Evaluator  
from search import Searcher  

class Board:
    def generate_legal_moves(self):
        legal_moves = []
        pseudo_moves = self.generate_moves()
        # print("\n[DEBUG] Number of pseudo-legal moves generated:", len(pseudo_moves))
        # print("[DEBUG] Moves:", [str(move) for move in pseudo_moves])
        for move in pseudo_moves:
            self.make_move(move)
            if not self.is_check():
                legal_moves.append(move)
            self.undo_move(move)
        # print("[DEBUG] Number of legal moves after filtering checks:", len(legal_moves))
        # print("[DEBUG] Legal Moves:", [str(move) for move in legal_moves])
        return legal_moves
    def __init__(self):
        # 12 bitboards: White and Black pieces separately
        # Order: [Pawns, Knights, Bishops, Rooks, Queens, Kings] x 2
        self.bitboards = {
            'P': np.uint64(0), 'N': np.uint64(0), 'B': np.uint64(0), 'R': np.uint64(0), 'Q': np.uint64(0), 'K': np.uint64(0),
            'p': np.uint64(0), 'n': np.uint64(0), 'b': np.uint64(0), 'r': np.uint64(0), 'q': np.uint64(0), 'k': np.uint64(0)
        }

        # Other board state
        self.white_to_move = True
        self.castling_rights = {'K': True, 'Q': True, 'k': True, 'q': True}  # White kingside, White queenside, etc.
        self.en_passant_square = None  # If there's an en passant target
        self.halfmove_clock = 0  # For 50-move draw rule
        self.fullmove_number = 1  # Starts at 1 and increments after Black's move

        # History stack for make/undo – one snapshot per ply
        self._state_stack = []

        # Initialize to starting position
        self.set_starting_position()

    def set_starting_position(self):
        """
        Index 0 = a8, …, 63 = h1 (big‑endian).  
        This mapping is what the move‑generation logic expects.
        """
        # White pieces (bottom two ranks from the human view but high indices here)
        self.bitboards['P'] = np.uint64(0x00FF000000000000)   # a2‑h2 (index 48‑55)
        self.bitboards['N'] = np.uint64(0x4200000000000000)   # b1 g1 (62, 57)
        self.bitboards['B'] = np.uint64(0x2400000000000000)   # c1 f1
        self.bitboards['R'] = np.uint64(0x8100000000000000)   # a1 h1
        self.bitboards['Q'] = np.uint64(0x0800000000000000)   # d1
        self.bitboards['K'] = np.uint64(0x1000000000000000)   # e1

        # Black pieces (top two ranks in the human view but low indices here)
        self.bitboards['p'] = np.uint64(0x000000000000FF00)   # a7‑h7 (index 8‑15)
        self.bitboards['n'] = np.uint64(0x0000000000000042)   # b8 g8
        self.bitboards['b'] = np.uint64(0x0000000000000024)   # c8 f8
        self.bitboards['r'] = np.uint64(0x0000000000000081)   # a8 h8
        self.bitboards['q'] = np.uint64(0x0000000000000008)   # d8
        self.bitboards['k'] = np.uint64(0x0000000000000010)   # e8

        self.white_to_move = True
        self.castling_rights = {'K': True, 'Q': True, 'k': True, 'q': True}
        self.en_passant_square = None
        self.halfmove_clock = 0
        self.fullmove_number = 1

    def print_board(self):
        # Generate a human-readable board
        files = 'a b c d e f g h'
        print('    ' + files)
        for rank in range(7, -1, -1):  # Correctly print from rank 8 down to rank 1
            row = []
            for file in range(8):
                sq = rank * 8 + file
                piece = '.'
                for p, bb in self.bitboards.items():
                    if (bb >> np.uint64(sq)) & np.uint64(1):
                        piece = p
                        break
                row.append(piece)
            print(f"{rank+1} | {' '.join(row)} | {rank+1}")
        print('    ' + files)

    def generate_moves(self):
        moves = []
        if self.white_to_move:
            moves += self.generate_pawn_moves(white=True)
            moves += self.generate_knight_moves(white=True)
            moves += self.generate_king_moves(white=True)
            moves += self.generate_rook_moves(white=True)
            moves += self.generate_bishop_moves(white=True)
            moves += self.generate_queen_moves(white=True)
        else:
            moves += self.generate_pawn_moves(white=False)
            moves += self.generate_knight_moves(white=False)
            moves += self.generate_king_moves(white=False)
            moves += self.generate_rook_moves(white=False)
            moves += self.generate_bishop_moves(white=False)
            moves += self.generate_queen_moves(white=False)
        return moves

    def generate_king_moves(self, white):
        moves = []
        if white:
            king = self.bitboards['K']
        else:
            king = self.bitboards['k']

        king_offsets = [8, -8, 1, -1, 9, -9, 7, -7]

        for sq in range(64):
            if (king >> np.uint64(sq)) & np.uint64(1):
                for offset in king_offsets:
                    to_sq = sq + offset
                    if 0 <= to_sq < 64 and self.valid_king_move(sq, to_sq):
                        if self.is_empty(to_sq) or self.is_enemy(to_sq, white):
                            capture = self.is_enemy(to_sq, white)
                            moves.append(Move(sq, to_sq, capture=capture))
                moves += self.generate_castling_moves(white, sq)
        return moves

    def valid_king_move(self, from_sq, to_sq):
        from_file = from_sq % 8
        to_file = to_sq % 8
        from_rank = from_sq // 8
        to_rank = to_sq // 8
        return max(abs(from_file - to_file), abs(from_rank - to_rank)) == 1

    def generate_castling_moves(self, white, king_sq):
        moves = []
        if white:
            if self.castling_rights['K']:
                if self.is_empty(61) and self.is_empty(62):
                    moves.append(Move(king_sq, 62, castle='kingside'))
            if self.castling_rights['Q']:
                if self.is_empty(59) and self.is_empty(58) and self.is_empty(57):
                    moves.append(Move(king_sq, 58, castle='queenside'))
        else:
            if self.castling_rights['k']:
                if self.is_empty(5) and self.is_empty(6):
                    moves.append(Move(king_sq, 6, castle='kingside'))
            if self.castling_rights['q']:
                if self.is_empty(3) and self.is_empty(2) and self.is_empty(1):
                    moves.append(Move(king_sq, 2, castle='queenside'))
        return moves
    def generate_rook_moves(self, white):
        return self.generate_sliding_moves(white, ['R', 'r'], directions=[8, -8, 1, -1])

    def generate_bishop_moves(self, white):
        return self.generate_sliding_moves(white, ['B', 'b'], directions=[9, -9, 7, -7])

    def generate_queen_moves(self, white):
        return self.generate_sliding_moves(white, ['Q', 'q'], directions=[8, -8, 1, -1, 9, -9, 7, -7])

    def generate_sliding_moves(self, white, pieces, directions):
        moves = []
        piece = pieces[0] if white else pieces[1]
        bitboard = self.bitboards[piece]

        for sq in range(64):
            if (bitboard >> np.uint64(sq)) & np.uint64(1):
                for direction in directions:
                    to_sq = sq
                    while True:
                        to_sq += direction
                        if not (0 <= to_sq < 64):
                            break
                        if not self.valid_slide(sq, to_sq, direction):
                            break
                        if self.is_empty(to_sq):
                            moves.append(Move(sq, to_sq))
                        elif self.is_enemy(to_sq, white):
                            moves.append(Move(sq, to_sq, capture=True))
                            break
                        else:
                            break
        return moves

    def valid_slide(self, from_sq, to_sq, direction):
        from_file, from_rank = from_sq % 8, from_sq // 8
        to_file,   to_rank   = to_sq   % 8, to_sq   // 8
        df, dr = to_file - from_file, to_rank - from_rank

        # Horizontal rays
        if direction in (1, -1):
            return dr == 0

        # Vertical rays
        if direction in (8, -8):
            return df == 0

        # Diagonals: 9 NW‑SE  /  7 NE‑SW (in index space)
        if abs(direction) == 9:   # NW‑SE
            return abs(df) == abs(dr) and df == dr
        if abs(direction) == 7:   # NE‑SW
            return abs(df) == abs(dr) and df == -dr

        return False

    def generate_knight_moves(self, white):
        moves = []
        if white:
            knights = self.bitboards['N']
        else:
            knights = self.bitboards['n']

        knight_offsets = [17, 15, 10, 6, -17, -15, -10, -6]

        for sq in range(64):
            if (knights >> np.uint64(sq)) & np.uint64(1):
                for offset in knight_offsets:
                    to_sq = sq + offset
                    if 0 <= to_sq < 64 and self.valid_knight_jump(sq, to_sq):
                        if self.is_empty(to_sq) or self.is_enemy(to_sq, white):
                            capture = self.is_enemy(to_sq, white)
                            moves.append(Move(sq, to_sq, capture=capture))
        return moves

    def valid_knight_jump(self, from_sq, to_sq):
        from_file = from_sq % 8
        to_file = to_sq % 8
        return abs(from_file - to_file) in [1, 2]

    def generate_pawn_moves(self, white):
        moves = []
        if white:
            pawns = self.bitboards['P']
            direction = -8
            start_rank = 6
            promotion_rank = 0
            enemy_pieces = (self.bitboards['p'] | self.bitboards['n'] |
                            self.bitboards['b'] | self.bitboards['r'] |
                            self.bitboards['q'] | self.bitboards['k'])
        else:
            pawns = self.bitboards['p']
            direction = 8
            start_rank = 1
            promotion_rank = 7
            enemy_pieces = (self.bitboards['P'] | self.bitboards['N'] |
                            self.bitboards['B'] | self.bitboards['R'] |
                            self.bitboards['Q'] | self.bitboards['K'])

        for sq in range(64):
            if (pawns >> np.uint64(sq)) & np.uint64(1):
                to_sq = sq + direction

                if self.is_empty(to_sq):
                    if to_sq // 8 == promotion_rank:
                        for promo_piece in ['q', 'r', 'b', 'n']:
                            moves.append(Move(sq, to_sq, promotion=promo_piece.upper() if white else promo_piece))
                    else:
                        moves.append(Move(sq, to_sq))

                    if sq // 8 == start_rank:
                        to_sq2 = sq + 2 * direction
                        if self.is_empty(to_sq2):
                            moves.append(Move(sq, to_sq2))

                for capture_dir in [-1, 1]:
                    cap_sq = sq + direction + capture_dir
                    if 0 <= cap_sq < 64 and self.is_enemy(cap_sq, white):
                        if cap_sq // 8 == promotion_rank:
                            for promo_piece in ['q', 'r', 'b', 'n']:
                                moves.append(Move(sq, cap_sq, promotion=promo_piece.upper() if white else promo_piece, capture=True))
                        else:
                            moves.append(Move(sq, cap_sq, capture=True))

                if self.en_passant_square is not None:
                    for capture_dir in [-1, 1]:
                        cap_sq = sq + direction + capture_dir
                        if cap_sq == self.en_passant_square:
                            moves.append(Move(sq, self.en_passant_square, en_passant=True))

        return moves

    def is_empty(self, square):
        if not (0 <= square < 64):
            return False
        for bb in self.bitboards.values():
            if (bb >> np.uint64(square)) & np.uint64(1):
                return False
        return True

    def is_enemy(self, square, white):
        if not (0 <= square < 64):
            return False
        if white:
            return any((self.bitboards[p] >> np.uint64(square)) & np.uint64(1) for p in ['p', 'n', 'b', 'r', 'q', 'k'])
        else:
            return any((self.bitboards[p] >> np.uint64(square)) & np.uint64(1) for p in ['P', 'N', 'B', 'R', 'Q', 'K'])

    def make_move(self, move):
        # Push a snapshot so we can undo later (supports deep search)
        self._state_stack.append({
            'bitboards': {k: v.copy() for k, v in self.bitboards.items()},
            'white_to_move': self.white_to_move,
            'castling_rights': self.castling_rights.copy(),
            'en_passant_square': self.en_passant_square,
            'halfmove_clock': self.halfmove_clock,
            'fullmove_number': self.fullmove_number
        })

        moving_piece = None
        for piece, bb in self.bitboards.items():
            if (bb >> np.uint64(move.from_square)) & np.uint64(1):
                moving_piece = piece
                break

        if moving_piece is None:
            raise ValueError("No piece to move.")

        # Clear source square
        self.bitboards[moving_piece] &= ~(np.uint64(1) << np.uint64(move.from_square))
        # Capture
        if getattr(move, 'capture', False) or getattr(move, 'en_passant', False):
            for piece, bb in self.bitboards.items():
                if (bb >> np.uint64(move.to_square)) & np.uint64(1):
                    self.bitboards[piece] &= ~(np.uint64(1) << np.uint64(move.to_square))
                    break
            if getattr(move, 'en_passant', False):
                ep_capture_sq = move.to_square + (8 if self.white_to_move else -8)
                for piece, bb in self.bitboards.items():
                    if (bb >> np.uint64(ep_capture_sq)) & np.uint64(1):
                        self.bitboards[piece] &= ~(np.uint64(1) << np.uint64(ep_capture_sq))
                        break

        # Place piece in destination
        if getattr(move, 'promotion', None):
            promo_piece = move.promotion if self.white_to_move else move.promotion.lower()
            self.bitboards[promo_piece] |= (np.uint64(1) << np.uint64(move.to_square))
        else:
            self.bitboards[moving_piece] |= (np.uint64(1) << np.uint64(move.to_square))

        # Castling move
        if getattr(move, 'castle', None) == 'kingside':
            if self.white_to_move:
                self.bitboards['R'] &= ~(np.uint64(1) << np.uint64(63))
                self.bitboards['R'] |= (np.uint64(1) << np.uint64(61))
            else:
                self.bitboards['r'] &= ~(np.uint64(1) << np.uint64(7))
                self.bitboards['r'] |= (np.uint64(1) << np.uint64(5))
        elif getattr(move, 'castle', None) == 'queenside':
            if self.white_to_move:
                self.bitboards['R'] &= ~(np.uint64(1) << np.uint64(56))
                self.bitboards['R'] |= (np.uint64(1) << np.uint64(59))
            else:
                self.bitboards['r'] &= ~(np.uint64(1) << np.uint64(0))
                self.bitboards['r'] |= (np.uint64(1) << np.uint64(3))

        # Update en passant square
        if moving_piece.upper() == 'P' and abs(move.from_square - move.to_square) == 16:
            self.en_passant_square = (move.from_square + move.to_square) // 2
        else:
            self.en_passant_square = None

        # Update castling rights
        if moving_piece.upper() == 'K':
            if self.white_to_move:
                self.castling_rights['K'] = False
                self.castling_rights['Q'] = False
            else:
                self.castling_rights['k'] = False
                self.castling_rights['q'] = False
        if moving_piece.upper() == 'R':
            if move.from_square == 63:
                self.castling_rights['K'] = False
            elif move.from_square == 56:
                self.castling_rights['Q'] = False
        if moving_piece.upper() == 'r':
            if move.from_square == 7:
                self.castling_rights['k'] = False
            elif move.from_square == 0:
                self.castling_rights['q'] = False

        # Update move counters
        if moving_piece.upper() == 'P' or getattr(move, 'capture', False):
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        if not self.white_to_move:
            self.fullmove_number += 1

        self.white_to_move = not self.white_to_move

    def undo_move(self, move):
        if not self._state_stack:
            raise ValueError("No previous move to undo.")

        state = self._state_stack.pop()

        # Restore everything exactly as it was
        self.bitboards = {k: v.copy() for k, v in state['bitboards'].items()}
        self.white_to_move = state['white_to_move']
        self.castling_rights = state['castling_rights'].copy()
        self.en_passant_square = state['en_passant_square']
        self.halfmove_clock = state['halfmove_clock']
        self.fullmove_number = state['fullmove_number']

    def is_check(self):
        king_square = None
        king_piece = 'K' if self.white_to_move else 'k'

        for i in range(64):
            if (self.bitboards[king_piece] >> np.uint64(i)) & np.uint64(1):
                king_square = i
                break

        if king_square is None:
            return False  # King missing (edge case)

        return self.is_square_attacked(king_square, not self.white_to_move)

    def is_square_attacked(self, square, by_white):
        if by_white:
            enemy_pawns = self.bitboards['P']
            enemy_knights = self.bitboards['N']
            enemy_bishops = self.bitboards['B']
            enemy_rooks = self.bitboards['R']
            enemy_queens = self.bitboards['Q']
            enemy_king = self.bitboards['K']
        else:
            enemy_pawns = self.bitboards['p']
            enemy_knights = self.bitboards['n']
            enemy_bishops = self.bitboards['b']
            enemy_rooks = self.bitboards['r']
            enemy_queens = self.bitboards['q']
            enemy_king = self.bitboards['k']

        # Pawn attacks
        pawn_attack_offsets = [-9, -7] if by_white else [7, 9]
        for offset in pawn_attack_offsets:
            sq = square + offset
            if 0 <= sq < 64 and self.valid_slide(square, sq, offset):
                if (enemy_pawns >> np.uint64(sq)) & np.uint64(1):
                    return True

        # Knight attacks
        knight_offsets = [17, 15, 10, 6, -17, -15, -10, -6]
        for offset in knight_offsets:
            sq = square + offset
            if 0 <= sq < 64 and self.valid_knight_jump(square, sq):
                if (enemy_knights >> np.uint64(sq)) & np.uint64(1):
                    return True

        # King attacks
        king_offsets = [8, -8, 1, -1, 9, -9, 7, -7]
        for offset in king_offsets:
            sq = square + offset
            if 0 <= sq < 64 and self.valid_king_move(square, sq):
                if (enemy_king >> np.uint64(sq)) & np.uint64(1):
                    return True

        # Sliding attacks
        sliding_directions = {
            'bishop': [9, -9, 7, -7],
            'rook': [8, -8, 1, -1],
            'queen': [8, -8, 1, -1, 9, -9, 7, -7],
        }

        for directions, pieces in [
            (sliding_directions['bishop'], [enemy_bishops, enemy_queens]),
            (sliding_directions['rook'], [enemy_rooks, enemy_queens]),
        ]:
            for direction in directions:
                sq = square
                while True:
                    sq += direction
                    if not (0 <= sq < 64):
                        break
                    if not self.valid_slide(square, sq, direction):
                        break
                    occupied = False
                    for bb in self.bitboards.values():
                        if (bb >> np.uint64(sq)) & np.uint64(1):
                            occupied = True
                            break
                    if occupied:
                        if any((bb >> np.uint64(sq)) & np.uint64(1) for bb in pieces):
                            return True
                        break
        return False

    def is_checkmate(self):
        if self.is_check() and len(self.generate_legal_moves()) == 0:
            return True
        return False

    def is_stalemate(self):
        if not self.is_check() and len(self.generate_legal_moves()) == 0:
            return True
        return False

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def print_bitboards(self):
        print("Current Bitboards:")
        for piece, bb in self.bitboards.items():
            print(f"{piece}: {bb:064b}")

## if __name__ == "__main__":  # [Commented out Board test block]
##     board = Board()
##     board.print_board()
##     board.print_bitboards()
