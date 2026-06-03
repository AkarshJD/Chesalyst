import numpy as np
from chesalyst.move import Move


class Board:
    def __init__(self):
        self.bitboards = {
            'P': np.uint64(0), 'N': np.uint64(0), 'B': np.uint64(0),
            'R': np.uint64(0), 'Q': np.uint64(0), 'K': np.uint64(0),
            'p': np.uint64(0), 'n': np.uint64(0), 'b': np.uint64(0),
            'r': np.uint64(0), 'q': np.uint64(0), 'k': np.uint64(0),
        }
        self.white_to_move = True
        self.castling_rights = {'K': True, 'Q': True, 'k': True, 'q': True}
        self.en_passant_square = None
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self._state_stack = []
        self.set_starting_position()

    def set_starting_position(self):
        # Index 0 = a8 (top-left), index 63 = h1 (bottom-right)
        # index = rank_from_top * 8 + file  (rank 0 = rank 8 in chess)
        self.bitboards['P'] = np.uint64(0x00FF000000000000)  # a2-h2 (48-55)
        self.bitboards['N'] = np.uint64(0x4200000000000000)  # b1, g1
        self.bitboards['B'] = np.uint64(0x2400000000000000)  # c1, f1
        self.bitboards['R'] = np.uint64(0x8100000000000000)  # a1, h1
        self.bitboards['Q'] = np.uint64(0x0800000000000000)  # d1
        self.bitboards['K'] = np.uint64(0x1000000000000000)  # e1
        self.bitboards['p'] = np.uint64(0x000000000000FF00)  # a7-h7 (8-15)
        self.bitboards['n'] = np.uint64(0x0000000000000042)  # b8, g8
        self.bitboards['b'] = np.uint64(0x0000000000000024)  # c8, f8
        self.bitboards['r'] = np.uint64(0x0000000000000081)  # a8, h8
        self.bitboards['q'] = np.uint64(0x0000000000000008)  # d8
        self.bitboards['k'] = np.uint64(0x0000000000000010)  # e8
        self.white_to_move = True
        self.castling_rights = {'K': True, 'Q': True, 'k': True, 'q': True}
        self.en_passant_square = None
        self.halfmove_clock = 0
        self.fullmove_number = 1

    # ------------------------------------------------------------------ #
    #  Legal move generation                                               #
    # ------------------------------------------------------------------ #

    def generate_legal_moves(self):
        legal_moves = []
        moving_side = self.white_to_move
        for move in self.generate_moves():
            self.make_move(move)
            if not self.is_in_check_for(moving_side):
                legal_moves.append(move)
            self.undo_move(move)
        return legal_moves

    def generate_moves(self):
        moves = []
        white = self.white_to_move
        moves += self.generate_pawn_moves(white)
        moves += self.generate_knight_moves(white)
        moves += self.generate_king_moves(white)
        moves += self.generate_rook_moves(white)
        moves += self.generate_bishop_moves(white)
        moves += self.generate_queen_moves(white)
        return moves

    # ------------------------------------------------------------------ #
    #  Piece move generators                                               #
    # ------------------------------------------------------------------ #

    def generate_pawn_moves(self, white):
        moves = []
        if white:
            pawns = self.bitboards['P']
            direction = -8
            start_rank = 6
            promotion_rank = 0
        else:
            pawns = self.bitboards['p']
            direction = 8
            start_rank = 1
            promotion_rank = 7

        for sq in range(64):
            if not ((pawns >> np.uint64(sq)) & np.uint64(1)):
                continue

            # Single push
            to_sq = sq + direction
            if 0 <= to_sq < 64 and self.is_empty(to_sq):
                if to_sq // 8 == promotion_rank:
                    for promo in (['Q', 'R', 'B', 'N'] if white else ['q', 'r', 'b', 'n']):
                        moves.append(Move(sq, to_sq, promotion=promo))
                else:
                    moves.append(Move(sq, to_sq))
                    # Double push from starting rank
                    if sq // 8 == start_rank:
                        to_sq2 = sq + 2 * direction
                        if self.is_empty(to_sq2):
                            moves.append(Move(sq, to_sq2))

            # Diagonal captures
            for cap_dir in [-1, 1]:
                cap_sq = sq + direction + cap_dir
                # Guard against file wrapping: captured square must be one file away
                if 0 <= cap_sq < 64 and abs(cap_sq % 8 - sq % 8) == 1:
                    if self.is_enemy(cap_sq, white):
                        if cap_sq // 8 == promotion_rank:
                            for promo in (['Q', 'R', 'B', 'N'] if white else ['q', 'r', 'b', 'n']):
                                moves.append(Move(sq, cap_sq, promotion=promo, capture=True))
                        else:
                            moves.append(Move(sq, cap_sq, capture=True))
                    # En passant
                    if self.en_passant_square is not None and cap_sq == self.en_passant_square:
                        moves.append(Move(sq, self.en_passant_square, en_passant=True))

        return moves

    def generate_knight_moves(self, white):
        moves = []
        knights = self.bitboards['N'] if white else self.bitboards['n']
        for sq in range(64):
            if not ((knights >> np.uint64(sq)) & np.uint64(1)):
                continue
            for offset in [17, 15, 10, 6, -17, -15, -10, -6]:
                to_sq = sq + offset
                if 0 <= to_sq < 64 and self._valid_knight_jump(sq, to_sq):
                    if self.is_empty(to_sq) or self.is_enemy(to_sq, white):
                        moves.append(Move(sq, to_sq, capture=self.is_enemy(to_sq, white)))
        return moves

    def generate_king_moves(self, white):
        moves = []
        king = self.bitboards['K'] if white else self.bitboards['k']
        for sq in range(64):
            if not ((king >> np.uint64(sq)) & np.uint64(1)):
                continue
            for offset in [8, -8, 1, -1, 9, -9, 7, -7]:
                to_sq = sq + offset
                if 0 <= to_sq < 64 and self._valid_king_step(sq, to_sq):
                    if self.is_empty(to_sq) or self.is_enemy(to_sq, white):
                        moves.append(Move(sq, to_sq, capture=self.is_enemy(to_sq, white)))
            moves += self._generate_castling_moves(white, sq)
        return moves

    def _generate_castling_moves(self, white, king_sq):
        moves = []
        if white:
            if self.castling_rights['K'] and self.is_empty(61) and self.is_empty(62):
                moves.append(Move(king_sq, 62, castle='kingside'))
            if self.castling_rights['Q'] and self.is_empty(59) and self.is_empty(58) and self.is_empty(57):
                moves.append(Move(king_sq, 58, castle='queenside'))
        else:
            if self.castling_rights['k'] and self.is_empty(5) and self.is_empty(6):
                moves.append(Move(king_sq, 6, castle='kingside'))
            if self.castling_rights['q'] and self.is_empty(3) and self.is_empty(2) and self.is_empty(1):
                moves.append(Move(king_sq, 2, castle='queenside'))
        return moves

    def generate_rook_moves(self, white):
        return self._generate_sliding_moves(white, 'R', 'r', [8, -8, 1, -1])

    def generate_bishop_moves(self, white):
        return self._generate_sliding_moves(white, 'B', 'b', [9, -9, 7, -7])

    def generate_queen_moves(self, white):
        return self._generate_sliding_moves(white, 'Q', 'q', [8, -8, 1, -1, 9, -9, 7, -7])

    def _generate_sliding_moves(self, white, white_piece, black_piece, directions):
        moves = []
        piece = white_piece if white else black_piece
        bitboard = self.bitboards[piece]
        for sq in range(64):
            if not ((bitboard >> np.uint64(sq)) & np.uint64(1)):
                continue
            for direction in directions:
                to_sq = sq
                while True:
                    to_sq += direction
                    if not (0 <= to_sq < 64):
                        break
                    if not self._valid_slide(sq, to_sq, direction):
                        break
                    if self.is_empty(to_sq):
                        moves.append(Move(sq, to_sq))
                    elif self.is_enemy(to_sq, white):
                        moves.append(Move(sq, to_sq, capture=True))
                        break
                    else:
                        break
        return moves

    # ------------------------------------------------------------------ #
    #  Make / Undo                                                         #
    # ------------------------------------------------------------------ #

    def make_move(self, move):
        self._state_stack.append({
            'bitboards': {k: v.copy() for k, v in self.bitboards.items()},
            'white_to_move': self.white_to_move,
            'castling_rights': self.castling_rights.copy(),
            'en_passant_square': self.en_passant_square,
            'halfmove_clock': self.halfmove_clock,
            'fullmove_number': self.fullmove_number,
        })

        moving_piece = None
        for piece, bb in self.bitboards.items():
            if (bb >> np.uint64(move.from_square)) & np.uint64(1):
                moving_piece = piece
                break
        if moving_piece is None:
            raise ValueError(f"No piece on square {move.from_square}")

        # Clear source
        self.bitboards[moving_piece] &= ~(np.uint64(1) << np.uint64(move.from_square))

        # Remove captured piece
        if move.capture or move.en_passant:
            cap_sq = move.to_square
            if move.en_passant:
                cap_sq = move.to_square + (8 if self.white_to_move else -8)
            for piece, bb in self.bitboards.items():
                if (bb >> np.uint64(cap_sq)) & np.uint64(1):
                    self.bitboards[piece] &= ~(np.uint64(1) << np.uint64(cap_sq))
                    break

        # Place on destination
        dest_piece = move.promotion if move.promotion else moving_piece
        self.bitboards[dest_piece] |= (np.uint64(1) << np.uint64(move.to_square))

        # Rook relocation for castling
        if move.castle == 'kingside':
            if self.white_to_move:
                self.bitboards['R'] &= ~(np.uint64(1) << np.uint64(63))
                self.bitboards['R'] |= (np.uint64(1) << np.uint64(61))
            else:
                self.bitboards['r'] &= ~(np.uint64(1) << np.uint64(7))
                self.bitboards['r'] |= (np.uint64(1) << np.uint64(5))
        elif move.castle == 'queenside':
            if self.white_to_move:
                self.bitboards['R'] &= ~(np.uint64(1) << np.uint64(56))
                self.bitboards['R'] |= (np.uint64(1) << np.uint64(59))
            else:
                self.bitboards['r'] &= ~(np.uint64(1) << np.uint64(0))
                self.bitboards['r'] |= (np.uint64(1) << np.uint64(3))

        # En passant square for next move
        if moving_piece.upper() == 'P' and abs(move.from_square - move.to_square) == 16:
            self.en_passant_square = (move.from_square + move.to_square) // 2
        else:
            self.en_passant_square = None

        # Castling rights: king moves
        if moving_piece == 'K':
            self.castling_rights['K'] = False
            self.castling_rights['Q'] = False
        elif moving_piece == 'k':
            self.castling_rights['k'] = False
            self.castling_rights['q'] = False
        # Castling rights: rook moves
        elif moving_piece == 'R':
            if move.from_square == 63:
                self.castling_rights['K'] = False
            elif move.from_square == 56:
                self.castling_rights['Q'] = False
        elif moving_piece == 'r':
            if move.from_square == 7:
                self.castling_rights['k'] = False
            elif move.from_square == 0:
                self.castling_rights['q'] = False

        if moving_piece.upper() == 'P' or move.capture or move.en_passant:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        if not self.white_to_move:
            self.fullmove_number += 1

        self.white_to_move = not self.white_to_move

    def undo_move(self, move):
        if not self._state_stack:
            raise ValueError("No state to undo.")
        state = self._state_stack.pop()
        self.bitboards = {k: v.copy() for k, v in state['bitboards'].items()}
        self.white_to_move = state['white_to_move']
        self.castling_rights = state['castling_rights'].copy()
        self.en_passant_square = state['en_passant_square']
        self.halfmove_clock = state['halfmove_clock']
        self.fullmove_number = state['fullmove_number']

    # ------------------------------------------------------------------ #
    #  Check / attack detection                                            #
    # ------------------------------------------------------------------ #

    def is_in_check_for(self, white):
        """True if the given side's king is currently attacked by the opponent."""
        king_piece = 'K' if white else 'k'
        for i in range(64):
            if (self.bitboards[king_piece] >> np.uint64(i)) & np.uint64(1):
                return self.is_square_attacked(i, not white)
        return False

    def is_check(self):
        """True if the side to move is in check."""
        return self.is_in_check_for(self.white_to_move)

    def is_checkmate(self):
        return self.is_check() and len(self.generate_legal_moves()) == 0

    def is_stalemate(self):
        return not self.is_check() and len(self.generate_legal_moves()) == 0

    def is_square_attacked(self, square, by_white):
        """True if `square` is attacked by any piece of the given color."""
        if by_white:
            pawns   = self.bitboards['P']
            knights = self.bitboards['N']
            bishops = self.bitboards['B']
            rooks   = self.bitboards['R']
            queens  = self.bitboards['Q']
            king    = self.bitboards['K']
        else:
            pawns   = self.bitboards['p']
            knights = self.bitboards['n']
            bishops = self.bitboards['b']
            rooks   = self.bitboards['r']
            queens  = self.bitboards['q']
            king    = self.bitboards['k']

        # Pawn attacks: look from the target square *backward* to where an
        # attacking pawn would stand.
        # White pawns move in -8 dir and attack at offsets -9/-7 from the pawn.
        # So a white pawn at (square+9) or (square+7) attacks `square`.
        # Black pawns move in +8 dir and attack at offsets +9/+7 from the pawn.
        # So a black pawn at (square-9) or (square-7) attacks `square`.
        pawn_offsets = [9, 7] if by_white else [-9, -7]
        for offset in pawn_offsets:
            sq = square + offset
            if 0 <= sq < 64 and self._valid_slide(square, sq, offset):
                if (pawns >> np.uint64(sq)) & np.uint64(1):
                    return True

        # Knight attacks
        for offset in [17, 15, 10, 6, -17, -15, -10, -6]:
            sq = square + offset
            if 0 <= sq < 64 and self._valid_knight_jump(square, sq):
                if (knights >> np.uint64(sq)) & np.uint64(1):
                    return True

        # King attacks
        for offset in [8, -8, 1, -1, 9, -9, 7, -7]:
            sq = square + offset
            if 0 <= sq < 64 and self._valid_king_step(square, sq):
                if (king >> np.uint64(sq)) & np.uint64(1):
                    return True

        # Diagonal sliders (bishop / queen)
        for direction in [9, -9, 7, -7]:
            sq = square
            while True:
                sq += direction
                if not (0 <= sq < 64) or not self._valid_slide(square, sq, direction):
                    break
                occupied = any((bb >> np.uint64(sq)) & np.uint64(1) for bb in self.bitboards.values())
                if occupied:
                    if (bishops >> np.uint64(sq)) & np.uint64(1) or (queens >> np.uint64(sq)) & np.uint64(1):
                        return True
                    break

        # Straight sliders (rook / queen)
        for direction in [8, -8, 1, -1]:
            sq = square
            while True:
                sq += direction
                if not (0 <= sq < 64) or not self._valid_slide(square, sq, direction):
                    break
                occupied = any((bb >> np.uint64(sq)) & np.uint64(1) for bb in self.bitboards.values())
                if occupied:
                    if (rooks >> np.uint64(sq)) & np.uint64(1) or (queens >> np.uint64(sq)) & np.uint64(1):
                        return True
                    break

        return False

    # ------------------------------------------------------------------ #
    #  Geometry helpers                                                    #
    # ------------------------------------------------------------------ #

    def _valid_king_step(self, from_sq, to_sq):
        return (max(abs(from_sq % 8 - to_sq % 8),
                    abs(from_sq // 8 - to_sq // 8)) == 1)

    def _valid_knight_jump(self, from_sq, to_sq):
        return abs(from_sq % 8 - to_sq % 8) in (1, 2)

    def _valid_slide(self, from_sq, to_sq, direction):
        df = to_sq % 8 - from_sq % 8
        dr = to_sq // 8 - from_sq // 8
        if direction in (1, -1):
            return dr == 0
        if direction in (8, -8):
            return df == 0
        if abs(direction) == 9:
            return abs(df) == abs(dr) and df == dr
        if abs(direction) == 7:
            return abs(df) == abs(dr) and df == -dr
        return False

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def is_empty(self, square):
        if not (0 <= square < 64):
            return False
        return not any((bb >> np.uint64(square)) & np.uint64(1) for bb in self.bitboards.values())

    def is_enemy(self, square, white):
        if not (0 <= square < 64):
            return False
        enemy = ['p', 'n', 'b', 'r', 'q', 'k'] if white else ['P', 'N', 'B', 'R', 'Q', 'K']
        return any((self.bitboards[p] >> np.uint64(square)) & np.uint64(1) for p in enemy)

    def print_board(self):
        files = 'a b c d e f g h'
        print('    ' + files)
        for rank in range(7, -1, -1):
            row = []
            for file in range(8):
                sq = rank * 8 + file
                piece = '.'
                for p, bb in self.bitboards.items():
                    if (bb >> np.uint64(sq)) & np.uint64(1):
                        piece = p
                        break
                row.append(piece)
            print(f"{rank + 1} | {' '.join(row)} | {rank + 1}")
        print('    ' + files)

    def copy(self):
        import copy
        return copy.deepcopy(self)
