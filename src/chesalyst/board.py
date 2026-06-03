import numpy as np
from chesalyst.move import Move

_PIECE_ORDER = ('P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k')

# Expected file delta per sliding direction — used for wrap detection.
# Indexed as a list with offset 9 so direction+9 gives the delta.
_DIR_FILE_DELTA = [None] * 19
for _d, _df in [(8,0),(-8,0),(1,1),(-1,-1),(9,1),(-9,-1),(7,-1),(-7,1)]:
    _DIR_FILE_DELTA[_d + 9] = _df
del _d, _df

# Precomputed attack masks (Python ints) for non-sliding pieces.
_KING_ATTACKS   = [0] * 64  # squares a king on sq can reach
_KNIGHT_ATTACKS = [0] * 64
_WP_ATTACK_FROM = [0] * 64  # white-pawn squares that attack sq
_BP_ATTACK_FROM = [0] * 64  # black-pawn squares that attack sq

def _precompute_attack_tables():
    for sq in range(64):
        f, r = sq & 7, sq >> 3
        for df, dr in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                _KING_ATTACKS[sq] |= 1 << (nr * 8 + nf)
        for df, dr in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                _KNIGHT_ATTACKS[sq] |= 1 << (nr * 8 + nf)
        # White pawn at P attacks P-9 and P-7; so P=sq+9 or P=sq+7 attacks sq.
        p = sq + 9
        if p < 64 and f < 7:
            _WP_ATTACK_FROM[sq] |= 1 << p
        p = sq + 7
        if p < 64 and f > 0:
            _WP_ATTACK_FROM[sq] |= 1 << p
        # Black pawn at P attacks P+9 and P+7; so P=sq-9 or P=sq-7 attacks sq.
        p = sq - 9
        if p >= 0 and f > 0:
            _BP_ATTACK_FROM[sq] |= 1 << p
        p = sq - 7
        if p >= 0 and f < 7:
            _BP_ATTACK_FROM[sq] |= 1 << p

_precompute_attack_tables()


def _lsb_iter(bb):
    """Yield indices of set bits from LSB to MSB (Kernighan's trick)."""
    n = int(bb)
    while n:
        lsb = n & (-n)
        yield lsb.bit_length() - 1
        n ^= lsb


class Board:
    def __init__(self):
        self.bitboards = {
            'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0, 'K': 0,
            'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0, 'k': 0,
        }
        self.white_to_move = True
        self.castling_rights = {'K': True, 'Q': True, 'k': True, 'q': True}
        self.en_passant_square = None
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self._state_stack = []
        self.white_occ = np.uint64(0)
        self.black_occ = np.uint64(0)
        self.all_occ = np.uint64(0)
        self.set_starting_position()

    def set_starting_position(self):
        # Index 0 = a8 (top-left), index 63 = h1 (bottom-right)
        # index = rank_from_top * 8 + file  (rank 0 = rank 8 in chess)
        # Store as Python ints for fast bit ops in make_move/_rebuild_occ.
        self.bitboards['P'] = 0x00FF000000000000  # a2-h2 (48-55)
        self.bitboards['N'] = 0x4200000000000000  # b1, g1
        self.bitboards['B'] = 0x2400000000000000  # c1, f1
        self.bitboards['R'] = 0x8100000000000000  # a1, h1
        self.bitboards['Q'] = 0x0800000000000000  # d1
        self.bitboards['K'] = 0x1000000000000000  # e1
        self.bitboards['p'] = 0x000000000000FF00  # a7-h7 (8-15)
        self.bitboards['n'] = 0x0000000000000042  # b8, g8
        self.bitboards['b'] = 0x0000000000000024  # c8, f8
        self.bitboards['r'] = 0x0000000000000081  # a8, h8
        self.bitboards['q'] = 0x0000000000000008  # d8
        self.bitboards['k'] = 0x0000000000000010  # e8
        self.white_to_move = True
        self.castling_rights = {'K': True, 'Q': True, 'k': True, 'q': True}
        self.en_passant_square = None
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self._rebuild_occ()

    def _rebuild_occ(self):
        bb = self.bitboards
        # bitboard values are plain Python ints after set_starting_position /
        # make_move; int() below handles numpy uint64 from test setups.
        self.white_occ = int(bb['P']) | int(bb['N']) | int(bb['B']) | int(bb['R']) | int(bb['Q']) | int(bb['K'])
        self.black_occ = int(bb['p']) | int(bb['n']) | int(bb['b']) | int(bb['r']) | int(bb['q']) | int(bb['k'])
        self.all_occ = self.white_occ | self.black_occ

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
        white = self.white_to_move
        all_moves = (
            self.generate_pawn_moves(white)
            + self.generate_knight_moves(white)
            + self.generate_king_moves(white)
            + self.generate_rook_moves(white)
            + self.generate_bishop_moves(white)
            + self.generate_queen_moves(white)
        )
        captures = [m for m in all_moves if m.capture or m.en_passant or m.promotion]
        quiet = [m for m in all_moves if not (m.capture or m.en_passant or m.promotion)]
        return captures + quiet

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
            promo_pieces = ['Q', 'R', 'B', 'N']
        else:
            pawns = self.bitboards['p']
            direction = 8
            start_rank = 1
            promotion_rank = 7
            promo_pieces = ['q', 'r', 'b', 'n']

        all_occ_int = self.all_occ
        enemy_occ_int = self.black_occ if white else self.white_occ

        for sq in _lsb_iter(pawns):
            to_sq = sq + direction
            if 0 <= to_sq < 64 and not (all_occ_int & (1 << to_sq)):
                if to_sq >> 3 == promotion_rank:
                    for promo in promo_pieces:
                        moves.append(Move(sq, to_sq, promotion=promo))
                else:
                    moves.append(Move(sq, to_sq))
                    if sq >> 3 == start_rank:
                        to_sq2 = sq + 2 * direction
                        if not (all_occ_int & (1 << to_sq2)):
                            moves.append(Move(sq, to_sq2))

            for cap_dir in (-1, 1):
                cap_sq = sq + direction + cap_dir
                if 0 <= cap_sq < 64 and abs((cap_sq & 7) - (sq & 7)) == 1:
                    if enemy_occ_int & (1 << cap_sq):
                        if cap_sq >> 3 == promotion_rank:
                            for promo in promo_pieces:
                                moves.append(Move(sq, cap_sq, promotion=promo, capture=True))
                        else:
                            moves.append(Move(sq, cap_sq, capture=True))
                    if self.en_passant_square is not None and cap_sq == self.en_passant_square:
                        moves.append(Move(sq, self.en_passant_square, en_passant=True))

        return moves

    def generate_knight_moves(self, white):
        moves = []
        knights = self.bitboards['N'] if white else self.bitboards['n']
        own_occ = self.white_occ if white else self.black_occ
        enemy_occ = self.black_occ if white else self.white_occ

        for sq in _lsb_iter(knights):
            targets = _KNIGHT_ATTACKS[sq] & ~own_occ
            n = targets
            while n:
                lsb = n & (-n)
                to_sq = lsb.bit_length() - 1
                moves.append(Move(sq, to_sq, capture=bool(enemy_occ & lsb)))
                n ^= lsb
        return moves

    def generate_king_moves(self, white):
        moves = []
        king = self.bitboards['K'] if white else self.bitboards['k']
        own_occ = self.white_occ if white else self.black_occ
        enemy_occ = self.black_occ if white else self.white_occ

        for sq in _lsb_iter(king):
            targets = _KING_ATTACKS[sq] & ~own_occ
            n = targets
            while n:
                lsb = n & (-n)
                to_sq = lsb.bit_length() - 1
                moves.append(Move(sq, to_sq, capture=bool(enemy_occ & lsb)))
                n ^= lsb
            moves += self._generate_castling_moves(white, sq)
        return moves

    def _generate_castling_moves(self, white, king_sq):
        moves = []
        all_occ_int = self.all_occ
        if white:
            if self.castling_rights['K'] and not (all_occ_int & ((1 << 61) | (1 << 62))):
                moves.append(Move(king_sq, 62, castle='kingside'))
            if self.castling_rights['Q'] and not (all_occ_int & ((1 << 57) | (1 << 58) | (1 << 59))):
                moves.append(Move(king_sq, 58, castle='queenside'))
        else:
            if self.castling_rights['k'] and not (all_occ_int & ((1 << 5) | (1 << 6))):
                moves.append(Move(king_sq, 6, castle='kingside'))
            if self.castling_rights['q'] and not (all_occ_int & ((1 << 1) | (1 << 2) | (1 << 3))):
                moves.append(Move(king_sq, 2, castle='queenside'))
        return moves

    def generate_rook_moves(self, white):
        return self._generate_sliding_moves(white, 'R', 'r', (8, -8, 1, -1))

    def generate_bishop_moves(self, white):
        return self._generate_sliding_moves(white, 'B', 'b', (9, -9, 7, -7))

    def generate_queen_moves(self, white):
        return self._generate_sliding_moves(white, 'Q', 'q', (8, -8, 1, -1, 9, -9, 7, -7))

    def _generate_sliding_moves(self, white, white_piece, black_piece, directions):
        moves = []
        piece = white_piece if white else black_piece
        bitboard = self.bitboards[piece]
        own_occ = self.white_occ if white else self.black_occ
        enemy_occ = self.black_occ if white else self.white_occ
        all_occ_int = self.all_occ

        for sq in _lsb_iter(bitboard):
            sq_file = sq & 7
            for direction in directions:
                expected_df = _DIR_FILE_DELTA[direction + 9]
                to_sq = sq
                prev_file = sq_file
                while True:
                    to_sq += direction
                    if not (0 <= to_sq < 64):
                        break
                    cur_file = to_sq & 7
                    if cur_file - prev_file != expected_df:
                        break
                    prev_file = cur_file
                    mask = 1 << to_sq
                    if not (all_occ_int & mask):
                        moves.append(Move(sq, to_sq))
                    elif enemy_occ & mask:
                        moves.append(Move(sq, to_sq, capture=True))
                        break
                    else:
                        break
        return moves

    # ------------------------------------------------------------------ #
    #  Make / Undo                                                         #
    # ------------------------------------------------------------------ #

    def make_move(self, move):
        # Snapshot stores numpy uint64 references directly — they are immutable,
        # so in-place modifications to self.bitboards create new objects and
        # leave the snapshotted values untouched.
        cr = self.castling_rights
        bb = self.bitboards
        self._state_stack.append((
            (bb['P'], bb['N'], bb['B'], bb['R'], bb['Q'], bb['K'],
             bb['p'], bb['n'], bb['b'], bb['r'], bb['q'], bb['k']),
            self.white_occ, self.black_occ, self.all_occ,
            self.white_to_move,
            (cr['K'], cr['Q'], cr['k'], cr['q']),
            self.en_passant_square,
            self.halfmove_clock,
            self.fullmove_number,
        ))

        from_mask = 1 << move.from_square
        moving_piece = None
        side_pieces = ('P','N','B','R','Q','K') if self.white_to_move else ('p','n','b','r','q','k')
        for piece in side_pieces:
            if int(bb[piece]) & from_mask:
                moving_piece = piece
                break
        if moving_piece is None:
            raise ValueError(f"No piece on square {move.from_square}")

        from_clr = ~from_mask  # Python int bitwise NOT; AND with uint64 is safe
        bb[moving_piece] = int(bb[moving_piece]) & from_clr

        if move.capture or move.en_passant:
            cap_sq = move.to_square
            if move.en_passant:
                cap_sq = move.to_square + (8 if self.white_to_move else -8)
            cap_clr = ~(1 << cap_sq)
            enemy_pieces = ('p','n','b','r','q','k') if self.white_to_move else ('P','N','B','R','Q','K')
            for piece in enemy_pieces:
                val = int(bb[piece])
                if val & (1 << cap_sq):
                    bb[piece] = val & cap_clr
                    break

        dest_piece = move.promotion if move.promotion else moving_piece
        bb[dest_piece] = int(bb[dest_piece]) | (1 << move.to_square)

        if move.castle == 'kingside':
            if self.white_to_move:
                bb['R'] = (int(bb['R']) & ~(1 << 63)) | (1 << 61)
            else:
                bb['r'] = (int(bb['r']) & ~(1 << 7)) | (1 << 5)
        elif move.castle == 'queenside':
            if self.white_to_move:
                bb['R'] = (int(bb['R']) & ~(1 << 56)) | (1 << 59)
            else:
                bb['r'] = (int(bb['r']) & ~(1 << 0)) | (1 << 3)

        if moving_piece.upper() == 'P' and abs(move.from_square - move.to_square) == 16:
            self.en_passant_square = (move.from_square + move.to_square) // 2
        else:
            self.en_passant_square = None

        if moving_piece == 'K':
            self.castling_rights['K'] = False
            self.castling_rights['Q'] = False
        elif moving_piece == 'k':
            self.castling_rights['k'] = False
            self.castling_rights['q'] = False
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
        self._rebuild_occ()

    def undo_move(self, move):
        if not self._state_stack:
            raise ValueError("No state to undo.")
        bb_vals, wo, bo, ao, wtm, cr_tuple, ep, hmc, fmn = self._state_stack.pop()
        bb = self.bitboards
        (bb['P'], bb['N'], bb['B'], bb['R'], bb['Q'], bb['K'],
         bb['p'], bb['n'], bb['b'], bb['r'], bb['q'], bb['k']) = bb_vals
        self.white_occ = wo
        self.black_occ = bo
        self.all_occ = ao
        self.white_to_move = wtm
        self.castling_rights = {
            'K': cr_tuple[0], 'Q': cr_tuple[1],
            'k': cr_tuple[2], 'q': cr_tuple[3],
        }
        self.en_passant_square = ep
        self.halfmove_clock = hmc
        self.fullmove_number = fmn

    # ------------------------------------------------------------------ #
    #  Check / attack detection                                            #
    # ------------------------------------------------------------------ #

    def is_in_check_for(self, white):
        """True if the given side's king is currently attacked by the opponent."""
        king_bb = self.bitboards['K' if white else 'k']
        for sq in _lsb_iter(king_bb):
            return self.is_square_attacked(sq, not white)
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
            pawns   = int(self.bitboards['P'])
            knights = int(self.bitboards['N'])
            bishops = int(self.bitboards['B'])
            rooks   = int(self.bitboards['R'])
            queens  = int(self.bitboards['Q'])
            king    = int(self.bitboards['K'])
            pawn_mask = _WP_ATTACK_FROM[square]
        else:
            pawns   = int(self.bitboards['p'])
            knights = int(self.bitboards['n'])
            bishops = int(self.bitboards['b'])
            rooks   = int(self.bitboards['r'])
            queens  = int(self.bitboards['q'])
            king    = int(self.bitboards['k'])
            pawn_mask = _BP_ATTACK_FROM[square]

        if pawns & pawn_mask:
            return True
        if knights & _KNIGHT_ATTACKS[square]:
            return True
        if king & _KING_ATTACKS[square]:
            return True

        all_occ_int = self.all_occ
        sq_file = square & 7

        bq = bishops | queens
        if bq:
            for direction in (9, -9, 7, -7):
                expected_df = _DIR_FILE_DELTA[direction + 9]
                sq = square
                prev_file = sq_file
                while True:
                    sq += direction
                    if not (0 <= sq < 64):
                        break
                    cur_file = sq & 7
                    if cur_file - prev_file != expected_df:
                        break
                    prev_file = cur_file
                    if all_occ_int & (1 << sq):
                        if bq & (1 << sq):
                            return True
                        break

        rq = rooks | queens
        if rq:
            for direction in (8, -8, 1, -1):
                expected_df = _DIR_FILE_DELTA[direction + 9]
                sq = square
                prev_file = sq_file
                while True:
                    sq += direction
                    if not (0 <= sq < 64):
                        break
                    cur_file = sq & 7
                    if cur_file - prev_file != expected_df:
                        break
                    prev_file = cur_file
                    if all_occ_int & (1 << sq):
                        if rq & (1 << sq):
                            return True
                        break

        return False

    # ------------------------------------------------------------------ #
    #  Geometry helpers (kept for reference, no longer used in hot paths) #
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
        return not (self.all_occ & (1 << square))

    def is_enemy(self, square, white):
        if not (0 <= square < 64):
            return False
        enemy_occ = self.black_occ if white else self.white_occ
        return bool(enemy_occ & (1 << square))

    def print_board(self):
        files = 'a b c d e f g h'
        print('    ' + files)
        for rank in range(7, -1, -1):
            row = []
            for file in range(8):
                sq = rank * 8 + file
                mask = 1 << sq
                piece = '.'
                for p, bb in self.bitboards.items():
                    if int(bb) & mask:
                        piece = p
                        break
                row.append(piece)
            print(f"{rank + 1} | {' '.join(row)} | {rank + 1}")
        print('    ' + files)

    def copy(self):
        import copy
        return copy.deepcopy(self)
