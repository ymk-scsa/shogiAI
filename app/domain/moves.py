import cshogi
import numpy as np
from cshogi import (
    PIECE_TYPES,
    MAX_PIECES_IN_HAND,
)

# 移動方向を表す定数
UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT = range(10)

MOVE_OFFSET = {
    UP: (-1, 0),
    UP_LEFT: (-1, -1),
    UP_RIGHT: (-1, 1),
    LEFT: (0, -1),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    DOWN_LEFT: (1, -1),
    DOWN_RIGHT: (1, 1),
    UP2_LEFT: (-2, -1),
    UP2_RIGHT: (-2, 1),
}

# 駒の種類と方向の対応
PIECE_DIRECTIONS = {
    cshogi.PAWN: ([UP], []),
    cshogi.LANCE: ([], [UP]),
    cshogi.KNIGHT: ([UP2_LEFT, UP2_RIGHT], []),
    cshogi.SILVER: ([UP_RIGHT, UP, DOWN_RIGHT, UP_LEFT, DOWN_LEFT], []),
    cshogi.GOLD: ([UP_RIGHT, UP, UP_LEFT, RIGHT, LEFT, DOWN], []),
    cshogi.BISHOP: ([], [UP_RIGHT, DOWN_RIGHT, UP_LEFT, DOWN_LEFT]),
    cshogi.ROOK: ([], [UP, DOWN, RIGHT, LEFT]),
    cshogi.KING: ([UP_RIGHT, UP, DOWN_RIGHT, RIGHT, LEFT, UP_LEFT, DOWN, DOWN_LEFT], []),
    cshogi.PROM_PAWN: ([UP_RIGHT, UP, UP_LEFT, RIGHT, LEFT, DOWN], []),
    cshogi.PROM_LANCE: ([UP_RIGHT, UP, UP_LEFT, RIGHT, LEFT, DOWN], []),
    cshogi.PROM_KNIGHT: ([UP_RIGHT, UP, UP_LEFT, RIGHT, LEFT, DOWN], []),
    cshogi.PROM_SILVER: ([UP_RIGHT, UP, UP_LEFT, RIGHT, LEFT, DOWN], []),
    cshogi.PROM_BISHOP: ([UP, DOWN, RIGHT, LEFT], [UP_RIGHT, DOWN_RIGHT, UP_LEFT, DOWN_LEFT]),
    cshogi.PROM_ROOK: ([UP_RIGHT, DOWN_RIGHT, UP_LEFT, DOWN_LEFT], [UP, DOWN, RIGHT, LEFT]),
}

BASE_INDEX = len(PIECE_TYPES) * 2 + sum(MAX_PIECES_IN_HAND) * 2
DIRECTION_NUM = len(MOVE_OFFSET)


# 駒の利きを表現する関数
def make_kiki_features(board: cshogi.Board, features: np.ndarray) -> None:
    attack_board = board.copy()

    for square in cshogi.SQUARES:
        piece = attack_board.piece(square)
        if piece is not None:
            piece_type = cshogi.piece_to_piece_type(piece)
            piece_color = 0 if piece == piece_type else 1
            piece_direction_list = PIECE_DIRECTIONS.get(piece_type, ([], []))

            # 自分の駒と相手の駒で処理を共通化
            for is_own in [True, False]:
                if (is_own and piece_color == board.turn) or (not is_own and piece_color != board.turn):
                    direction_offset = 0 if is_own else DIRECTION_NUM

                    # 1マス移動
                    for direction in piece_direction_list[0]:
                        move_y, move_x = MOVE_OFFSET[direction]
                        x = square // 9 + move_x
                        y = square % 9 + move_y * (-1 if not is_own else 1)

                        if 0 <= y and y < 9 and 0 <= x and x < 9:
                            features[BASE_INDEX + direction_offset + direction][x][y] = 1

                    # 複数マス移動
                    for direction in piece_direction_list[1]:
                        move_y, move_x = MOVE_OFFSET[direction]
                        x = square // 9
                        y = square % 9
                        while True:
                            x += move_x
                            y += move_y * (-1 if not is_own else 1)
                            if not (0 <= y and y < 9 and 0 <= x and x < 9):
                                break
                            features[BASE_INDEX + direction_offset + direction][x][y] = 1
                            if attack_board.piece_type(y + x * 9) != 0:
                                break


# 駒のヒモを表現する関数
def make_himo_features(board: cshogi.Board, features: np.ndarray) -> None:
    attack_board = board.copy()

    for square in cshogi.SQUARES:
        piece = attack_board.piece(square)
        if piece is not None:
            piece_type = cshogi.piece_to_piece_type(piece)
            piece_color = 0 if piece == piece_type else 1
            piece_direction_list = PIECE_DIRECTIONS.get(piece_type, ([], []))

            # 自分の駒と相手の駒で処理を共通化
            for is_own in [True, False]:
                if (is_own and piece_color == board.turn) or (not is_own and piece_color != board.turn):
                    direction_offset = 0 if is_own else DIRECTION_NUM

                    # 1マス移動
                    for direction in piece_direction_list[0]:
                        move_y, move_x = MOVE_OFFSET[direction]
                        x = square // 9 + move_x
                        y = square % 9 + move_y * (-1 if not is_own else 1)

                        if 0 <= y and y < 9 and 0 <= x and x < 9 and attack_board.piece_type(y + x * 9) != 0:
                            features[BASE_INDEX + direction_offset + direction][x][y] = 1

                    # 複数マス移動
                    for direction in piece_direction_list[1]:
                        move_y, move_x = MOVE_OFFSET[direction]
                        x = square // 9
                        y = square % 9
                        while True:
                            x += move_x
                            y += move_y * (-1 if not is_own else 1)
                            if not (0 <= y and y < 9 and 0 <= x and x < 9):
                                break
                            if attack_board.piece_type(y + x * 9) != 0:
                                features[BASE_INDEX + direction_offset + direction][x][y] = 1
                                break
