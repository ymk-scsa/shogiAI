from typing import Callable
from pydantic import BaseModel
import cshogi
from cshogi import (
    BLACK,
    WHITE,
    PIECE_TYPES,
    MAX_PIECES_IN_HAND,
    HAND_PIECES,
    move_is_drop,
    move_to,
    move_from,
    move_is_promotion,
    move_drop_hand_piece,
    BLACK_WIN,
    WHITE_WIN,
)
import numpy as np
from app.domain.moves import make_himo_features, make_kiki_features, DIRECTION_NUM

# 移動方向を表す定数
MOVE_DIRECTION = [
    UP,
    UP_LEFT,
    UP_RIGHT,
    LEFT,
    RIGHT,
    DOWN,
    DOWN_LEFT,
    DOWN_RIGHT,
    UP2_LEFT,
    UP2_RIGHT,
    UP_PROMOTE,
    UP_LEFT_PROMOTE,
    UP_RIGHT_PROMOTE,
    LEFT_PROMOTE,
    RIGHT_PROMOTE,
    DOWN_PROMOTE,
    DOWN_LEFT_PROMOTE,
    DOWN_RIGHT_PROMOTE,
    UP2_LEFT_PROMOTE,
    UP2_RIGHT_PROMOTE,
] = range(20)

# 入力特徴量の数
FEATURES_NUM = len(PIECE_TYPES) * 2 + sum(MAX_PIECES_IN_HAND) * 2
FEATURES_KIKI_NUM = len(PIECE_TYPES) * 2 + sum(MAX_PIECES_IN_HAND) * 2 + DIRECTION_NUM * 2
FEATURES_HIMO_NUM = len(PIECE_TYPES) * 2 + sum(MAX_PIECES_IN_HAND) * 2 + DIRECTION_NUM * 2
FEATURES_SMALL_NUM = len(PIECE_TYPES) * 2 + len(MAX_PIECES_IN_HAND) * 2

# 移動を表すラベルの数
MOVE_PLANES_NUM = len(MOVE_DIRECTION) + len(HAND_PIECES)
MOVE_LABELS_NUM = MOVE_PLANES_NUM * 81

FEATURES_MODE = [
    FEATURES_DEFAULT,
    FEATURES_KIKI,
    FEATURES_HIMO,
    FEATURES_SMALL,
] = range(4)


# 入力特徴量を作成
def make_input_features(board: cshogi.Board, features: np.ndarray) -> None:
    # 入力特徴量を0に初期化
    features.fill(0)

    # 盤上の駒
    if board.turn == BLACK:
        board.piece_planes(features)
        pieces_in_hand = board.pieces_in_hand
    else:
        board.piece_planes_rotate(features)
        pieces_in_hand = reversed(board.pieces_in_hand)
    # 持ち駒
    i = 28
    for hands in pieces_in_hand:
        for num, max_num in zip(hands, MAX_PIECES_IN_HAND):
            features[i : i + num].fill(1)
            i += max_num


def make_input_features_kiki(board: cshogi.Board, features: np.ndarray) -> None:
    make_input_features(board, features)
    make_kiki_features(board, features)


def make_input_features_himo(board: cshogi.Board, features: np.ndarray) -> None:
    make_input_features(board, features)
    make_himo_features(board, features)


def make_input_features_small(board: cshogi.Board, features: np.ndarray) -> None:
    # 入力特徴量を0に初期化
    features.fill(0)

    # 盤上の駒
    if board.turn == BLACK:
        board.piece_planes(features)
        pieces_in_hand = board.pieces_in_hand
    else:
        board.piece_planes_rotate(features)
        pieces_in_hand = reversed(board.pieces_in_hand)
    # 持ち駒
    i = 28
    for hands in pieces_in_hand:
        for num, max_num in zip(hands, MAX_PIECES_IN_HAND):
            fill = 81 * num // max_num - 1
            features[i][0 : fill // 9].fill(1)
            features[i][fill // 9][0 : fill % 9].fill(1)
            i += 1


class FeaturesSetting(BaseModel):
    make_features: Callable[[cshogi.Board, np.ndarray], None] = make_input_features
    features_num: int = FEATURES_NUM


FEATURES_SETTINGS = {
    FEATURES_DEFAULT: FeaturesSetting(),
    FEATURES_KIKI: FeaturesSetting(
        make_features=make_input_features_kiki,
        features_num=FEATURES_KIKI_NUM,
    ),
    FEATURES_HIMO: FeaturesSetting(
        make_features=make_input_features_kiki,
        features_num=FEATURES_HIMO_NUM,
    ),
    FEATURES_SMALL: FeaturesSetting(
        make_features=make_input_features_small,
        features_num=FEATURES_SMALL_NUM,
    ),
}


# 移動を表すラベルを作成(方策ネットワーク出力)
def make_move_label(move: int, color: int) -> int:
    if not move_is_drop(move):  # 駒の移動
        to_sq: int = move_to(move)  # 移動元
        from_sq = move_from(move)  # 移動先

        # 後手の場合盤を回転
        if color == WHITE:
            to_sq = 80 - to_sq
            from_sq = 80 - from_sq

        # 移動方向を割り出す
        to_x, to_y = divmod(to_sq, 9)
        from_x, from_y = divmod(from_sq, 9)
        dir_x = to_x - from_x
        dir_y = to_y - from_y
        if dir_y < 0:
            if dir_x == 0:
                move_direction = UP
            elif dir_y == -2 and dir_x == -1:
                move_direction = UP2_RIGHT
            elif dir_y == -2 and dir_x == 1:
                move_direction = UP2_LEFT
            elif dir_x < 0:
                move_direction = UP_RIGHT
            else:  # dir_x > 0
                move_direction = UP_LEFT
        elif dir_y == 0:
            if dir_x < 0:
                move_direction = RIGHT
            else:  # dir_x > 0
                move_direction = LEFT
        else:  # dir_y > 0
            if dir_x == 0:
                move_direction = DOWN
            elif dir_x < 0:
                move_direction = DOWN_RIGHT
            else:  # dir_x > 0
                move_direction = DOWN_LEFT

        # 成り
        if move_is_promotion(move):
            move_direction += 10
    else:  # 駒打ち
        to_sq = move_to(move)
        # 後手の場合盤を回転
        if color == WHITE:
            to_sq = 80 - to_sq

        # 駒打ちの移動方向
        move_direction = len(MOVE_DIRECTION) + move_drop_hand_piece(move)

    return int(move_direction * 81 + to_sq)


# 対局結果から報酬を作成(価値ネットワーク出力)
def make_result(game_result: int, color: int) -> float:
    if color == BLACK:
        if game_result == BLACK_WIN:
            return 1
        if game_result == WHITE_WIN:
            return 0
    else:
        if game_result == BLACK_WIN:
            return 0
        if game_result == WHITE_WIN:
            return 1
    return 0.5
    # dlshogi 1/6
