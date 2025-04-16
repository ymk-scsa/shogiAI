import os
import glob
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from cshogi import HuffmanCodedPosAndEval, Board, move16
import shogi
import shogi.KIF

parser = argparse.ArgumentParser(
    description="Convert KIF format game records to a hcpe file",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("kif_dir", help="the directory where kif files are stored")
parser.add_argument("save_path")
parser.add_argument("--test_ratio", type=float, default=0.1)
parser.add_argument("--filter_moves", type=int, default=100, help="filter game records with moves less than this value")
args = parser.parse_args()

kif_dir = args.kif_dir
if not os.path.isdir(kif_dir):
    print(f"Error: The directory '{kif_dir}' does not exist.")
    exit(1)

# KIFファイルの取得
print(f"Searching for .kif files in {kif_dir}")
kif_file_list = glob.glob(os.path.join(kif_dir, "**", "*.kif"), recursive=True)
print(f"Found {len(kif_file_list)} .kif files.")

# 学習データとテストデータに分割
file_list_train, file_list_test = train_test_split(kif_file_list, test_size=args.test_ratio)

# HCPEデータバッファ
hcpes_begin = np.zeros(1024, HuffmanCodedPosAndEval)
hcpes_middle = np.zeros(1024, HuffmanCodedPosAndEval)
hcpes_end = np.zeros(1024, HuffmanCodedPosAndEval)

# 出力ファイル
f_train_begin = open(args.save_path + "_train_begin.hcpe", "wb")
f_test_begin = open(args.save_path + "_test_begin.hcpe", "wb")
f_train_middle = open(args.save_path + "_train_middle.hcpe", "wb")
f_test_middle = open(args.save_path + "_test_middle.hcpe", "wb")
f_train_end = open(args.save_path + "_train_end.hcpe", "wb")
f_test_end = open(args.save_path + "_test_end.hcpe", "wb")

board = Board()
for file_list, fb, fm, fe in zip(
    [file_list_train, file_list_test],
    [f_train_begin, f_test_begin],
    [f_train_middle, f_test_middle],
    [f_train_end, f_test_end],
):
    kif_num = 0

    for filepath in file_list:
        kif_num += 1
        if kif_num % 10000 == 0:
            print(f"kif_num: {kif_num}")
        try:
            kif_data = shogi.KIF.Parser.parse_file(filepath)[0]
        except Exception as e:
            print(f"Failed to parse {filepath}: {e}")
            continue

        if len(kif_data["moves"]) < args.filter_moves:
            continue

        try:
            board.set_sfen(kif_data["sfen"])
            moves = kif_data["moves"]
            b_index = min(max(int(len(moves) * 0.25), 25), 50)
            e_index = len(moves) - min(max(int(len(moves) * 0.2), 20), 40)
            for usi_moves, hcpes, f in zip(
                [moves[:b_index], moves[b_index:e_index], moves[e_index:]],
                [hcpes_begin, hcpes_middle, hcpes_end],
                [fb, fm, fe],
            ):
                p = 0
                hcpes.fill(0)
                for usi_move in usi_moves:
                    move = board.move_from_usi(usi_move)
                    if not board.is_legal(move):
                        print(f"is not legal move {filepath}: {move}")
                        continue

                    hcpe = hcpes[p]
                    board.to_hcp(hcpe["hcp"])
                    hcpe["eval"] = 0  # KIFは評価値がないことが多いためデフォルト0
                    hcpe["bestMove16"] = move16(move)
                    hcpe["gameResult"] = (
                        1 if kif_data["win"] == "b" else 0 if kif_data["win"] == "w" else 2
                    )  # 勝敗情報は簡単化
                    p += 1
                    board.push(move)
                if p > 0:
                    hcpes[:p].tofile(f)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

    print("kif_num", kif_num)

f_train_begin.close()
f_test_begin.close()
f_train_middle.close()
f_test_middle.close()
f_train_end.close()
f_test_end.close()
