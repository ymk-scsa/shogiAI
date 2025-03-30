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
parser.add_argument("hcpe_train")
parser.add_argument("hcpe_test")
parser.add_argument("--test_ratio", type=float, default=0.1)
parser.add_argument("--filter_moves", type=int, default=50, help="filter game records with moves less than this value")
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
hcpes = np.zeros(1024, HuffmanCodedPosAndEval)

# 出力ファイル
f_train = open(args.hcpe_train, "wb")
f_test = open(args.hcpe_test, "wb")

board = Board()
for file_list, f in zip([file_list_train, file_list_test], [f_train, f_test]):
    kif_num = 0
    position_num = 0

    for filepath in file_list:
        try:
            kif_data = shogi.KIF.Parser.parse_file(filepath)[0]
        except Exception as e:
            print(f"Failed to parse {filepath}: {e}")
            continue

        if len(kif_data["moves"]) < args.filter_moves:
            continue

        try:
            board.set_sfen(kif_data["sfen"])
            p = 0
            for usi_move in kif_data["moves"]:
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
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

        if p == 0:
            continue

        hcpes[:p].tofile(f)
        kif_num += 1
        position_num += p

    print("kif_num", kif_num)
    print("position_num", position_num)
