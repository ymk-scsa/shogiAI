from cshogi import HuffmanCodedPosAndEval, Board, BLACK, move16, move_to_usi
from cshogi import CSA
import numpy as np
import os
import glob
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
    description="Convert CSA format game records to a hcpe file", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("csa_dir", help="the directory where csa files are stored")
parser.add_argument("hcpe_train")
parser.add_argument("hcpe_test")
parser.add_argument("--out_draw", action="store_true", help="output draw game records")
parser.add_argument("--out_maxmove", action="store_true", help="output maxmove game records")
parser.add_argument("--out_noeval", action="store_true", help="output positions without eval")
parser.add_argument("--out_mate", action="store_true", help="output mated positions")
parser.add_argument("--out_brinkmate", action="store_true")
parser.add_argument("--uniq", action="store_true")
parser.add_argument("--eval", type=int, help="eval threshold")
parser.add_argument("--filter_moves", type=int, default=50, help="filter game records with moves less than this value")
parser.add_argument(
    "--filter_rating", type=int, default=3500, help="filter game records with both ratings below this value"
)
parser.add_argument("--test_ratio", type=float, default=0.1)
args = parser.parse_args()

csa_dir = args.csa_dir
if not os.path.isdir(csa_dir):
    print(f"Error: The directory '{csa_dir}' does not exist.")
    exit(1)
else:
    print(f"The directory '{csa_dir}' exists and will be used.")

# csa_file_list の取得
print(f"Searching for .csa files in {csa_dir}")
csa_file_list = glob.glob(os.path.join(csa_dir, "**", "*.csa"), recursive=True)
print(f"Found {len(csa_file_list)} .csa files.")

filter_moves = args.filter_moves
filter_rating = args.filter_rating
endgames = ["%TORYO", "%KACHI"]
if args.out_draw:
    endgames.append("%SENNICHITE")
if args.out_maxmove:
    endgames.append("%JISHOGI")

csa_file_list = glob.glob(os.path.join(args.csa_dir, "**", "*.csa"), recursive=True)

file_list_train, file_list_test = train_test_split(csa_file_list, test_size=args.test_ratio)

hcpes = np.zeros(1024, HuffmanCodedPosAndEval)

f_train = open(args.hcpe_train, "wb")
f_test = open(args.hcpe_test, "wb")

board = Board()
for file_list, f in zip([file_list_train, file_list_test], [f_train, f_test]):
    kif_num = 0
    position_num = 0
    duplicates = set()
    for filepath in file_list:
        for kif in CSA.Parser.parse_file(filepath):
            if kif.endgame not in endgames or len(kif.moves) < filter_moves:
                continue
            if filter_rating > 0 and min(kif.ratings) < filter_rating:
                continue
            # 評価値がない棋譜を除外
            if all(comment == "" for comment in kif.comments[0::2]) or all(
                comment == "" for comment in kif.comments[1::2]
            ):
                continue
            # 重複削除
            if args.uniq:
                dup_key = "".join([move_to_usi(move) for move in kif.moves])
                if dup_key in duplicates:
                    print(f"duplicate {filepath}")
                    continue
                duplicates.add(dup_key)

            try:
                if args.out_brinkmate:
                    brinkmate_i = -1
                    if kif.endgame == "%TORYO":
                        board.set_sfen(kif.sfen)
                        for move in kif.moves:
                            assert board.is_legal(move)
                            board.push(move)
                        while board.is_check():
                            board.pop()
                            board.pop()
                        brinkmate_i = board.move_number

                board.set_sfen(kif.sfen)
                p = 0
                for i, (move, score, comment) in enumerate(zip(kif.moves, kif.scores, kif.comments)):
                    assert board.is_legal(move)
                    if not args.out_noeval and comment == "":
                        board.push(move)
                        continue
                    hcpe = hcpes[p]
                    board.to_hcp(hcpe["hcp"])
                    assert abs(score) <= 1000000
                    eval = min(32767, max(score, -32767))
                    if args.eval and abs(eval) > args.eval:
                        break
                    hcpe["eval"] = eval if board.turn == BLACK else -eval
                    hcpe["bestMove16"] = move16(move)
                    hcpe["gameResult"] = kif.win
                    p += 1
                    if args.out_brinkmate:
                        if i == brinkmate_i:
                            break
                    elif not args.out_mate and abs(score) >= 100000:
                        break
                    board.push(move)
            except Exception as e:
                print(f"skip {filepath}:{i}:{move_to_usi(move)}:{score}")
                print(e)
                continue

            if p == 0:
                continue

            hcpes[:p].tofile(f)

            kif_num += 1
            position_num += p

    print("kif_num", kif_num)
    print("position_num", position_num)
