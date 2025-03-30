# Shogi AI

## ディレクトリ構造
```
shogi_ai/
│── app/                    # アプリケーション全体を管理
│   │── main.py             # エントリーポイント
│   ├── interfaces/         # インターフェース（UI, API, CLI）
│   ├── usecases/           # ユースケース（アプリケーションの主要ロジック）
│   ├── domain/             # ビジネスルール（将棋のルール・AIロジック）
│   ├── infrastructure/     # インフラ層（データ保存、通信）
│   ├── config.py           # 設定ファイル
│── tests/                  # テストコード
│── models/                 # 学習済みモデルの保存
│── logs/                   # ログの保存
│── README.md               # プロジェクトの説明
│── requirements.txt        # 依存ライブラリ
│── pyproject.toml          # uvで管理する設定
```