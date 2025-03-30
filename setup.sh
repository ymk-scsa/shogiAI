#!/bin/bash

set -e  # エラー時にスクリプトを停止

# UV のインストール確認
if ! command -v uv &> /dev/null
then
    echo "uv が見つかりません。手動でインストールしてください。"
    echo "公式サイト: https://astral.sh/uv"
    exit 1
fi

echo "✅ uv がインストールされています。"

# 仮想環境のセットアップ
echo "🔧 仮想環境を作成・有効化中..."
uv venv
source .venv/bin/activate

echo "✅ 仮想環境がセットアップされました。"

# 依存関係のインストール
echo "📦 依存関係をインストール中..."
uv sync

echo "✅ 依存関係のインストールが完了しました。"

# pre-commit のセットアップ
echo "🔧 pre-commit をインストール中..."
uv run pre-commit install

echo "✅ pre-commit のセットアップが完了しました。"

# direnv のインストール確認
if ! command -v direnv &> /dev/null
then
    echo "direnv が見つかりません。手動でインストールしてください。"
    exit 1
fi

# direnvのセットアップ
echo "🔧 direnvを有効化中..."
direnv allow .

echo "✅ direnvがセットアップされました。"

# torch関係のセットアップ(GPUごとにバージョンを分けてもいい)
echo "🔧 pytorchのセットアップ中..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo "🚀 セットアップが完了しました！"
