#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$(pwd)
uv run app/usecases/mcts_player.py
