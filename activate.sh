#!/bin/bash
# Activate CASTLE dev environment
# Usage: source activate.sh

export VENV_PATH="$HOME/.venvs/ei-castle-dev"
export GIT_DIR="$HOME/.git-cifs/ei-castle-dev"
export GIT_WORK_TREE="/mnt/AI-Assistant/ei-castle-dev"
export PATH="$VENV_PATH/bin:$PATH"

echo "✅ CASTLE dev environment activated"
echo "   Python: $(python --version)"
echo "   Project: $GIT_WORK_TREE"
echo "   Branch: $(git branch --show-current)"
echo ""
echo "   git status / git commit / git push — all work normally"
echo "   python app.py — start Gradio UI on :7860"
