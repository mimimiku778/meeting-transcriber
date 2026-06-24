.PHONY: help dev install test lint format clean

help:
	@echo "make dev      - 開発依存をインストール (pip install -e .[dev])"
	@echo "make install  - install.sh (MCP登録/スキル配置/CLI)"
	@echo "make test     - pytest"
	@echo "make lint     - ruff check"
	@echo "make format   - ruff の自動修正＋整形"
	@echo "make clean    - キャッシュ削除"

dev:
	pip install -e ".[dev]"

install:
	./install.sh

test:
	pytest

lint:
	ruff check src tests

format:
	ruff check --fix src tests
	ruff format src tests

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
