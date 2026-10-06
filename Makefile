export UV_CACHE_DIR := $(CURDIR)/.uv-cache
export DVC_SITE_CACHE_DIR := $(CURDIR)/.cache/dvc
export TOKENIZERS_PARALLELISM := false
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib

.PHONY: install train train-all train-freeze compare plot check verify repro status

install:
	uv sync --locked

# Два прогона и график. DVC пропускает неизменённые тяжёлые стадии.
train:
	uv run dvc repro plot

train-all:
	uv run dvc repro train_all

train-freeze:
	uv run dvc repro train_freeze

plot:
	uv run dvc repro plot

# Базовая модель против адаптера на пяти фиксированных промптах.
compare:
	uv run dvc repro compare

check:
	mkdir -p .check-tmp
	TMPDIR=$(CURDIR)/.check-tmp bash tests/check.sh

verify:
	uv run python -m unittest discover -s tests -p 'test_*.py' -v

repro:
	uv run dvc repro

status:
	uv run dvc status
