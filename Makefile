PYTHON   ?= python3
VENV     := .venv
PIP      := $(VENV)/bin/pip
PY       := $(VENV)/bin/python

# Corpus sizes — override on the command line, e.g. make corpus-fm N=20000
N        ?= 50000
SR       ?= 48000
EPOCHS   ?= 200
LR       ?= 3e-4
BATCH    ?= 256

DATA_DIR  := data
CKPT_DIR  := checkpoints

# ── Environment ────────────────────────────────────────────────────────────────

.PHONY: install
install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-revival.txt

# ── Corpus generation (CLAP embedding pre-computation) ─────────────────────────

.PHONY: corpus-fm corpus-wt corpus-sub corpus

corpus-fm: $(DATA_DIR)/fm.npz
$(DATA_DIR)/fm.npz:
	mkdir -p $(DATA_DIR)
	$(PY) -m training.generate_embeddings \
		--synth fm --n $(N) --sr $(SR) --out $(DATA_DIR)/fm.npz

corpus-wt: $(DATA_DIR)/wt.npz
$(DATA_DIR)/wt.npz:
	mkdir -p $(DATA_DIR)
	$(PY) -m training.generate_embeddings \
		--synth wt --n $(N) --sr $(SR) --out $(DATA_DIR)/wt.npz

corpus-sub: $(DATA_DIR)/sub.npz
$(DATA_DIR)/sub.npz:
	mkdir -p $(DATA_DIR)
	$(PY) -m training.generate_embeddings \
		--synth sub --n $(N) --sr $(SR) --out $(DATA_DIR)/sub.npz

corpus: corpus-fm corpus-wt corpus-sub

# ── Training ───────────────────────────────────────────────────────────────────

.PHONY: train-fm train-wt train-sub train

train-fm: $(DATA_DIR)/fm.npz
	mkdir -p $(CKPT_DIR)
	$(PY) -m training.train \
		--synth fm --data $(DATA_DIR)/fm.npz \
		--out $(CKPT_DIR)/fm.pt \
		--epochs $(EPOCHS) --lr $(LR) --batch $(BATCH)

train-wt: $(DATA_DIR)/wt.npz
	mkdir -p $(CKPT_DIR)
	$(PY) -m training.train \
		--synth wavetable --data $(DATA_DIR)/wt.npz \
		--out $(CKPT_DIR)/wt.pt \
		--epochs $(EPOCHS) --lr $(LR) --batch $(BATCH)

train-sub: $(DATA_DIR)/sub.npz
	mkdir -p $(CKPT_DIR)
	$(PY) -m training.train \
		--synth subtractive --data $(DATA_DIR)/sub.npz \
		--out $(CKPT_DIR)/sub.pt \
		--epochs $(EPOCHS) --lr $(LR) --batch $(BATCH)

# Full pipeline: generate all corpora, then train all heads
train: corpus train-fm train-wt train-sub

# ── Inference ──────────────────────────────────────────────────────────────────
# make infer AUDIO=patch.flac SYNTH=fm MIDI="52 54 56 59 61"

AUDIO ?= patch.flac
SYNTH ?= fm
MIDI  ?= 60
OUT   ?= output.wav

.PHONY: infer
infer: $(CKPT_DIR)/$(SYNTH).pt
	$(PY) -m inference.infer \
		--audio $(AUDIO) \
		--model $(CKPT_DIR)/$(SYNTH).pt \
		--midi $(MIDI) \
		--out $(OUT)

# ── Utilities ──────────────────────────────────────────────────────────────────

.PHONY: clean-data clean-checkpoints clean
clean-data:
	rm -f $(DATA_DIR)/*.npz

clean-checkpoints:
	rm -f $(CKPT_DIR)/*.pt

clean: clean-data clean-checkpoints
