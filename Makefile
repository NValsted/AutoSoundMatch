MIDI_DIR ?= data/midi
AUDIO_DIR ?= data/audio
MODEL_DIR ?= data/model
DOWNLOADS_DIR ?= data/downloads
SYNTH_PATH ?= ./data/synth/MikaMicro64.dll

DOCKER_IMAGE ?= nvalsted/autosoundmatch:latest

build-image:
ifeq ($(OS),Windows_NT)
	docker build --no-cache . -t ${DOCKER_IMAGE}
else
	sudo docker build --no-cache . -t ${DOCKER_IMAGE}
endif

run-image-interactive:
	docker run --rm -it ${DOCKER_IMAGE}

resources:
	poetry run python asm-cli.py setup-paths \
		--midi ${MIDI_DIR} \
		--audio ${AUDIO_DIR} \
		--model ${MODEL_DIR} \
		--downloads ${DOWNLOADS_DIR}
	# wget http://www.cp.jku.at/resources/2019_RLScoFo_TISMIR/data.tar.gz
	# tar -xzvf data.tar.gz

build-vst:
	@echo "Building OpnTaybel VST"
	@echo "NOT YET IMPLEMENTED"

prepare-data:
	poetry run python asm-cli.py setup-relational-models \
		--engine-url "sqlite:///data/local.db" \
		--synth-path ${SYNTH_PATH}
	poetry run python asm-cli.py partition-midi-files \
		--directory data/msmd_real_performances/msmd_all_deadpan/performance
	poetry run python asm-cli.py generate-param-tuples
	poetry run python asm-cli.py process-audio

model:
	@echo "Training main model"
	poetry run python asm-cli.py train-model

model-suite:
	@echo "NOT YET IMPLEMENTED"

evaluate:
	poetry run python asm-cli.py test-model

reset:
	@echo "Resetting project state"
	poetry run python asm-cli.py reset

inspect:
	@echo "Inspecting project state"
	@poetry run python -c "import warnings; warnings.filterwarnings('ignore'); from torch.cuda import is_available; print('USING GPU' if is_available() else 'USING CPU')"
	poetry run python asm-cli.py inspect-registry
