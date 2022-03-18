MIDI_DIR ?= data/midi
AUDIO_DIR ?= data/audio
MODEL_DIR ?= data/model
DOCKER_IMAGE ?= nvalsted/autosoundmatch:latest

SYNTH_PATH ?= ./data/vst/MikaMicro64.dll

build-image:
ifeq ($(OS),Windows_NT)
	docker build --no-cache . -t ${DOCKER_IMAGE}
else
	sudo docker build --no-cache . -t ${DOCKER_IMAGE}
endif

run-image-interactive:
	docker run --rm -it ${DOCKER_IMAGE}

fetch-resources:
	@echo "NOT YET IMPLEMENTED"

build-vst:
	@echo "Building OpnTaybel VST"
	@echo "NOT YET IMPLEMENTED"

prepare-data:
	mkdir -p ${MIDI_DIR} ${AUDIO_DIR} ${MODEL_DIR}
	@echo "Setting up local database and generating data"
	poetry run python asm-cli.py setup-relational-models \
		--synth-path ${SYNTH_PATH} \
		--engine-url "sqlite:///data/local.db"
	poetry run python asm-cli.py generate-param-tuples \
		--midi-path ${MIDI_DIR} \
		--audio-path ${AUDIO_DIR}

model:
	@echo "Training main model"
	poetry run python asm-cli.py train-model \
		--model-dir ${MODEL_DIR}

model-suite:
	@echo "NOT YET IMPLEMENTED"

evaluate:
	poetry run python asm-cli.py test-model

reset:
	@echo "Resetting project state"
	poetry run python asm-cli.py reset

inspect:
	@echo "Inspecting project state"
	poetry run python asm-cli.py inspect-registry
