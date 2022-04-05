MIDI_DIR ?= data/midi
AUDIO_DIR ?= data/audio
MODEL_DIR ?= data/model
DOWNLOADS_DIR ?= data/downloads

PRESETS_DIR ?= data/presets
SYNTH_PATH ?= data/synth/MikaMicro64.dll
BENCHMARK_DIVA ?= 0

DOCKER_IMAGE ?= nvalsted/autosoundmatch:latest

build-image:
ifeq ($(OS),Windows_NT)
	docker build --no-cache . -t ${DOCKER_IMAGE}
else
	sudo docker build --no-cache . -t ${DOCKER_IMAGE}
endif

run-image-interactive:
	docker run --rm -it ${DOCKER_IMAGE}

paths:
	poetry run python asm-cli.py setup-paths \
		--midi ${MIDI_DIR} \
		--audio ${AUDIO_DIR} \
		--model ${MODEL_DIR} \
		--downloads ${DOWNLOADS_DIR} \
		--presets ${PRESETS_DIR}

resources:
	@if [ ! -d "${DOWNLOADS_DIR}/msmd_real_performances" ]; \
	then \
		wget http://www.cp.jku.at/resources/2019_RLScoFo_TISMIR/data.tar.gz -O ${DOWNLOADS_DIR}/RLScoFO_data.tar.gz && \
		tar -xzf ${DOWNLOADS_DIR}/RLScoFO_data.tar.gz -C ${DOWNLOADS_DIR} && \
		rm ${DOWNLOADS_DIR}/RLScoFO_data.tar.gz; \
	else \
		echo "Resource http://www.cp.jku.at/resources/2019_RLScoFo_TISMIR/data.tar.gz already downloaded."; \
	fi

	@if [ ! -d "${DOWNLOADS_DIR}/lmd_matched" ]; \
	then \
		wget http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz -O ${DOWNLOADS_DIR}/lmd_matched.tar.gz && \
		tar -xzf ${DOWNLOADS_DIR}/lmd_matched.tar.gz -C ${DOWNLOADS_DIR} && \
		rm ${DOWNLOADS_DIR}/lmd_matched.tar.gz; \
	else \
		echo "Resource http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz already downloaded."; \
	fi

clear-resources:
	rm -r ${DOWNLOADS_DIR}/msmd_all
	rm -r ${DOWNLOADS_DIR}/msmd_real_performances
	rm -r ${DOWNLOADS_DIR}/nottingham
	rm -r ${DOWNLOADS_DIR}/lmd_matched

tables:
	poetry run python asm-cli.py setup-relational-models \
		--engine-url "sqlite:///data/local.db" \
		--synth-path ${SYNTH_PATH}

midi-partitions:
	poetry run python asm-cli.py partition-midi-files \
		--directory ${DOWNLOADS_DIR}/msmd_real_performances/msmd_all_deadpan/performance/ \
		--directory ${DOWNLOADS_DIR}/lmd_matched/A/A/

dataset:
	poetry run python asm-cli.py generate-param-triples
	poetry run python asm-cli.py process-audio

prepare-data:
	make tables
	make midi-partitions
	make dataset

model:
	@echo "Training main model"
	poetry run python asm-cli.py train-model

evaluate:
	poetry run python asm-cli.py test-model

reset:
	@echo "Resetting project state"
	poetry run python asm-cli.py reset

inspect:
	@echo "Inspecting project state"
	@poetry run python -c "import warnings; warnings.filterwarnings('ignore'); from torch.cuda import is_available; print('USING GPU' if is_available() else 'USING CPU')"
	poetry run python asm-cli.py inspect-registry

mono-benchmark:
	make reset
	poetry run python asm-cli.py setup-paths \
		--midi data/mono-midi \
		--audio data/mono-audio \
		--model ${MODEL_DIR} \
		--downloads ${DOWNLOADS_DIR} \
		--presets ${PRESETS_DIR}
ifeq (${BENCHMARK_DIVA},1)
	poetry run python asm-cli.py update-registry \
		src/config/fixtures/aiflowsynth/u-he_diva32.py
else
	poetry run python asm-cli.py update-registry \
		src/config/fixtures/synth32.py
endif
	poetry run python asm-cli.py setup-relational-models \
		--engine-url "sqlite:///data/local.db"
	poetry run python asm-cli.py mono-setup
	poetry run python asm-cli.py generate-param-triples \
		--num-midi 1 \
		--pairs 1
	poetry run python asm-cli.py process-audio
	for fxt in ae cnn flowreg mlp resnet vae wae vaeflow ; do \
		poetry run python asm-cli.py update-registry \
			src/config/fixtures/aiflowsynth/${fxt}.py; \
		make model; \
		make evaluate; \
	done

figures:
	poetry run python src/graphics/cli.py train-val-loss --latest
	poetry run python src/graphics/cli.py spectral-loss-distplot --latest
	poetry run python src/graphics/cli.py tsne-latent-space
	poetry run python src/graphics/cli.py inference-comparison
