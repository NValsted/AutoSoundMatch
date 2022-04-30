PYTHON_INTERPRETER_PATH ?= poetry run python

MIDI_DIR ?= data/midi
AUDIO_DIR ?= data/audio
MODEL_DIR ?= data/model
GENETIC_DIR ?= data/genetic
DOWNLOADS_DIR ?= data/downloads

PRESETS_DIR ?= data/presets
SYNTH_PATH ?= data/synth/TAL-NoiseMaker.vst3

# Benchmarks
SIGNAL_PROCESSING ?= default
TARGET_SYNTH ?= noisemaker
PARAM_LIMIT ?= 32

DOCKER_IMAGE ?= nvalsted/autosoundmatch:latest

.PHONY: nvidia-container-toolkit build-image run-image-interactive paths resources clear-resources tables midi-partitions dataset prepare-data genetic model evaluate reset inspect model-suite synth-dsp-fixtures mono-benchmark-setup poly-benchmark-setup mono-benchmark poly-benchmark

nvidia-container-toolkit:  # Should work for Ubuntu and Debian - Otherwise, see: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
	distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
	&& curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
	&& curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
		sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
		sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
	sudo apt-get install -y nvidia-docker2
	sudo systemctl restart docker

build-image:
ifeq ($(OS),Windows_NT)
	docker build . -t ${DOCKER_IMAGE}
else
	sudo docker build . -t ${DOCKER_IMAGE}
endif

run-image-interactive:
	docker run --rm -it ${DOCKER_IMAGE}

paths:
	${PYTHON_INTERPRETER_PATH} asm-cli.py setup-paths \
		--midi ${MIDI_DIR} \
		--audio ${AUDIO_DIR} \
		--model ${MODEL_DIR} \
		--downloads ${DOWNLOADS_DIR} \
		--presets ${PRESETS_DIR} \
		--genetic ${GENETIC_DIR}

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

ifeq ($(OS),Windows_NT)
	@if [ ! -f "${SYNTH_PATH}" ]; \
	then \
		wget https://tal-software.com//downloads/plugins/install_tal-noisemaker.zip -O ${DOWNLOADS_DIR}/install_tal-noisemaker.zip && \
		unzip ${DOWNLOADS_DIR}/install_tal-noisemaker.zip "TAL-NoiseMaker.vst3/**/*" -d ${DOWNLOADS_DIR} && \
		mkdir -p ${SYNTH_PATH} && \
		mv ${DOWNLOADS_DIR}/TAL-NoiseMaker.vst3/Contents/x86_64-win/TAL-NoiseMaker.vst3 ${SYNTH_PATH} && \
		rm ${DOWNLOADS_DIR}/install_tal-noisemaker.zip && \
		rm -r ${DOWNLOADS_DIR}/TAL-NoiseMaker.vst3; \
	else \
		echo "Synth already installed."; \
	fi
else ifeq ($(OS),Darwin)
	@echo "Not implemented - synth can be manually downloaded from https://tal-software.com//downloads/plugins/tal-noisemaker-installer.pkg"
else
	@if [ ! -f "${SYNTH_PATH}" ];
	then \
		wget https://tal-software.com/downloads/plugins/TAL-NoiseMaker_64_linux.zip -O ${DOWNLOADS_DIR}/TAL-NoiseMaker_64_linux.zip && \
		unzip ${DOWNLOADS_DIR}/TAL-NoiseMaker_64_linux.zip "libTAL-NoiseMaker.so" -d ${DOWNLOADS_DIR} && \
		mkdir -p ${SYNTH_PATH} && \
		mv ${DOWNLOADS_DIR}/libTAL-NoiseMaker.so ${SYNTH_PATH} && \
		rm ${DOWNLOADS_DIR}/TAL-NoiseMaker_64_linux.zip; \
	else \
		echo "Synth already installed."; \
	fi
endif

	@if ls ${PRESETS_DIR}/*TAL.vstpreset > /dev/null 2>&1; \
	then \
		echo "Presets already installed."; \
	else \
		wget https://tal-software.com//downloads/presets/TAL-NoiseMaker%20vst3.zip -O ${DOWNLOADS_DIR}/TAL-NoiseMaker%20vst3.zip && \
		unzip ${DOWNLOADS_DIR}/TAL-NoiseMaker%20vst3.zip "*.vstpreset" -d ${PRESETS_DIR} &&  \
		rm ${DOWNLOADS_DIR}/TAL-NoiseMaker%20vst3.zip; \
	fi

clear-resources:
	rm -r ${DOWNLOADS_DIR}/msmd_all
	rm -r ${DOWNLOADS_DIR}/msmd_real_performances
	rm -r ${DOWNLOADS_DIR}/nottingham
	rm -r ${DOWNLOADS_DIR}/lmd_matched
	rm ${SYNTH_PATH}
	rm ${PRESETS_DIR}/*TAL.vstpreset ${PRESETS_DIR}/*FN.vstpreset ${PRESETS_DIR}/*AS.vstpreset ${PRESETS_DIR}/*TUC.vstpreset ${PRESETS_DIR}/*FM.vstpreset

tables:
	${PYTHON_INTERPRETER_PATH} asm-cli.py setup-relational-models \
		--engine-url "sqlite:///data/local.db" \
		--synth-path ${SYNTH_PATH}

midi-partitions:
	${PYTHON_INTERPRETER_PATH} asm-cli.py partition-midi-files \
		--directory ${DOWNLOADS_DIR}/msmd_real_performances/msmd_all_deadpan/performance/ \
		--directory ${DOWNLOADS_DIR}/lmd_matched/A/A/

dataset:
	${PYTHON_INTERPRETER_PATH} asm-cli.py generate-param-triples
	${PYTHON_INTERPRETER_PATH} asm-cli.py process-audio

prepare-data:
	make tables
	make midi-partitions
	make dataset

genetic:
	${PYTHON_INTERPRETER_PATH} asm-cli.py update-registry \
		src/config/fixtures/genetic.py
	${PYTHON_INTERPRETER_PATH} asm-cli.py test-genetic-algorithm \
		--test-limit 128

model:
	${PYTHON_INTERPRETER_PATH} asm-cli.py train-model

evaluate:
	${PYTHON_INTERPRETER_PATH} asm-cli.py test-model

reset:
	@echo "Resetting project state"
	${PYTHON_INTERPRETER_PATH} asm-cli.py reset

inspect:
	@echo "Inspecting project state"
	@${PYTHON_INTERPRETER_PATH} -c "import warnings; warnings.filterwarnings('ignore'); from torch.cuda import is_available; print('USING GPU' if is_available() else 'USING CPU')"
	${PYTHON_INTERPRETER_PATH} asm-cli.py inspect-registry

model-suite:
	for fxt in ae cnn flowreg mlp resnet vae wae vaeflow ; do \
		${PYTHON_INTERPRETER_PATH} asm-cli.py update-registry \
			src/config/fixtures/aiflowsynth/$${fxt}.py; \
		make model; \
		make evaluate; \
	done

synth-dsp-fixtures:
ifeq (${SIGNAL_PROCESSING},acids-ircam)
	${PYTHON_INTERPRETER_PATH} asm-cli.py update-registry \
		src/config/fixtures/aiflowsynth/signal_processing.py
else
	${PYTHON_INTERPRETER_PATH} asm-cli.py update-registry \
		src/config/fixtures/default_signal_processing.py
endif

ifeq (${TARGET_SYNTH},diva)
	${PYTHON_INTERPRETER_PATH} asm-cli.py update-registry \
		src/config/fixtures/aiflowsynth/u-he_diva${PARAM_LIMIT}.py
	${PYTHON_INTERPRETER_PATH} asm-cli.py setup-diva-presets

else ifeq (${TARGET_SYNTH},mikamicro)
	${PYTHON_INTERPRETER_PATH} asm-cli.py update-registry \
		src/config/fixtures/mikamicro${PARAM_LIMIT}.py

else ifeq (${TARGET_SYNTH},noisemaker)
	${PYTHON_INTERPRETER_PATH} asm-cli.py update-registry \
		src/config/fixtures/noisemaker${PARAM_LIMIT}.py

else
	${PYTHON_INTERPRETER_PATH} asm-cli.py update-registry \
		src/config/fixtures/synth.py
endif

mono-benchmark-setup:
	make reset
	make paths
	make resources
	make synth-dsp-fixtures

	${PYTHON_INTERPRETER_PATH} asm-cli.py setup-relational-models \
		--engine-url "sqlite:///data/local.db"
	${PYTHON_INTERPRETER_PATH} asm-cli.py mono-setup

ifeq (${TARGET_SYNTH},diva)
	${PYTHON_INTERPRETER_PATH} asm-cli.py generate-param-triples \
		--num-presets 11000 \
		--num-midi 1 \
		--pairs 1 \
		--preset-glob "*.json"
else ifeq (${TARGET_SYNTH},noisemaker)
	${PYTHON_INTERPRETER_PATH} asm-cli.py generate-param-triples \
		--num-presets 11000 \
		--num-midi 1 \
		--pairs 1 \
		--preset-glob "*.vstpreset"
else
	${PYTHON_INTERPRETER_PATH} asm-cli.py generate-param-triples \
		--num-presets 11000 \
		--num-midi 1 \
		--pairs 1
endif

	${PYTHON_INTERPRETER_PATH} asm-cli.py process-audio

poly-benchmark-setup:
	make reset
	make paths
	make resources
	make synth-dsp-fixtures

	${PYTHON_INTERPRETER_PATH} asm-cli.py setup-relational-models \
		--engine-url "sqlite:///data/local.db"
	make midi-partitions

ifeq (${TARGET_SYNTH},diva)
	${PYTHON_INTERPRETER_PATH} asm-cli.py generate-param-triples \
		--num-presets 3000 \
		--num-midi 500 \
		--pairs 4 \
		--preset-glob "*.json"
else ifeq (${TARGET_SYNTH},noisemaker)
	${PYTHON_INTERPRETER_PATH} asm-cli.py generate-param-triples \
		--num-presets 3000 \
		--num-midi 500 \
		--pairs 4 \
		--preset-glob "*.vstpreset"
else
	${PYTHON_INTERPRETER_PATH} asm-cli.py generate-param-triples \
		--num-presets 3000 \
		--num-midi 500 \
		--pairs 4
endif

	${PYTHON_INTERPRETER_PATH} asm-cli.py process-audio

mono-benchmark:
	make mono-benchmark-setup
	make model-suite
	make genetic

poly-benchmark:
	make poly-benchmark-setup
	make model-suite
	make genetic
