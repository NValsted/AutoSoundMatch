midi_dir := data/midi
audio_dir := data/audio
docker_image := nvalsted/autosoundmatch:latest

build-image:
ifeq ($(OS),Windows_NT)
	docker build --no-cache . -t ${docker_image}
else
	sudo docker build --no-cache . -t ${docker_image}
endif

run-image-interactive:
	docker run --rm -it ${docker_image}

fetch-resources:
	@echo "NOT YET IMPLEMENTED"

build-vst:
	@echo "Building OpnTaybel VST"
	@echo "NOT YET IMPLEMENTED"

prepare-data:
	mkdir -p ${midi_dir} ${audio_dir}
	@echo "Setting up local database and generating data"
	poetry run python asm-cli.py setup-relational-models \
		--synth-path "./data/vst/Serum_x64.dll" \
		--engine-url "sqlite:///data/local.db"
	poetry run python asm-cli.py generate-param-tuples \
		--midi-path ${midi_dir} \
		--audio-path ${audio_dir}

reset:
	@echo "Resetting project state"
	poetry run python asm-cli.py reset

inspect:
	@echo "Inspecting project state"
	poetry run python asm-cli.py inspect-registry
