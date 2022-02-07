midi_dir := data/midi
audio_dir := data/audio

fetch-resources:
	@echo "NOT YET IMPLEMENTED"

build-vst:
	@echo "Building OpnTaybel VST"
	@echo "NOT YET IMPLEMENTED"

prepare-data:
	mkdir -p ${midi_dir} ${audio_dir}
	@echo "Setting up local database and generating data"
	pyflow asm-cli.py setup-relational-models \
		--synth-path "./data/vst/Serum_x64.dll" \
		--engine-url "sqlite:///data/local.db"
	pyflow asm-cli.py generate-param-triples \
		--midi-path ${midi_dir} \
		--audio-path ${audio_dir}

reset:
	@echo "Resetting project state"
	pyflow asm-cli.py reset

inspect:
	@echo "Inspecting project state"
	pyflow asm-cli.py inspect-registry
