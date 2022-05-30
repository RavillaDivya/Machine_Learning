# TerRi
TERrestrialRIder

"Hey Terri, what's trending?"

Use TerRi, the Terrestrial Rider companion app, to find out information around you as you are travelling.

## Steps

For audio to text on Mac

1. brew install portaudio
2. pip install pyaudio
If this doesn't work try

pip install --global-option='build_ext' --global-option="-I$(brew --prefix)/include" --global-option="-L$(brew --prefix)/lib" pyaudio

3. brew install flac
