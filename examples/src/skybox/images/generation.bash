# Needs montage from ImageMagick in PATH
# Needs kram from https://github.com/alecazam/kram in PATH

# Generate a skybox image from 6 jpeg in the folder in first argument.
# The images must be named right.jpg, left.jpg, top.jpg, bottom.jpg, back.jpg, front.jpg
#
# bash examples/src/skybox/images/generation.bash ./path/to/images/folder

SCRIPT_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
CHUNK_SIZE="256x256"

set -e

# ensure an argument is passed
if [ $# -eq 0 ]; then
    echo "No arguments supplied!"
    echo
    echo "Usage: bash examples/src/skybox/images/generation.bash ./path/to/images/folder"
    exit 1
fi

montage $1/right.jpg $1/left.jpg $1/top.jpg $1/bottom.jpg $1/back.jpg $1/front.jpg -tile 6x1 -geometry $CHUNK_SIZE+0+0 $1/combined-output.png
kram encode -type cube -format rgba8 -chunks $CHUNK_SIZE -i $1/combined-output.png -o $SCRIPT_DIRECTORY/rgba8.ktx2
kram encode -type cube -format bc7 -chunks $CHUNK_SIZE -i $1/combined-output.png -o $SCRIPT_DIRECTORY/bc7.ktx2
kram encode -type cube -format astc4x4 -chunks $CHUNK_SIZE -i $1/combined-output.png -o $SCRIPT_DIRECTORY/astc.ktx2
rm $1/combined-output.png
