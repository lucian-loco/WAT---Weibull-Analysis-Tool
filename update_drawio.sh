#!/bin/bash
echo "Updating draw.io templates..."
set -e

if [ -z "$EOS_USER" ] || [ -z "$EOS_PASS" ]; then
    echo "WARNING: Either EOS_USER or EOS_PASS is empty."
fi


DRAWIO_LIBRARY_URL="https://api.cernbox.cern.ch/remote.php/dav/files/$EOS_USER/eos/project/h/hit/draw.io/Installation"
OUTPUT_DIR="src/static/drawio"
FILES=(
    "Chassis+Rack.xml"
    "Modules 3U.xml"
    "Modules FMC.xml"
    "Modules LXI.xml"
    "Modules PCI.xml"
    "Modules PMC.xml"
    "Modules VME.xml"
    "Modules VMOD.xml"
    "P2.xml"
    "Patch.xml"
    "WFIP.xml"
)


mkdir -p "$OUTPUT_DIR"

for file in "${FILES[@]}"; do
    echo "Downloading $file..."
    wget -q -x --user "$EOS_USER" --password "$EOS_PASS" "$DRAWIO_LIBRARY_URL/$file" -O "$OUTPUT_DIR/$file"
done

echo Done!
