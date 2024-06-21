#!/bin/sh

if [ -z "$1" ]; then
echo "supply jupiter notebook name without extension"
exit 1
fi

jupyter nbconvert --to script "$1"
