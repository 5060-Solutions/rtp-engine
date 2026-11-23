#!/usr/bin/env bash
# Download the ITU-T G.729 software package and unpack the Annex A and
# Annex B test vectors into target/g729-vectors, where the codec's bit-exactness tests
# look for them (override with G729_TEST_VECTORS).
#
# The vectors are ITU-T material and are not committed to this repository.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dest="${G729_TEST_VECTORS:-$root/target/g729-vectors}"
url='https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-G.729-201206-I!!SOFT-ZST-E&type=items'

if [[ -f "$dest/SPEECH.BIT" && -f "$dest/tstseq1a.bit" ]]; then
    echo "G.729 test vectors already in $dest"
    exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

curl -fsSL --retry 3 -o "$tmp/g729.zip" "$url"
unzip -q "$tmp/g729.zip" \
    'Software/G729_Release3/g729AnnexA/test_vectors/*' \
    'Software/G729_Release3/g729AnnexB/test_vectors/*' -d "$tmp"

mkdir -p "$dest"
cp "$tmp"/Software/G729_Release3/g729AnnexA/test_vectors/* "$dest/"
cp "$tmp"/Software/G729_Release3/g729AnnexB/test_vectors/* "$dest/"
echo "G.729 Annex A and B test vectors in $dest"
