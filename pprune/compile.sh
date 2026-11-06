#!/bin/bash
# Compile paper_kv_faithfulness.tex to PDF (run twice for cross-references)
set -e
cd "$(dirname "$0")"

############ wtv paper
echo "====== Compiling: paper_wtv.pdf ====== "
echo rm -f paper_wtv.pdf
rm -f paper_wtv.pdf
mkdir -p save
# mv paper_wtv.aux paper_wtv.log paper_wtv.out save

echo xelatex -interaction=nonstopmode paper_wtv.tex
xelatex -interaction=nonstopmode paper_wtv.tex

echo xelatex -interaction=nonstopmode paper_wtv.tex
xelatex -interaction=nonstopmode paper_wtv.tex

echo "====== Done: paper_wtv.pdf ====== "

############ revised paper
echo "====== Compiling: paper_kv_revised.pdf ====== "
echo rm -f paper_kv_revised.pdf
rm -f paper_kv_revised.pdf
mkdir -p save
mv paper_kv_revised.aux paper_kv_revised.log paper_kv_revised.out save

echo xelatex -interaction=nonstopmode paper_kv_revised.tex
xelatex -interaction=nonstopmode paper_kv_revised.tex

echo xelatex -interaction=nonstopmode paper_kv_revised.tex
xelatex -interaction=nonstopmode paper_kv_revised.tex

echo "====== Done: paper_kv_revised.pdf ====== "

############ original paper
echo "====== Compiling: paper_kv_faithfulness.pdf ====== "
echo rm -f paper_kv_faithfulness.pdf
rm -f paper_kv_faithfulness.pdf
mkdir -p save
mv paper_kv_faithfulness.aux paper_kv_faithfulness.log paper_kv_faithfulness.out save

echo xelatex -interaction=nonstopmode paper_kv_faithfulness.tex
xelatex -interaction=nonstopmode paper_kv_faithfulness.tex

echo xelatex -interaction=nonstopmode paper_kv_faithfulness.tex
xelatex -interaction=nonstopmode paper_kv_faithfulness.tex

echo "====== Done: paper_kv_faithfulness.pdf ====== "

############ short paper
echo "====== Compiling: paper_kv_short.pdf ====== "
echo rm -f paper_kv_short.pdf
rm -f paper_kv_short.pdf
mkdir -p save
mv paper_kv_short.aux paper_kv_short.log paper_kv_short.out save

echo xelatex -interaction=nonstopmode paper_kv_short.tex
xelatex -interaction=nonstopmode paper_kv_short.tex

echo xelatex -interaction=nonstopmode paper_kv_short.tex
xelatex -interaction=nonstopmode paper_kv_short.tex

echo "====== Done: paper_kv_short.pdf ====== "
