#!/bin/bash
# Compile paper_kv_faithfulness.tex to PDF (run twice for cross-references)
set -e
cd "$(dirname "$0")"

############ revised paper
echo rm paper_kv_revised.pdf
rm paper_kv_revised.pdf
mkdir -p save
mv paper_kv_revised.aux paper_kv_revised.log paper_kv_revised.out save

echo xelatex -interaction=nonstopmode paper_kv_revised.tex
xelatex -interaction=nonstopmode paper_kv_revised.tex

echo xelatex -interaction=nonstopmode paper_kv_revised.tex
xelatex -interaction=nonstopmode paper_kv_revised.tex

echo "====== Done: paper_kv_revised.pdf ====== "

############ original paper
echo rm paper_kv_faithfulness.pdf
rm paper_kv_faithfulness.pdf
mkdir -p save
mv paper_kv_faithfulness.aux paper_kv_faithfulness.log paper_kv_faithfulness.out save

echo xelatex -interaction=nonstopmode paper_kv_faithfulness.tex
xelatex -interaction=nonstopmode paper_kv_faithfulness.tex

echo xelatex -interaction=nonstopmode paper_kv_faithfulness.tex
xelatex -interaction=nonstopmode paper_kv_faithfulness.tex

echo "====== Done: paper_kv_faithfulness.pdf ====== "

############ short paper
echo rm paper_kv_short.pdf
rm paper_kv_short.pdf
mkdir -p save
mv paper_kv_short.aux paper_kv_short.log paper_kv_short.out save

echo xelatex -interaction=nonstopmode paper_kv_short.tex
xelatex -interaction=nonstopmode paper_kv_short.tex

echo xelatex -interaction=nonstopmode paper_kv_short.tex
xelatex -interaction=nonstopmode paper_kv_short.tex

echo "====== Done: paper_kv_short.pdf ====== "
