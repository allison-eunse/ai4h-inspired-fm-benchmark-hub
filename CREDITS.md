# Credits & Acknowledgments

This benchmark hub uses data and tools from multiple sources. We gratefully acknowledge the following contributions:

---

## Data Sources

### Genomic Benchmarks (DNA Sequence Classification)

The DNA sequence benchmark datasets (enhancers, promoters, regulatory elements) are from the **Genomic Benchmarks** collection.

> GREŠOVÁ, Katarína, et al. **Genomic Benchmarks: A Collection of Datasets for Genomic Sequence Classification.** *bioRxiv*, 2022.

- **Paper**: https://www.biorxiv.org/content/10.1101/2022.06.08.495248
- **HuggingFace**: https://huggingface.co/datasets/katielink/genomic-benchmarks
- **License**: Apache 2.0

```bibtex
@article{gresova2022genomic,
  title={Genomic Benchmarks: A Collection of Datasets for Genomic Sequence Classification},
  author={Gresova, Katarina and Martinek, Vlastimil and Cechak, David and Simecek, Petr and Alexiou, Panagiotis},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory},
  url={https://www.biorxiv.org/content/10.1101/2022.06.08.495248}
}
```

---

### Nucleotide Transformer Benchmark

Multi-task DNA benchmark from InstaDeepAI.

> DALLA-TORRE, Hugo, et al. **The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics.** *bioRxiv*, 2023.

- **Paper**: https://www.biorxiv.org/content/10.1101/2023.01.11.523679
- **HuggingFace**: https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised

```bibtex
@article{dalla2023nucleotide,
  title={The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics},
  author={Dalla-Torre, Hugo and Gonzalez, Liam and Mendoza-Revilla, Javier and others},
  journal={bioRxiv},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

---

### 10x Genomics PBMC 3K Dataset

Single-cell RNA-seq data from peripheral blood mononuclear cells.

> **10x Genomics.** 3k PBMCs from a Healthy Donor. 10x Genomics Datasets, 2016.

- **URL**: https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0
- **License**: Free for research use

---

### ENCODE Consortium

Regulatory element annotations derived from ENCODE experimental data.

> **The ENCODE Project Consortium.** An integrated encyclopedia of DNA elements in the human genome. *Nature*, 489(7414), 57-74, 2012.

- **URL**: https://www.encodeproject.org/

```bibtex
@article{encode2012integrated,
  title={An integrated encyclopedia of DNA elements in the human genome},
  author={{ENCODE Project Consortium}},
  journal={Nature},
  volume={489},
  number={7414},
  pages={57--74},
  year={2012},
  publisher={Nature Publishing Group}
}
```

---

### Roadmap Epigenomics

Epigenomic data across human tissues and cell types.

> **Roadmap Epigenomics Consortium.** Integrative analysis of 111 reference human epigenomes. *Nature*, 518(7539), 317-330, 2015.

- **URL**: http://www.roadmapepigenomics.org/

---

### Eukaryotic Promoter Database (EPD)

Curated promoter sequences.

> DREOS, René, et al. **The eukaryotic promoter database in its 30th year: focus on non-vertebrate organisms.** *Nucleic Acids Research*, 45(D1), D51-D55, 2017.

- **URL**: https://epd.expasy.org/epd/

---

## Tools & Software

### Scanpy

Used for single-cell RNA-seq data processing.

> WOLF, F. Alexander, ANGERER, Philipp, and THEIS, Fabian J. **SCANPY: large-scale single-cell gene expression data analysis.** *Genome Biology*, 19(1), 1-5, 2018.

```bibtex
@article{wolf2018scanpy,
  title={SCANPY: large-scale single-cell gene expression data analysis},
  author={Wolf, F Alexander and Angerer, Philipp and Theis, Fabian J},
  journal={Genome biology},
  volume={19},
  number={1},
  pages={1--5},
  year={2018},
  publisher={Springer}
}
```

---

## Framework Alignment

### ITU/WHO Focus Group on AI for Health (FG-AI4H)

This benchmark hub's methodology is derived from the public deliverables of FG-AI4H.

- **DEL3**: System Requirement Specifications
- **DEL5.4**: Test Data Specification  
- **DEL7.x**: Test Suite Guidelines
- **DEL10.8**: Analytical Validation Protocols

- **URL**: https://www.itu.int/en/ITU-T/focusgroups/ai4h/

---

## License

This benchmark hub is licensed under the MIT License. Individual datasets retain their original licenses as noted above.
