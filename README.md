# Upon Request

This repository contains the code and lightweight data tables used to analyse the use of request based data availability statements such as “data available upon request” in the genomics, genetics, and bioinformatics literature.

The analyses focus on how often such wording is used, how it has changed over time, and how it relates to other signals of open science support at the article and journal level.

## Overview

Using full text JATS XML articles from PubMed Central Open Access, we detect and classify “upon request” statements and distinguish between vague formulations and cases linked to explicit access mechanisms or legitimate restrictions. In parallel, we extract multiple indicators of open science practices, including data deposition, code availability, protocol sharing, and source data provision.

The repository accompanies a policy and meta research analysis and is intended to support transparency, reproducibility, and independent auditing.

## Repository content

- Scripts for metadata retrieval, XML parsing, and text mining  
- Rule based classification of request based availability statements  
- Extraction and scoring of open science support indicators  
- Small derived tables and reference files stored in `2.data/`  

Large datasets and full text XML corpora are not hosted on GitHub.

## Data availability

Full text XML files and large derived datasets used for the analyses are archived separately on Zenodo.  
Links to the corresponding Zenodo records will be provided.

## Status

This repository reflects the analysis pipeline used for the associated manuscript.  
Minor updates and documentation improvements may occur, but the overall structure is stable.

## License

This project is released under the GNU General Public License v3.0.  
See the LICENSE file for details.

## Citation

If you use this code or derived analyses in academic work, please cite the associated manuscript.  
A `CITATION.cff` file will be added.
