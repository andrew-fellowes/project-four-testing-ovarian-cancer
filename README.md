# Evaluating the Accuracy of a Predicitive Cancer Test

## Overview

The purpose of this project is to create a tool that can help pathologists predict which biopsy samples tested by a neural network-based diagnostic test are likely to have received a false positive or a false negative result, thereby increasing the diagnostic efficacy of the test.

The Peter MacCallum Cancer Centre uses a test from SOPHIA Genetics to test biopsy samples to determine whether patients with ovarian cancer will respond to a particular drug therapy. The test looks for ‘genomic instability’, which is where there are multiple breaks in the DNA sequence. If this feature (or 'phenotype') is seen in ovarian cancer genomes, it is a positive predictive biomarker for treatment with new drugs (i.e. such patients are likely to do better than if treated with standard drugs).

BRCA1 and BRCA2 are genes that encode proteins that help repair damaged DNA. Everyone has two copies of each of these genes—one copy inherited from each parent. BRCA1 and BRCA2 are called tumor suppressor genes because when they stop functioning, usually due to harmful (or pathogenic) variants (or mutations), tumours may develop.

Some people have a genetic predisposition to ovarian (and breast) cancer because they have already inherited a pathogenic BRCA1 or BRCA2 variant from one parent. Although they would have also inherited a normal copy of that gene from the other parent (because embryos inheriting two harmful variants cannot develop), the normal copy can be lost or changed in some cells in the body during the person’s lifetime. Cells that lose their remaining functioning BRCA1 or BRCA2 genes can grow out of control and become cancer.

In ovarian cancer, the cancerous cells will often have multiple breaks in their DNA (genetic instability) because BRCA1 or BRCA2 is no longer functional. Patients with this 'phenotype' are likely to respond well to a particular drug therapy that inhibits backup DNA repair processes, resulting in cancer cell death. This drug is prescribed once a patient is in remission and helps control future tumor growth. The drug has the advantage of having fewer side-effects than other cancer treatments available.

The ’gold standard’ test for genetic instability is the Myriad Genetics myChoice test; however, this is not available in Australia. An alternative, the SOPHIA Genetics test, uses a machine learning model trained to detect genomic instability within cancer genome. This model has high accuracy compared to Myriad (approximately 95%) however, false negatives and false positives remain an issue and have serious consequences for the patients – whether being prescribed a drug that is ineffective (false positive) or not receiving a drug when it is suitable (false negative). The idea is

We have been supplied training data from the Peter MacCallum Cancer Centre. This is a series of test results from 135 cases, with corresponding Myriad and SOPHIA results. The data includes many independent variables for training a model.

Two approaches have been taken: development of a neural network using TensorFlow, creating a binary classification model that can predict which SOPHIA genomic instability results may be misclassifications; alternatively, given the limited number of results and the high risk of a neural network becoming overfitted, a Random Forest model to achieve the same prediction.


## Features

| Heading | Description |
| ---------- | ---------- |
| Run | Nominal value describing a batch of samples analysed together. Maximum number of samples in a run is 24. Each run has an associated SeqRunID. |
| SampleID | Unique laboratory identifier for a sample (i.e a tumour or 'specimen'). |
| Source | Original provider of the specimen. AZ = AstraZeneca, WEHI = Walter & Eliza Hall Institute, Greece = Genotypos, Athens, Brazil = DASA, Sao Paulo. |
| MonthsOld | Age of specimen in months from collection date to receipt by Peter Mac (if known). |
| Purity | Microscopy-based estimate of percentage of tumour cells in the specimen. Indicates the amount of contaminating normal cells in the specimen. |
| SeqRunID | Analyser generated identifier for each batch of analyses. |
| DDMSampleID | Bioinformatic pipeline generated unique identifier for each sample. One to one correspondence with SampleID. |
| MIDS | Bioinformatic pipeline generated multiplex identifier. Unique within each run. Same as suffix of DDMSampleID. Useful to define sort order of samples within each batch. |
| TotalReads(M) | Total number of sequence reads associated with the sample.. Sum of lpWGS reads and Target Panel reads |
| lpWGSReads(M) | Sequence reads scattered randomly across the genome at low fold coverage - 1x or less ('low pass'), meaning the average number of reads covering any base pair is at most one. |
| TargetPanelReads(M) | Sequence reads over genes of interest (BRCA1 and BRCA2). High fold coverage (100x minimum fold coverage per base pair) over these regions of the genome is achieved by molecular enrichment technique called hybridisation capture |
| %ReadslpWGS | Percentage of Total reads that are lpWGS reads. Usually about 65% |
| %ReadsPanel | Percentage of Total reads that are targeted to genes. Usually about 35% |
| 1000x | Percentage of base pair positions in Target Panel regions (not lpWGS regions) that get >1000x unique coverage |
| 500x | Percentage of base pair positions in Target Panel regions (not lpWGS regions) that get >500x unique coverage  |
| 200x | Percentage of base pair positions in Target Panel regions (not lpWGS regions) that get >200x unique coverage  |
| 100x | Percentage of base pair positions in Target Panel regions (not lpWGS regions) that get >100x unique coverage  |
| 50x | Percentage of base pair positions in Target Panel regions (not lpWGS regions) that get >50x unique coverage  |
| 25x | Percentage of base pair positions in Target Panel regions (not lpWGS regions) that get >25x unique coverage  |
| DupFrac | Fraction of all reads that are duplicates of each other (based on their mapped position on reference genome). Duplicates arise from PCR amplification of the sample. Duplicate reads are bioinformatically removed before coverage is calculated i.e. coverage is calculated for 'unique' reads |
| LowCovRegions | The number of regions with the Target Panel where coverage goes below 100x unique reads. I A high value indicates a poor sample or analysis |
| PurityPloidyRatio | Ratio between tumour content and local number of chromosomal regions (ploidy), estimated from the strength of the copy number signal seen in the assay. |
| ResNoise | Standard deviation of the normalized lpWGS coverage profile with respect to the smoothed lpWGS coverage profile |
| SignalNoiseRatio | Strength of the signal induced in the normalized lpWGS coverage profile by all copy number aberrations present in the sample divided by the residual noise |
| QAStatus | High : quality is sufficient to compute GI index. Medium: quality is lower and ML algorithm may not succeed in computing GI index. Low: quality too low to compute a GI index. |
| Gene | Name of gene in which any mutation was detected by the Targeted Panel (either BRCA1 or BRCA2) |
| Variant | Standardised nomenclature describing the mutation and its effect on the affected gene/protein |
| %VariantFraction | Fraction of reads supporting the variant as a percentage of all reads covering the mutated position |
| MyriadGIScore | The Genomic Instability score determined for the sample using the Myriad Genetics myChoice HRD assay, an FDA approved measure of HRD as a biomarker. This is the Reference Method ('Gold Standard', or source of truth in determining the accuracy of the SOPHiA method). Range from 1 to 100, a value greater than 42 corresponds to genomically unstable (HRD positive) |
| MyriadGIStatus | 1 = HRD Positive 2 = HRD Negative |
| SOPHiAGIIndex | The Genomic Instability Index for the method being validated - the SOPHiA Genetics HRD assay. Range form -20 to 20, a value greater than 0 corresponds to genomically unstable |
| SophiaGIStatus | 1 = HRD Positive, 2 = HRD Negative, 3 = Inconclusive, 4 = Rejected |

## External Documents

[Link](https://www.biorxiv.org/content/biorxiv/early/2022/07/08/2022.07.06.498851.full.pdf) to preprint describing training of the SOPHiA Genetics CNN algorithm.

[Genomic Integrity Report](HRD_202305051913-21752-0072-GI-Report.pdf) - contains useful information.

[LICENSE](LICENSE)