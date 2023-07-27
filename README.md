# project-four-testing-ovarian-cancer

Clean data to start the project is ./Resources/PeterMac_HRD_Validation.csv

Column Descriptions:

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
