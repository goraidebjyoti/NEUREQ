List of the folder:

1) 2021 Clinical Trials Track 75 patient case descriptions
2) 2022 Clinical Trials Track 50 patient case descriptions
3) 20,000 synthetic gold queries/patient case descriptions
4) One relevant trial for each synthetic gold query/patient case descriptions
5) Top 100 trials using BM25 for each 2021 query [1,75] and synthetic query [75,:]
6) Sanitised LLM responses for synthetic triplet (query, pos_trial, neg_trial)
7) Encoded responses .csv file from LLM sanitised responses for the synthetic triplet pair for training the NEUREQ model
