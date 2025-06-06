
This repository contains the code for constructing the dataset described in our paper "Inter-Speaker Relative Cues for Text-Guided Target Speech Extraction", accepted at Interspeech 2025.

Description:
-
Due to licensing restrictions associated with several corpora used in our work, we are unable to release the corresponding audio samples. However, you may obtain the original corpora by requesting access or signing a license agreement with the respective owners.
We provide metadata JSON files containing summarized statistics and essential information used to construct our dataset.

Specifically, they are categorized into three groups: 'emotion', 'age', and 'without_emotion_age'. 
Each group contains six JSON files: train_part1.json, train_part2.json, val_part1.json, val_part2.json, test_part1.json, and test_part2.json.

generate_rir.py and prompt_template.py are used for simulating room impulse responses (RIRs) and generating template prompts, respectively.

You can use run.sh script to execute multi-process parallelization to speed up dataset simulation.

Acknowledgement:
-
Template prompts built refers [Listen-Chat-Edit-on-Edge](https://github.com/SiavashShams/Listen-Chat-Edit-on-Edge/blob/main/data/datasets/prompt_templates.py)
