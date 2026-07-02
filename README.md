# ER-CTTS

## [Speech Communication'2026] Emphasis Rendering for Conversational Text-to-Speech with Multi-modal Multi-scale Context Modeling

## Introduction

This is an implementation of the following paper: ["Emphasis Rendering for Conversational Text-to-Speech with Multi-modal Multi-scale Context Modeling"](https://www.sciencedirect.com/science/article/pii/S0167639326000014).

**Rui Liu**, **Zhenqi Jia**, **Jie Yang**, **Yifan Hu**, **Haizhou Li**

Published in **Speech Communication, 2026**.

## Dataset

The experiments are conducted on the DailyTalk dataset with emphasis annotations.

The emphasis labels are provided in:

```bash id="tdm7h3"
emphasis/label.json
```

Each entry in `emphasis/label.json` contains the emphasis-related annotations used for emphasis rendering in conversational text-to-speech.

## Preprocessing

Run

```bash id="wlv3ts"
python3 prepare_align.py --dataset DailyTalk
```

for basic preparation.

For forced alignment, [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) can be used to obtain the alignments between utterances and phoneme sequences.

After preparing the alignments, run the preprocessing script by

```bash id="k9y7db"
python3 preprocess.py --dataset DailyTalk
```

Please make sure that the emphasis label file is placed at:

```bash id="doz4m8"
emphasis/label.json
```

before training the model.

## Training

Train ER-CTTS with

```bash id="1b9ue8"
python3 train.py --dataset DailyTalk
```

## Inference

Only batch inference is supported, as the generation of a target utterance may require contextual history of the conversation.

Try

```bash id="84a3jr"
python3 synthesize.py --source preprocessed_data/DailyTalk/test_*.txt --restore_step RESTORE_STEP --mode batch --dataset DailyTalk
```

to synthesize all utterances in `preprocessed_data/DailyTalk/test_*.txt`.

## Citation

If you would like to use our dataset and code or refer to our paper, please cite as follows.

```bibtex id="j6ogve"
@article{liu2026emphasis,
  title={Emphasis rendering for conversational text-to-speech with multi-modal multi-scale context modeling},
  author={Liu, Rui and Zhenqi, Jia and Yang, Jie and Hu, Yifan and Li, Haizhou},
  journal={Speech Communication},
  pages={103353},
  year={2026},
  publisher={Elsevier}
}
```

## Contact

For any questions, please contact the authors.
