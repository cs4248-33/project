# Data Augmentation Strategies to Combat Rare Words in Machine Translation

By Felix Ong Wei Cong, Kum Wing Ho, Pow Zhi Xiang, Brandon Thio Zhi Kai, Lee Jia Wei

---

This repo contains the relevant code used to augment the IWSLT 2017 Chinese–English dataset, as well as finetuning and evaluating the OpusMT model used for the English-Chinese translation task. [Read the full report here.](report.pdf)

## Abstract

Neural Machine Translation (NMT) models have progressed significantly in recent years, especially since the advent of the transformer architecture. However, there is still room for improvement, especially when it comes to translation of sentences that contain rare or out-of-vocabulary (OOV) words. In this paper, we focus on the former issue of rare words, and identify data augmentation as a potential solution. We evaluate various augmentation strategies that preserve syntactic correctness of the generated data, and our findings on custom datasets show that such techniques achieve significant BLEU score gains on sentences containing rare words, whilst also improving accuracy on the original dataset.

### 1. Downloading the Dataset

```sh
pip install requirements.txt
```

Cloning the repo should have included all the relevant prerequisites, such as the training, validation and test splits of the [IWSLT 2017 Chinese–English dataset](https://huggingface.co/datasets/iwslt2017), as well as our list of OOC words and custom test data. These files can be found in the `data` folder. However, if you need to download the IWSLT dataset manually:

```sh
python data.py
```

### 2. Augmentation

Generating augmented data involves 3 main steps. The following example assumes the token substitution augmentation strategy (TOK).

1. Generate synthetic English sentences from the IWSLT training data:

```bash
cd augment
```

```bash
python augment.py \
    --input_json_path ../data/train.json \
    --ooc_words_txt_path ../data/ooc_words.txt \
    --output_json_path tok_aug.json \
    --strategy TOK \
    --n_generate 20000
```

Note: run `python augment.py -h` for more details about each of the command line arguments.

2. Generate translations of the English sentences by running inference on `tok_aug.json` with your baseline NMT model of choice. The relevant steps can be found in the latter section on [Evaluation](#3-evaluation).

3. Combine the plain text translation file that was generated by the model in step 2 into `tok_aug.json`, and concatenate it with the original IWSLT training data:

```bash
cd augment
```

```bash
python combine.py \
    --train_json_path ../data/train.json \
    --augment_json_path tok_aug.json \
    --translations_txt_path ../opus_baseline/generated_predictions.txt \
    --concatenated_train_json_path ../data/tok_train.json
```

Note: run `python combine.py -h` for more details about each of the command line arguments.

### 3. Evaluation

`ft.py` is adapted from the [huggingface repo](https://github.com/huggingface/transformers/tree/main/examples/pytorch/translation) for finetuning and evaluating transformer models on translation tasks.

The script supports only custom JSONLINES files, with each line being a dictionary with a key `"translation"` and its value another dictionary whose keys is the language pair. For example:

```json
{
  "translation": {
    "en": "Last year I showed these two slides so that  demonstrate that the arctic ice cap,  which for most of the last three million years  has been the size of the lower 48 states,  has shrunk by 40 percent.",

    "zh": "\u53bb\u5e74\u6211\u7ed9\u5404\u4f4d\u5c55\u793a\u4e86\u4e24\u4e2a \u5173\u4e8e\u5317\u6781\u51b0\u5e3d\u7684\u6f14\u793a \u5728\u8fc7\u53bb\u4e09\u767e\u4e07\u5e74\u4e2d \u5176\u9762\u79ef\u7531\u76f8\u5f53\u4e8e\u7f8e\u56fd\u5357\u65b948\u5dde\u9762\u79ef\u603b\u548c \u7f29\u51cf\u4e8640%"
  }
}
```

To fine-tune the OpusMT en-zh model on the IWSLT data:

```bash
python ft.py \
    --model_name_or_path Helsinki-NLP/opus-mt-en-zh \
    --do_train \
    --do_eval \
    --do_predict \
    --source_lang en \
    --target_lang zh \
    --max_source_length 512 \
    --num_train_epochs 3 \
    --save_total_limit 5 \
    --load_best_model_at_end \
    --metric_for_best_model bleu \
    --eval_steps 5000 \
    --logging_steps 5000 \
    --save_steps 5000 \
    --evaluation_strategy steps \
    --train_file ./data/train.json \
    --test_file ./data/test.json \
    --validation_file ./data/validation.json \
    --output_dir ./opus_baseline \
    --per_device_train_batch_size=64 \
    --per_device_eval_batch_size=64 \
    --predict_with_generate \
    --num_beams 4
```

After training a model, perform inference with the following script, which will generate predictions for the given `<test_file>` and save it to `<output_dir>/generated_predictions.txt`:

```bash
python ft.py \
    --model_name_or_path ./opus_baseline \
    --do_predict \
    --source_lang en \
    --target_lang zh \
    --max_source_length 512 \
    --test_file ./data/test.json \
    --validation_file ./data/validation.json \
    --output_dir ./opus_baseline \
    --per_device_train_batch_size=64 \
    --per_device_eval_batch_size=64 \
    --predict_with_generate \
    --num_beams 4
```

#### Note: Multiple GPUs

To utilise multiple GPUs, use `torchrun` to run `ft.py` instead of `python ft.py`. As an example, for 2 GPUs:

```bash
torchrun \
    --nproc_per_node 2 ft.py \
    <Rest of command here>
```
