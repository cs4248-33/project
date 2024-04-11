# Harmonising the Chinese and English Language with IWSLT 2017 Dataset

By Felix Ong Wei Cong, Kum Wing Ho, Pow Zhi Xiang, Brandon Thio Zhi Kai, Lee Jia Wei

---

## Abstract

The IWSLT 2017 Chinese–English translation project aims to develop a robust sentence-level machine translation model using a large dataset of 230,000 paired sentences for training and 8500 paired sentences for testing.

We have conducted literature review of sentence-level translation models and techniques, identifying suitable neural network architectures that we can use, taking into account the tradeoffs between performance and compute requirements. We have also researched on preprocessing techniques and appropriate evaluation metrics.

## Code

This repo contains the relevant code used to augment the IWSLT 2017 Chinese–English dataset, as well as finetuning and evaluating the MarianMT model for the English-Chinese translation task.

### Supported Architectures

- `BartForConditionalGeneration`
- `FSMTForConditionalGeneration` (translation only)
- `MBartForConditionalGeneration`
- `MarianMTModel`
- `PegasusForConditionalGeneration`
- `T5ForConditionalGeneration`
- `MT5ForConditionalGeneration`

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

Here is an example of a translation fine-tuning with a MarianMT model:

```bash opus_finetune.sh
python ft.py \
    --model_name_or_path Helsinki-NLP/opus-mt-en-zh \
    --do_train \
    --do_eval \
    --do_predict \
    --source_lang en \
    --target_lang zh \
    --max_source_length 512 \
    --num_train_epochs 1 \
    --save_total_limit 2 \
    --eval_steps 5000 \
    --logging_steps 5000 \
    --save_steps 5000 \
    --evaluation_strategy steps \
    --train_file ./data/train.json \
    --test_file ./data/test.json \
    --validation_file ./data/validation.json \
    --output_dir ./opus \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
```

MBart and some T5 models require special handling.

T5 models `google-t5/t5-small`, `google-t5/t5-base`, `google-t5/t5-large`, `google-t5/t5-3b` and `google-t5/t5-11b` must use an additional argument: `--source_prefix "translate {source_lang} to {target_lang}"`. For example:

```bash mt5_finetune.sh
python ft.py \
    --model_name_or_path google/mt5-small \
    --do_train \
    --do_eval \
    --do_predict \
    --source_lang en \
    --target_lang zh \
    --source_prefix "translate English to Chinese: " \
    --num_train_epochs 1 \
    --save_total_limit 2 \
    --eval_steps 5000 \
    --logging_steps 5000 \
    --save_steps 5000 \
    --evaluation_strategy steps \
    --train_file ./data/train.json \
    --test_file ./data/test.json \
    --validation_file ./data/validation.json \
    --output_dir ./mt5 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
```

If you get a terrible BLEU score, make sure that you didn't forget to use the `--source_prefix` argument.

For the aforementioned group of T5 models it's important to remember that if you switch to a different language pair, make sure to adjust the source and target values in all 3 language-specific command line argument: `--source_lang`, `--target_lang` and `--source_prefix`.

MBart models require a different format for `--source_lang` and `--target_lang` values, e.g. instead of `en` it expects `en_XX`, for `zh` it expects `zh_CN`. The full MBart specification for language codes can be found [here](https://huggingface.co/facebook/mbart-large-cc25). For example:

```bash mbart_finetune.sh
python ft.py \
    --model_name_or_path facebook/mbart-large-50  \
    --do_train \
    --do_eval \
    --do_predict \
    --source_lang en_XX \
    --target_lang zh_CN \
    --num_train_epochs 1 \
    --save_total_limit 2 \
    --eval_steps 5000 \
    --logging_steps 5000 \
    --save_steps 5000 \
    --evaluation_strategy steps \
    --train_file ./data/train.json \
    --test_file ./data/test.json \
    --validation_file ./data/validation.json \
    --output_dir ./mbart \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
```

### Inference

After training a model, perform inference with the following script, which will generate predictions for the given `<test_file>` and save it to `<output_dir>/generated_predictions.txt:

```bash opus_predict
python ft.py \
    --model_name_or_path ./opus \
    --do_predict \
    --source_lang en \
    --target_lang zh \
    --max_source_length 512 \
    --test_file ./data/test.json \
    --validation_file ./data/validation.json \
    --output_dir ./opus \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
```

### Multiple GPUs

To utilise multiple GPUs, use `torchrun` to run `ft.py` instead of `python ft.py`. For example:

```bash opus_finetune_multi_gpu.sh
torchrun \
    --nproc_per_node 2 ft.py \
    --model_name_or_path Helsinki-NLP/opus-mt-en-zh \
    --do_train \
    --do_eval \
    --do_predict \
    --source_lang en \
    --target_lang zh \
    --max_source_length 512 \
    --num_train_epochs 1 \
    --save_total_limit 2 \
    --eval_steps 5000 \
    --logging_steps 5000 \
    --save_steps 5000 \
    --evaluation_strategy steps \
    --train_file ./data/train.json \
    --test_file ./data/test.json \
    --validation_file ./data/validation.json \
    --output_dir ./opus \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
```
