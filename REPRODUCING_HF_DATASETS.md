# Reproducing the Hugging Face datasets

This document collects the settings used to generate the PaS datasets released on Hugging Face (Food, Birds, Insects; see the table in the [README](README.md#datasets--checkpoints)), so a reproduction run can use them instead of the illustrative examples in each Stage's README.

## Stage 1: Concept generation

Most of Stage 1's hyperparameters are baked into `Stage_1/config/{food,birds,insects}.yaml`, but `--num_seeds` is a required CLI flag with no config-file equivalent: `../run.sh stage1 -- python3 main.py --config config/<domain>.yaml --output_dir output/<domain> --num_seeds <N>`.

Per-domain values:


|         | `--num_seeds` | `num_iters_seeds` | `seed_expansion.num_iters` | `target_num_concepts` | `force_stop`                       |
| ------- | -------------- | ----------------- | -------------------------- | --------------------- | ---------------------------------- |
| Food    | not recorded   | 5                  | 3                           | 5,000                  | false (`relative_threshold: 0.01`) |
| Birds   | 50              | 10                 | 10                          | 2,000                  | false (`relative_threshold: 0.01`) |
| Insects | 100             | 20                 | 20                          | 10,000                 | false (`relative_threshold: 0.01`) |


Food's per-call seed count was not recorded; pick a value in the range used for Birds/Insects.

## Stage 2: Web retrieval

`--suffix` matters: each run appended the domain's category description (or a short variant of it) to every concept before querying, not the bare concept name.

`--min_similarity` is the retrieval threshold λWeb. Its value was not recorded, so the commands below leave it unset (no filtering); the similarity of every retrieved row is written to the output parquet regardless, so a threshold can be applied afterwards.


|         | `--image_size` | `--retries` | `--suffix` (or equivalent) | `--num_images`/`--num_result_ids` |
| ------- | -------------- | ----------- | -------------------------- | --------------------------------- |
| Food    | 384            | 2           | `" food dish"`             | 1000                              |
| Birds   | 384            | 5           | `" bird"`                  | 1000                              |
| Insects | 324            | 10          | `" insect species"`        | not recorded                      |


Food:

```bash
python scripts/main_web.py \
    --query_files ../Stage_1/output/food/valid_concepts.json \
    --output_file output/food/web_retrieval.parquet \
    --prefix "" --suffix " food dish" \
    --indices_json_path laion5b/indices.json \
    --num_images 1000 --num_result_ids 1000

img2dataset \
    --url_list output/food/web_retrieval.parquet \
    --input_format parquet --url_col url \
    --output_folder output/food/images/web \
    --image_size 384 --resize_mode keep_ratio_largest --resize_only_if_bigger True \
    --output_format files --encode_format jpg --retries 2 \
    --incremental_mode extend \
    --disallowed_header_directives '["noai","noimageai","noindex","noimageindex"]'

mkdir -p output/food/images/web_wds
python scripts/generate_wds_from_web.py \
    --input output/food/images/web \
    --queries output/food/web_retrieval.parquet \
    --output output/food/images/web_wds
```

Birds:

```bash
python scripts/main_web.py \
    --query_files ../Stage_1/output/birds/valid_concepts.json \
    --output_file output/birds/web_retrieval.parquet \
    --prefix "" --suffix " bird" \
    --indices_json_path laion5b/indices.json \
    --num_images 1000 --num_result_ids 1000

img2dataset \
    --url_list output/birds/web_retrieval.parquet \
    --input_format parquet --url_col url \
    --output_folder output/birds/images/web \
    --image_size 384 --resize_mode keep_ratio_largest --resize_only_if_bigger True \
    --output_format files --encode_format jpg --retries 5 \
    --incremental_mode extend \
    --disallowed_header_directives '["noai","noimageai","noindex","noimageindex"]'

mkdir -p output/birds/images/web_wds
python scripts/generate_wds_from_web.py \
    --input output/birds/images/web \
    --queries output/birds/web_retrieval.parquet \
    --output output/birds/images/web_wds
```

Insects (`--num_images` was not recorded; set `<N_WEB>` to the number of web images to retrieve per concept):

```bash
python scripts/main_web.py \
    --query_files ../Stage_1/output/insects/valid_concepts.json \
    --output_file output/insects/web_retrieval.parquet \
    --prefix "" --suffix " insect species" \
    --indices_json_path laion5b/indices.json \
    --num_images <N_WEB> --num_result_ids <N_WEB>

img2dataset \
    --url_list output/insects/web_retrieval.parquet \
    --input_format parquet --url_col url \
    --output_folder output/insects/images/web \
    --image_size 324 --resize_mode keep_ratio_largest --resize_only_if_bigger True \
    --output_format files --encode_format jpg --retries 10 \
    --incremental_mode extend \
    --disallowed_header_directives '["noai","noimageai","noindex","noimageindex"]'

mkdir -p output/insects/images/web_wds
python scripts/generate_wds_from_web.py \
    --input output/insects/images/web \
    --queries output/insects/web_retrieval.parquet \
    --output output/insects/images/web_wds
```

`--prefix` was not recorded; the default (empty) is assumed here.

## Stage 2: Prompt generation

Synthetic generation needs a `prompts.json` first: `scripts/main_prompts.py` asks an LLM (L_4) for `--num-prompts` image descriptions per concept. `--num-prompts` is `N_Pro` in the paper's `|C| x N_Pro x N_Synth` formula, so skipping this step or changing its value changes the size of the synthetic set. All three domains used the same model and count:


|         | `-m`/`--model` | `-n`/`--num-prompts` | `-d`/`--domain` | `-cd`/`--category-desc` |
| ------- | -------------- | -------------------- | --------------- | ----------------------- |
| Food    | `mistral`      | 5                    | `food dish`     | `"food dish"`           |
| Birds   | `mistral`      | 5                    | `birds`         | `"bird species"`        |
| Insects | `mistral`      | 5                    | `insects`       | `"insect species"`      |


`--model` defaults to `mixtral`, so `-m mistral` has to be passed explicitly. Food:

```bash
mkdir -p output/food/
python3 scripts/main_prompts.py \
    -m mistral -n 5 \
    --concepts ../Stage_1/output/food/valid_concepts.json \
    --domain "food dish" --category-desc "food dish" \
    --output output/food/prompts.json
```

## Stage 2: Synthetic generation

All three domains used `--denoising-steps 20` throughout. **`--img-per-prompt` differs sharply by domain:**


|         | `--img-per-prompt` | `--denoising-steps` |
| ------- | ------------------ | ------------------- |
| Food    | 10                 | 20                  |
| Birds   | 10                 | 20                  |
| Insects | 100                | 20                  |


```bash
PYTHONPATH=. python3 scripts/main_synth.py \
    --prompts output/food/prompts.json \
    --output-dir output/food/images/synth \
    --denoising-steps 20 --img-per-prompt 10

mkdir -p output/food/images/synth_wds
PYTHONPATH=. python scripts/generate_wds_from_synth.py \
    --prompts output/food/prompts.json \
    --output-dir output/food/images/synth_wds \
    --input-dir output/food/images/synth \
    --img-per-prompt 10 --overwrite
```

## Stage 3: Evaluation-set overlap removal

Images that are near-duplicates of the benchmark test sets are removed: an image goes when its SSCD cosine similarity to the nearest evaluation image is above `0.45`, a stricter threshold than deduplication's `0.6`. The evaluation sets are the ones each domain is benchmarked on:

|         | Evaluation test sets                          |
| ------- | --------------------------------------------- |
| Food    | Food-101, FoodX-251, Food-2K                  |
| Birds   | CUB-200-2011, NABirds, iNatBirds              |
| Insects | AMI-GBIF (the benchmark's regional subsets)   |

```bash
python3 scripts/remove_eval_overlap.py \
    --embedding_dir embeddings/combined/sscd_large/ \
    --reference_embedding_dir embeddings/evaluation/sscd_large/ \
    --threshold 0.45 \
    --output_df_path curation/all/remove_eval_overlap.parquet
```

The test-set images themselves are not redistributed here; obtain each benchmark from its own source, pack it into a webdataset, and embed it with `scripts/generate_sscd_embeddings.py`. Pass the resulting parquet to `merge_statistics.py --eval_overlap_df_path`.

## Stage 3: Deduplication

All three domains use the same values: `--k 64 --threshold 0.6`, on the approximate IVF-PQ k-NN graph (`--knn_algorithm ivfpq`, the default).

Deduplication runs once, over the union of the web and synthetic sources, so near-duplicates are removed globally rather than within each source.

```bash
python ../Stage_2/scripts/combine_wds_sources.py \
    --sources web=../Stage_2/output/food/images/web_wds,synthetic=../Stage_2/output/food/images/synth_wds \
    --output_dir ../Stage_2/output/food/images/combined_wds

DF_PATHS=$(ls ../Stage_2/output/food/images/combined_wds/*.parquet | paste -sd, -)

python3 scripts/deduplication.py \
    --df_paths "$DF_PATHS" \
    --tar_dir ../Stage_2/output/food/images/combined_wds \
    --embedding_dir embeddings/combined/sscd_large/ \
    --output_df_path curation/all/remove_duplicates.parquet \
    --output_adj_matrix_path curation/all/02_adj_matrix_dedup_64nn_0.6.npz \
    --threshold 0.6 --k 64 --knn_algorithm ivfpq
```

## Stage 3: CLIPN OOD scoring

Checkpoint: `pretrained/clipn_ctw_vitb16.pt` (download per `Stage_3/README.md`'s CLIPN section; there is no direct URL, so follow the upstream [xmed-lab/CLIPN](https://github.com/xmed-lab/CLIPN) repo).


|         | `--batch_size` | base classes                    |
| ------- | -------------- | ------------------------------- |
| Food    | 1024           | `["food", "food dish"]`         |
| Birds   | 1024           | `["birds", "bird species"]`     |
| Insects | 4096           | `["insects", "insect species"]` |


In every domain the base `detect_ood_clipn` scoring uses two classes. Pass them as a JSON file, the same way `Stage_3/README.md`'s own example does:

```bash
echo '["food", "food dish"]' > classes.json

python scripts/ood_clipn.py \
    --pre_trained_model_path pretrained/clipn_ctw_vitb16.pt \
    --classes classes.json --batch_size 1024 \
    --df_paths "$DF_PATHS" \
    --tar_dir ../Stage_2/output/food/images/combined_wds \
    --output_df_path statistics/combined_detect_ood_clipn.parquet
```

`merge_statistics.py`'s `diff_mars` column is computed from `detect_ood_clipn` against `detect_ood_clipn_tmars` (the same base classes, but on the blurred images), so that variant is required too, not optional:
```bash
python scripts/ood_clipn.py \
    --pre_trained_model_path pretrained/clipn_ctw_vitb16.pt \
    --classes classes.json --batch_size 1024 \
    --df_paths "$DF_PATHS" \
    --tar_dir ../Stage_2/output/food/images/combined_wds \
    --append_blurred \
    --output_df_path statistics/combined_detect_ood_clipn_tmars.parquet
```

The `_queries` variant (`detect_ood_clipn_queries`) uses each domain's own Stage 1 `valid_concepts.json` output as its classes, as in `Stage_3/README.md`'s example.

```bash
python scripts/ood_clipn.py \
    --pre_trained_model_path pretrained/clipn_ctw_vitb16.pt \
    --classes ../Stage_1/output/food/valid_concepts.json --batch_size 1024 \
    --df_paths "$DF_PATHS" \
    --tar_dir ../Stage_2/output/food/images/combined_wds \
    --output_df_path statistics/combined_detect_ood_clipn_queries.parquet

python scripts/ood_clipn.py \
    --pre_trained_model_path pretrained/clipn_ctw_vitb16.pt \
    --classes ../Stage_1/output/food/valid_concepts.json --batch_size 1024 \
    --df_paths "$DF_PATHS" \
    --tar_dir ../Stage_2/output/food/images/combined_wds \
    --append_blurred \
    --output_df_path statistics/combined_detect_ood_clipn_queries_tmars.parquet
```

With all four variants produced, merge them:

```bash
python scripts/merge_statistics.py \
    --input_df_paths statistics/combined_detect_ood_clipn.parquet,statistics/combined_detect_ood_clipn_queries.parquet,statistics/combined_detect_ood_clipn_tmars.parquet,statistics/combined_detect_ood_clipn_queries_tmars.parquet \
    --dedup_df_path curation/all/remove_duplicates.parquet \
    --eval_overlap_df_path curation/all/remove_eval_overlap.parquet \
    --output_df_path statistics/all_statistics_after_dedup.parquet
```

## Stage 3: Pareto filtering

**Stop criteria differ across domains.** Food and Birds used `KneeBasedStopCriteria`; Insects used `RemainingImagesStopCriteria` (the one `Stage_3/README.md`'s own example shows, with a different threshold number).

Food and Birds (`columns`/`curves`/`directions` identical across both):

```bash
python scripts/pareto_filtering.py \
    --df_path statistics/all_statistics_after_dedup.parquet \
    --output_df_path curation/all/full_pareto_food.parquet \
    --flatten_columns detect_ood_clipn detect_ood_clipn_queries diff_mars \
    --pareto_columns detect_ood_clipn_YesNo detect_ood_clipn_queries_YesNo diff_mars_YesNo \
    --pareto_objectives min min max \
    --stop_criteria KneeBasedStopCriteria \
    --stop_criteria_args "{columns: [detect_ood_clipn_YesNo, detect_ood_clipn_queries_YesNo, diff_mars_YesNo], curves: [concave, concave, convex], directions: [increasing, increasing, decreasing]}"
```

Insects:

```bash
python scripts/pareto_filtering.py \
    --df_path statistics/all_statistics_after_dedup.parquet \
    --output_df_path curation/all/full_pareto_insects.parquet \
    --flatten_columns detect_ood_clipn detect_ood_clipn_queries diff_mars \
    --pareto_columns detect_ood_clipn_YesNo detect_ood_clipn_queries_YesNo diff_mars_YesNo \
    --pareto_objectives min min max \
    --stop_criteria RemainingImagesStopCriteria \
    --stop_criteria_args "{threshold_number: 1900000}"
```


## Stage 3: Export

The same for every domain: write the images Pareto filtering selected as the final webdataset.

```bash
python scripts/export_dataset.py \
    --pareto_df_path curation/all/full_pareto_food.parquet \
    --tar_dir ../Stage_2/output/food/images/combined_wds \
    --output_dir output/food/dataset
```
