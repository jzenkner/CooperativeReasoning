#!/bin/bash
# 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt

echo $CUDA_VISIBLE_DEVICES
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

rules=false

declare -a experiments_array=(
  "NONE"
  "LENGTH_GENERALIZATION"
  "ADD_OP_FUNCTIONALITY"
  "SWITCH_CONCEPT_ORDER"
  "COMPOSE_NEW_OP"
  "COMPOSE_DIFFERENT_CONCEPTS"
)


compute_false_positives=true
greedy=true
compute_align=false
nsa=false

num_examples=4

lambdabeam_max_list_length=5
lambdabeam_max_int=50
lambdabeam_max_const=5
max_program_arity=2
max_num_statements=5

eval_run=e2e_predict_1
save_dir=./stg_results/50ints/evaluation/lambdabeam_${eval_run}_falsepositives_${compute_false_positives}

# To use test data generated locally:
base_data_dir=../STG/lambdabeam_data/lambdabeam_data

# To use models trained locally:
train_run=1
base_model_dir=~/stg_results/50ints/exedec_train_lambdabeam_run-1

num_test=1000

# When testing for Rules et al we need the None model
if [[ "$rules" == "true" ]]; then
  experiment=NONE
  lambdabeam_max_list_length=10
  num_examples=3
  num_test=25
  base_data_dir=./tiips_data/iclr_rebuttal/lambdabeam
fi

if [ "$compute_false_positives" = true ]; then
  num_examples=$((num_examples))
fi

embedding_dim=360
hidden_dim=720

# Reimplement the length and distance computation from launch_train.py.
# It is important that these distances are exactly as used in training.
object_token_length=$((lambdabeam_max_list_length + 5))
max_input_objects=$((max_program_arity + max_num_statements - 1))
max_input_length=$((max_input_objects * object_token_length))
max_output_prediction_length=$(((num_examples - 1) * object_token_length))
max_program_part_length=7
spec_decomposer_max_distance=$((max_input_length > max_output_prediction_length ? max_input_length : max_output_prediction_length))
synthesizer_max_distance=$((max_input_length > max_program_part_length ? max_input_length : max_program_part_length))
spec_decomposer_max_program_cross_embed_distance=$((max_input_length * (num_examples - 1) > max_output_prediction_length ? max_input_length * (num_examples - 1) : max_output_prediction_length))
synthesizer_max_program_cross_embed_distance=$((max_input_length * (num_examples - 1) > max_program_part_length ? max_input_length * (num_examples - 1) : max_program_length))

echo "spec_decomposer_max_distance=${spec_decomposer_max_distance}"
echo "synthesizer_max_distance=${synthesizer_max_distance}"
echo "spec_decomposer_max_program_cross_embed_distance=${spec_decomposer_max_program_cross_embed_distance}"
echo "synthesizer_max_program_cross_embed_distance=${synthesizer_max_program_cross_embed_distance}"

# Compute lengths. These don't have to be exact, only long enough.
max_num_variables=10
max_io_length=$((max_num_variables * object_token_length))
max_num_program_parts=7
max_program_length=$((max_program_part_length * max_num_program_parts))
max_spec_part_length=50

test_dataset_format=${base_data_dir}/{experiment}_data/entire_programs_test.tf_records*
spec_decomposer_path_format=${base_model_dir}/spec_decomposer_model/train_predict_only/checkpoints/adr=0.1,ara=True,dr=0.1,e={experiment},ed=${embedding_dim},hd=${hidden_dim},l=5e-05,md=60,mpced=180,npb=32,s={seed},scnpr=0.0,ura=True,dm=standard/
synthesizer_path_format=${base_model_dir}/synthesizer_model/train_predict_only/checkpoints/adr=0.1,ara=False,dr=0.1,e={experiment},ed=${embedding_dim},hd=${hidden_dim},l=5e-05,md=60,mpced=180,npb=32,s={seed},scnpr={corruption_rate},ura=True,dm=standard/

if [[ "$nsa" == "true" ]]; then
  synthesizer_path_format=${base_model_dir}/joint_model/train_predict_only/checkpoints/adr=0.1,ara=False,dr=0.1,e={experiment},ed=${embedding_dim},hd=${hidden_dim},l=5e-05,md=60,mpced=180,npb=32,s={seed},scnpr=0.0,ura=True,dm=standard/
fi

# Generate comma-separated strings to pass as an argument.
experiments=$(printf ",%s" "${experiments_array[@]}")
experiments=${experiments:1}

for prediction_type in tiips separate baseline; do
  echo "${prediction_type}"
  echo ${num_test}

  python -m spec_decomposition.launch_end_to_end_predict \
  --exp_title=end_to_end_predict-lambdabeam-run-${eval_run}-${prediction_type} \
  --compute_false_positives=${compute_false_positives} \
  --greedy_selection=${greedy} \
  --compute_align=${compute_align} \
  --save_dir=${save_dir} \
  --dataset_type=lambdabeam \
  --experiments=${experiments} \
  --max_list_length=${lambdabeam_max_list_length} \
  --max_int=${lambdabeam_max_int} \
  --test_dataset_format=${test_dataset_format} \
  --num_test_batches=${num_test} \
  --num_examples=${num_examples} \
  --max_io_length=${max_io_length} \
  --max_program_length=${max_program_length} \
  --max_spec_part_length=${max_spec_part_length} \
  --spec_decomposer_path_format=${spec_decomposer_path_format} \
  --synthesizer_path_format=${synthesizer_path_format} \
  --embedding_dim=${embedding_dim} \
  --hidden_dim=${hidden_dim} \
  --spec_decomposer_num_position_buckets=32 \
  --synthesizer_num_position_buckets=32 \
  --spec_decomposer_max_distance=${spec_decomposer_max_distance} \
  --synthesizer_max_distance=${synthesizer_max_distance} \
  --spec_decomposer_max_program_cross_embed_distance=${spec_decomposer_max_program_cross_embed_distance}  \
  --synthesizer_max_program_cross_embed_distance=${synthesizer_max_program_cross_embed_distance} \
  --use_relative_attention=True \
  --beam_size=10 \
  --prediction_type=${prediction_type} \
  --detect_invalid=true \
  --use_execution=true \
  --discard_repeat_functionality=true \
  --aligned_relative_attention=true \
  --corruption_rate=0.0 \
  --seed=20
done