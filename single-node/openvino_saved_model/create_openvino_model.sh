source ${INTEL_CVSDK_DIR}/bin/setups_vars.sh

# For CPU
python ${INTEL_CVSDK_DIR}/deployment_tools/model_optimizer/mo_tf.py --input_model ../frozen_model/saved_model_frozen.pb --input_shape=[1,144,144,4] --data_type FP32  --output_dir models/FP32  --model_name saved_model

# For NCS
python ${INTEL_CVSDK_DIR}/deployment_tools/model_optimizer/mo_tf.py --input_model ../frozen_model/saved_model_frozen.pb --input_shape=[1,144,144,4] --data_type FP16  --output_dir models/FP16  --model_name saved_model

