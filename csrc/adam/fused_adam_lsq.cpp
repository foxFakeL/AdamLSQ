#include "cpu_adam_lsq.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("adam_update_lsq", &ds_adam_step_lsq,
        "DeepSpeed Fused CPU Adam + LSQ update (C++)");
  m.def("create_adam_lsq", &create_adam_lsq_optimizer,
        "DeepSpeed Fused CPU Adam + LSQ (C++)");
  m.def("register_quant_params", &register_quant_params_lsq,
        "Register quantization parameters (delta, z) for LSQ optimizer (C++)");
  m.def("destroy_adam_lsq", &destroy_adam_lsq_optimizer,
        "DeepSpeed Fused CPU Adam + LSQ destroy (C++)");
}