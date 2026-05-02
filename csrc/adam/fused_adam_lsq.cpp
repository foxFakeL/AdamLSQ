#include "cpu_adam_lsq.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("adam_update_lsq", &ds_adam_step_lsq,
        "DeepSpeed Fused CPU Adam + LSQ update (C++) - 无状态架构");
  m.def("adam_update", &ds_adam_step,
        "DeepSpeed Fused CPU Adam update (C++) - 无量化，纯Adam");
  m.def("create_adam_lsq", &create_adam_lsq_optimizer,
        "DeepSpeed Fused CPU Adam + LSQ (C++)");
  m.def("destroy_adam_lsq", &destroy_adam_lsq_optimizer,
        "DeepSpeed Fused CPU Adam + LSQ destroy (C++)");
}