# TPU reference

> **Note:** This is a text-only copy of the repository's `tpu.md`. The embedded
> base64 screenshots for the Cloud Console walkthrough have been stripped;
> `![][imageN]` markers show where they appeared. See the original `tpu.md` at the
> repo root for the images.

**TPU resources**

* **TPU Developers Hub:** [https://cloud.google.com/products/tpu/tpu-developer](https://cloud.google.com/products/tpu/tpu-developer)
* Section 2 of "How to Scale Your Model", "How to think about TPUs": [https://jax-ml.github.io/scaling-book/tpus/](https://jax-ml.github.io/scaling-book/tpus/)

**Getting throughput out of a TPU**

These apply to any framework targeting the MXU:

* **Align tensor dimensions to multiples of 128** (or at least 8/16) so they fit the
  Matrix Multiply Unit's systolic array tiles cleanly. "How to Scale Your Model"
  covers the reasoning.
* **Use bfloat16.** It is native on the MXU; float32 is emulated.
* **Avoid graph breaks in compiled forward passes** — no Python control flow on
  tensor values, no scalar-to-tensor conversions (`.item()`), no prints.
* **Keep shapes static.** Recompilation on a new shape stalls execution, so bucket
  sequence lengths and batch sizes rather than passing arbitrary dimensions.
