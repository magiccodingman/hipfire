#!/usr/bin/env python3
from pathlib import Path

path = Path("crates/hipfire-quantize/src/pipeline.rs")
text = path.read_text()

# Regression 1: the stacked-MoE predicate was computed but not used, causing
# every tensor (including dense Qwen tensors) to enter handle_moe_expert_3d.
old_moe = '''        let is_moe_expert_3d = (is_moe || is_gemma4)
            && (name.ends_with("experts.gate_up_proj") || name.ends_with("experts.down_proj"))
            && meta.shape.len() == 3;
        {
            let __ctx = PerTensorCtx'''
new_moe = '''        let is_moe_expert_3d = (is_moe || is_gemma4)
            && (name.ends_with("experts.gate_up_proj") || name.ends_with("experts.down_proj"))
            && meta.shape.len() == 3;
        if is_moe_expert_3d {
            let __ctx = PerTensorCtx'''
if text.count(old_moe) != 1:
    raise SystemExit(f"expected exactly one stacked-MoE regression site, found {text.count(old_moe)}")
text = text.replace(old_moe, new_moe, 1)

# Regression 2: should_quantize() intentionally returns false for norms/biases
# so they remain F16, but the refactored handle_main_quant lost its old F16
# fallback. Those tensors were therefore silently omitted from the HFQ.
old_tail = '''            } // end else (non-Q8HFQ path)
        }
}
'''
new_tail = '''            } // end else (non-Q8HFQ path)
        } else {
            // Non-matmul tensors (norms, biases, small scalars, and included
            // vision tensors) are intentionally excluded by should_quantize().
            // Preserve them as F16 rather than silently dropping them from the
            // HFQ. The model-discovery pass has already removed FP8 scale
            // sidecars from all_tensors, and integer metadata is handled before
            // this function, so this is the intended floating-point fallback.
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name,
                raw_data,
                meta,
                fp8_scale_for,
                st_files,
            );
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            let f16_bytes: Vec<u8> = f32_data
                .iter()
                .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                .collect();

            *state.quantized_params += n_elements as u64;
            eprintln!(
                "  {:>8}: {} {:?} ({} elements, {:.1} KB -> {:.1} KB) [fallback]",
                "F16",
                name,
                meta.shape,
                n_elements,
                raw_data.len() as f64 / 1024.0,
                f16_bytes.len() as f64 / 1024.0
            );
            state.hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::F16,
                shape,
                group_size: 0,
                data: f16_bytes,
                spilled_len: 0,
            });
        }
}
'''
if text.count(old_tail) != 1:
    raise SystemExit(f"expected exactly one handle_main_quant tail, found {text.count(old_tail)}")
text = text.replace(old_tail, new_tail, 1)

path.write_text(text)
print("patched", path)
