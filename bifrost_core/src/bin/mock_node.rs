use bifrost_core::engine::{
    self, build_index_mask, calculate_absolute_magnitudes, calculate_residue,
    calculate_top_k_threshold, extract_top_k_values, ErrorAccumulator,
};
use bifrost_core::protocol::GradientUpdate;

use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== BIFROST MOCK NODE: PHASE 5 INDEPENDENT CORE TEST ===");

    // --- 1. DATA INGESTION ---
    // Real CICIDS2017 data (MachineLearningCSV.zip, unzipped into
    // data/MachineLearningCVE/). Using Monday since it's the smallest file
    // and benign-only, good for a first correctness pass.
    let csv_path = "data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv";

    if !Path::new(csv_path).exists() {
        return Err(format!(
            "Dataset not found at {}. Download MachineLearningCSV.zip from CICIDS2017 \
             and unzip it into data/MachineLearningCVE/",
            csv_path
        )
        .into());
    }

    println!("[MOCK NODE] Using real CICIDS2017 data: {}", csv_path);

    let parameter_count = 113;
    let mut error_buffer = ErrorAccumulator::new(parameter_count);
    let compression_ratio = 0.10;
    let node_id = "mock-node-0";

    let mut last_payload: Option<GradientUpdate> = None;

    for round in 1..=3u32 {
        println!("\n--- MOCK NODE ROUND {} ---", round);

        // --- 2. DP-SGD LOCAL TRAINING ---
        let mut gradients = engine::train_local_model(csv_path)?;

        // --- 3. ERROR ACCUMULATION (merge previous round's untransmitted residue) ---
        for (g, past_error) in gradients.iter_mut().zip(error_buffer.residue.iter()) {
            *g += past_error;
        }

        // --- 4. TOP-K SPARSIFICATION / PACKING ---
        let mut magnitudes = calculate_absolute_magnitudes(&gradients);
        let threshold = calculate_top_k_threshold(&mut magnitudes, compression_ratio);
        let indices = build_index_mask(&gradients, threshold);
        let values = extract_top_k_values(&gradients, &indices);

        // --- 5. BANK RESIDUE FOR NEXT ROUND ---
        error_buffer.residue = calculate_residue(&gradients, &indices);
        let residue_sum: f32 = error_buffer.residue.iter().map(|x| x.abs()).sum();

        println!(
            "[MOCK NODE] Round {}: packed {}/{} params (threshold {:.4}), residue banked: {:.4}",
            round,
            indices.len(),
            gradients.len(),
            threshold,
            residue_sum
        );

        last_payload = Some(GradientUpdate {
            node_id: node_id.to_string(),
            round_id: round,
            indices,
            values,
        });
    }

    // --- 6. OUTPUT GENERATION: serialize the final payload for Member 2 ---
    if let Some(payload) = last_payload {
        let serialized = serde_json::to_string_pretty(&payload)?;

        fs::create_dir_all("payloads")?;
        let out_path = "payloads/sample_gradient_update.json";
        fs::write(out_path, &serialized)?;

        println!("\n[MOCK NODE] Sample GradientUpdate payload written to {}", out_path);
        println!("{}", serialized);
    }

    // NOTE: no cleanup of csv_path anymore — it's the real downloaded
    // dataset now, not a temp file we generated, so we leave it on disk.

    println!("\n=== MOCK NODE RUN COMPLETE — NO NETWORKING CODE EXECUTED ===");
    Ok(())
}
