use bifrost_core::engine::{
    self, build_index_mask, calculate_absolute_magnitudes, calculate_residue,
    calculate_top_k_threshold, extract_top_k_values, ErrorAccumulator,
};
use bifrost_core::protocol::GradientUpdate;

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== BIFROST MOCK NODE: PHASE 5 INDEPENDENT CORE TEST ===");

    // --- 1. DATA INGESTION ---
    let csv_data = "\
Destination_Port,Flow_Duration,Total_Fwd_Packets,Total_Backward_Packets,Label
80,1000.0,5,10,BENIGN
443,2000.0,10,20,ATTACK
8080,500.0,2,2,BENIGN
22,5000.0,15,30,ATTACK
21,7000.0,25,40,ATTACK
53,300.0,1,1,BENIGN
";
    let csv_path = "mock_node_traffic.csv";
    {
        let mut file = File::create(csv_path)?;
        file.write_all(csv_data.as_bytes())?;
    }

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

    // Cleanup local dataset
    if Path::new(csv_path).exists() {
        fs::remove_file(csv_path)?;
    }

    println!("\n=== MOCK NODE RUN COMPLETE — NO NETWORKING CODE EXECUTED ===");
    Ok(())
}
