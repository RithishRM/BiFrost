use bifrost_core::{engine, network};

use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== BIFROST COHESIVE SYSTEM TEST (PHASE 3 + PHASE 4) ===");
    let test_port = 50051;
    let threshold = 1; // Batch size threshold for MixNet shuffling
    let byzantine_bounds = 0; // Number of Byzantine nodes to tolerate in Krum

    let test_csv_path = "data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv";

    if !std::path::Path::new(test_csv_path).exists() {
        return Err(format!(
            "Dataset not found at {}. Download MachineLearningCSV.zip from CICIDS2017 \
             and unzip it into data/MachineLearningCVE/",
            test_csv_path
        )
        .into());
    }

    println!("[MAIN] Using dataset: {}", test_csv_path);

    let input_dim = 4;
    let hidden_dim = 8;
    let output_dim = 1;
    let mut local_model = engine::Bifrostmodel::new(input_dim, hidden_dim, output_dim);

    let parameter_count = 113; // Exact size of flattened BiFrost model
    let mut local_error_buffer = engine::ErrorAccumulator::new(parameter_count);

    tokio::spawn(async move {
        if let Err(e) = network::run_server(test_port, threshold, byzantine_bounds).await {
            eprintln!("[SERVER] Server crashed: {:?}", e);
        }
    });

    // Pause to allow gRPC socket binding
    tokio::time::sleep(Duration::from_millis(500)).await;

    for round in 1..=3 {
        println!("\n=================== STARTING ROUND {} ===================", round);

        let mut computed_gradients = engine::train_local_model(test_csv_path)?;
        println!("[TEST] Raw Grad[0] before error merge: {:.6}", computed_gradients[0]);

        for (grad, past_error) in computed_gradients.iter_mut().zip(local_error_buffer.residue.iter()) {
            *grad += past_error;
        }
        println!("[TEST] Grad[0] after residue merge:    {:.6}", computed_gradients[0]);

        let compression_ratio = 0.10; // Target top 10%
        let mut magnitudes = engine::calculate_absolute_magnitudes(&computed_gradients);
        let k_threshold = engine::calculate_top_k_threshold(&mut magnitudes, compression_ratio);

        let compressed_indices = engine::build_index_mask(&computed_gradients, k_threshold);
        let compressed_values = engine::extract_top_k_values(&computed_gradients, &compressed_indices);

        println!(
            "[MAIN] Compression complete! Active parameters: {}/{} (Threshold: {:.4})",
            compressed_indices.len(),
            computed_gradients.len(),
            k_threshold
        );

        local_error_buffer.residue = engine::calculate_residue(&computed_gradients, &compressed_indices);
        let residue_sum: f32 = local_error_buffer.residue.iter().map(|&x| x.abs()).sum();
        println!("[TEST] Total residue banked for next round: {:.4}", residue_sum);

        println!("[CLIENT] Transmitting Round {} payload to OperationalHub...", round);
        let node_id = "arch-edge-node-1";

        // Transmits gradients and retrieves consensus master weights from server
        let master_weights = network::run_real_client(test_port, computed_gradients, node_id).await?;

        if !master_weights.is_empty() {
            local_model.update_weights(&master_weights);
        } else {
            println!("[CLIENT] Master weights pending (waiting for MixNet batch threshold).");
        }
    }

    println!("\n=== BIFROST INTEGRATED SYSTEM RUN SUCCESSFUL ===");
    Ok(())
}
