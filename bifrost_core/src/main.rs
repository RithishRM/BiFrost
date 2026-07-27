mod protocol;
mod engine;
mod network;
mod aggregation;

use std::fs::File;
use std::io::Write;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== BIFROST COHESIVE SYSTEM TEST ===");
    let test_port = 50051;

    // 1. Generate local traffic dataset simulation
    let csv_data = "\
Destination_Port,Flow_Duration,Total_Fwd_Packets,Total_Backward_Packets,Label
80,1000.0,5,10,BENIGN
443,2000.0,10,20,ATTACK
8080,500.0,2,2,BENIGN
22,5000.0,15,30,ATTACK
";
    let test_csv_path = "local_network_traffic.csv";
    let mut file = File::create(test_csv_path)?;
    file.write_all(csv_data.as_bytes())?;

    // --- PHASE 4: INITIALIZE ERROR ACCUMULATOR ---
    let parameter_count = 113; // The exact size of your flattened BiFrost model
    let mut local_error_buffer = engine::ErrorAccumulator::new(parameter_count);

    // This will hold our final output for the server transmission step
    let mut final_computed_gradients = vec![];

    // ==========================================
    // START TEST LOOP: Simulate 3 rounds of training
    // ==========================================
    for round in 1..=3 {
        println!("\n--- STARTING ROUND {} ---", round);
        
        let mut computed_gradients = engine::train_local_model(test_csv_path)?;
        
        // Print the first index BEFORE merging
        println!("[TEST] Grad[0] before merge: {:.6}", computed_gradients[0]);
        
        // --- PHASE 4: ERROR ACCUMULATION (MERGE) ---
        // Merge past untransmitted errors into the fresh gradients
        for (grad, past_error) in computed_gradients.iter_mut().zip(local_error_buffer.residue.iter()) {
            *grad += past_error;
        }
        
        // Print the first index AFTER merging
        println!("[TEST] Grad[0] after merge:  {:.6}", computed_gradients[0]);

        // 3. Phase 3: Top-k Sparsification Compression
        let compression_ratio = 0.10; // Target the top 10%
        let mut magnitudes = engine::calculate_absolute_magnitudes(&computed_gradients);
        let threshold = engine::calculate_top_k_threshold(&mut magnitudes, compression_ratio);
        
        let compressed_indices = engine::build_index_mask(&computed_gradients, threshold);
        let _compressed_values = engine::extract_top_k_values(&computed_gradients, &compressed_indices);

        println!(
            "[MAIN] Compression complete! Active parameters: {}/{} (Threshold: {:.4})", 
            compressed_indices.len(), 
            computed_gradients.len(),
            threshold
        );

        // --- PHASE 4: CAPTURE AND SAVE THE RESIDUE ---
        local_error_buffer.residue = engine::calculate_residue(&computed_gradients, &compressed_indices);
        
        // Verify the residue actually holds data now
        let residue_sum: f32 = local_error_buffer.residue.iter().map(|&x| x.abs()).sum();
        println!("[TEST] Total residue banked for next round: {:.4}", residue_sum);

        // Save the current round's gradients so the client can transmit them after the loop ends
        final_computed_gradients = computed_gradients;
    }
    // ==========================================
    // END TEST LOOP
    // ==========================================


    println!("\n[MAIN] Training loops complete. Booting network layer...");

    // 4. Spawn Phase 1 gRPC Server in the background
    tokio::spawn(async move {
        if let Err(e) = network::run_server(test_port).await {
            eprintln!("Server crashed: {:?}", e);
        }
    });

    // Pause a split second to allow socket allocation
    tokio::time::sleep(Duration::from_millis(500)).await;

    // 5. Transmit to Server
    // Note: Once you update `network::run_real_client` to accept your new sparse protocol, 
    // you will pass `compressed_indices` and `compressed_values` here instead of the raw vector!
    network::run_real_client(test_port, final_computed_gradients).await?;

    // 6. Cleanup local footprint
    if std::path::Path::new(test_csv_path).exists() {
        std::fs::remove_file(test_csv_path)?;
    }
    
    println!("=== BIFROST RUN SUCCESSFUL ===");
    Ok(())
}
