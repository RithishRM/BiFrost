mod protocol;
mod engine;
mod network;
mod aggregation;
mod shuffler;

use std::fs::File;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== BIFROST PHASE 3: FULL SYSTEM INTEGRATION TEST ===");
    let test_port = 50051;
    let batch_threshold = 4;
    let byzantine_bounds = 1;

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

    // 2. Spawn gRPC Server with MixNet Shuffler & Krum Aggregator
    tokio::spawn(async move {
        if let Err(e) = network::run_server(test_port, batch_threshold, byzantine_bounds).await {
            eprintln!("Server crashed: {:?}", e);
        }
    });

    // Pause briefly for socket binding
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // 3. Initialize local model instance
    let mut local_model = engine::Bifrostmodel::new(4, 8, 1);

    // 4. Simulate Epoch 1 Training Iteration
    println!("\n--- EPOCH 1: TRAINING & SYNCHRONIZATION ---");
    let computed_gradients = engine::train_local_model(test_csv_path)?;
    println!("[CLIENT] DP-SGD vector generated (113 elements). Streaming to MixNet...");

    let master_weights = network::run_real_client(test_port, computed_gradients, "arch-edge-node-1").await?;
    println!("[CLIENT] Master consensus vector received from server!");

    // 5. Overwrite local weights with master consensus
    local_model.update_weights(&master_weights);

    // Clean up temporary files
    if std::path::Path::new(test_csv_path).exists() {
        std::fs::remove_file(test_csv_path)?;
    }

    println!("\n=== BIFROST PHASE 3 INTEGRATION COMPLETE: ALL PHASES OPERATIONAL ===");
    Ok(())
}
