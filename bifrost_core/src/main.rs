mod protocol;
mod engine;
mod network;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== BIFROST INITIALIZATION ===");

    let test_port = 50051;

    // 1. Spawn the Server into a background task thread

use std::fs::File;
use std::io::Write;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== BIFROST SYSTEM TEST: MERGED ENGINE & NETWORK ===");
    let test_port = 50051;

    // 1. Generate local traffic dataset simulation for Member 1's engine
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

    // 2. Execute Member 1's local training engine to extract real model gradients
    println!("[MAIN] Triggering local RNN Training iteration...");
    let computed_gradients = engine::train_local_model(test_csv_path)?;
    println!("[MAIN] Engine execution complete. Vector Size: {} elements.", computed_gradients.len());

    // 3. Spawn your eager-reconstruction gRPC Server in the background
    tokio::spawn(async move {
        if let Err(e) = network::run_server(test_port).await {
            eprintln!("Server crashed: {:?}", e);
        }
    });


    // 2. Pause brief moment to allow socket allocation on your Arch machine
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // 3. Trigger your mock client stream
    network::run_mock_client(test_port).await?;

    // Allow a split second for socket allocation
    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

    // 4. Run the updated network client using the real data vector!
    network::run_real_client(test_port, computed_gradients).await?;

    // Clean up temporary files
    std::fs::remove_file(test_csv_path)?;
    println!("=== BIFROST COHESIVE SYSTEMS RUN SUCCESSFUL ===");
    Ok(())
}
