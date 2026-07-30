use bifrost_core::{engine, network};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== BIFROST BYZANTINE-ROBUSTNESS TEST (5 nodes, 1 adversary) ===");

    let test_port = 50052; 
    let node_count = 5;
    let byzantine_bounds = 1; 
    let threshold = node_count;

    let dataset_path = "data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv";
    if !std::path::Path::new(dataset_path).exists() {
        return Err(format!("Dataset not found at {}", dataset_path).into());
    }

    tokio::spawn(async move {
        if let Err(e) = network::run_server(test_port, threshold, byzantine_bounds).await {
            eprintln!("[SERVER] Server crashed: {:?}", e);
        }
    });
    tokio::time::sleep(Duration::from_millis(500)).await;

    println!("[MAIN] Training local model on real dataset to derive honest gradient baseline...");
    let honest_baseline = engine::train_local_model(dataset_path)?;
    println!("[MAIN] Honest baseline computed ({} params).", honest_baseline.len());

    let mut node_payloads: Vec<(String, Vec<f32>)> = Vec::with_capacity(node_count);

    for i in 0..4 {
        let jitter_scale = 1.0 + (i as f32) * 0.01;
        let honest_vector: Vec<f32> = honest_baseline.iter().map(|g| g * jitter_scale).collect();
        node_payloads.push((format!("honest-node-{}", i), honest_vector));
    }

    let adversarial_vector: Vec<f32> = honest_baseline.iter().map(|g| g * -50.0 + 100.0).collect();
    node_payloads.push(("adversarial-node".to_string(), adversarial_vector));

    println!("[MAIN] Prepared {} node payloads (4 honest, 1 adversarial).", node_payloads.len());
    println!("[MAIN] Transmitting all 5 concurrently to trigger Krum aggregation...\n");

    let mut handles = Vec::with_capacity(node_count);
    for (node_id, gradients) in node_payloads {
        handles.push(tokio::spawn(async move {
            let result = network::run_real_client(test_port, gradients, &node_id)
                .await
                .map_err(|e| e.to_string());
            (node_id, result)
        }));
    }

    for handle in handles {
        let (node_id, result): (String, Result<Vec<f32>, String>) = handle.await?;
        match result {
            Ok(master_weights) => {
                println!(
                    "[MAIN] Node '{}' received response ({} master weight values).",
                    node_id,
                    master_weights.len()
                );
            }
            Err(e) => {
                println!("[MAIN] Node '{}' transmission failed: {}", node_id, e);
            }
        }
    }

    println!("\n=== LOOK ABOVE FOR '[KRUM] Filter complete! Selected Node Index [N]' ===");
    println!("=== Confirm the selected index does NOT correspond to the adversarial node's submission order ===");
    println!("=== BYZANTINE-ROBUSTNESS TEST COMPLETE ===");

    Ok(())
}
