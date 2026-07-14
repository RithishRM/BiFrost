mod protocol;
mod engine;
mod network;
mod aggregation; // <--- Register Task 1 & 2

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== BIFROST PHASE 2 INTEGRATION: ATTACK SIMULATION ===");

    // 1. Simulate a cluster matrix of collected 113-element node updates
    // Let's assume we collected 5 nodes total, tolerating 1 attacker (f = 1)
    let mut collected_gradients = vec![
        vec![0.12f32; 113], // Node 0: Honest
        vec![0.11f32; 113], // Node 1: Honest
        vec![0.13f32; 113], // Node 2: Honest
        vec![0.12f32; 113], // Node 3: Honest
        vec![999.9f32; 113], // Node 4: MALICIOUS ATTACKER (Poisoned Gradients)
    ];

    println!("[MAIN] Ingested {} node updates into memory buffer.", collected_gradients.len());
    println!("[MAIN] Active Threat Level Configured: 1 Byzantine Attacker Allowed.");

    // 2. Pass the raw cluster matrix through our parallel Krum firewall
    let byzantine_bounds = 1; 
    if let Some(consensus_gradient) = aggregation::parallel_krum_filter(&collected_gradients, byzantine_bounds) {
        println!("[MAIN] Success! Secure Consensus Vector Extracted.");
        
        // Verify that the chosen vector is NOT the attacker's massive numbers
        if consensus_gradient[0] > 10.0 {
            println!("❌ FAILURE: The firewall let the poisoned attacker through!");
        } else {
            println!("✅ SUCCESS: Krum completely neutralized Node 4 and selected a clean update baseline (value: {})!", consensus_gradient[0]);
        }
    } else {
        println!("❌ FAILURE: Aggregation engine failed baseline validation.");
    }

    Ok(())
}
