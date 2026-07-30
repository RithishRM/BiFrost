// Phase 5 extension: Deterministic, unambiguous Krum validation.
//
// Bypasses gRPC and the MixNet shuffle entirely — calls parallel_krum_filter
// directly with a known-ordered set of vectors, then matches the returned
// selection back against the known inputs by exact equality. This proves,
// with certainty (not inference), which node Krum selected.

use bifrost_core::aggregation::{parallel_krum_filter, squared_euclidean_distance};
use bifrost_core::engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== BIFROST DETERMINISTIC KRUM VALIDATION (no networking, no shuffle) ===");

    let dataset_path = "data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv";
    if !std::path::Path::new(dataset_path).exists() {
        return Err(format!("Dataset not found at {}", dataset_path).into());
    }

    println!("[MAIN] Training local model on real dataset to derive honest gradient baseline...");
    let honest_baseline = engine::train_local_model(dataset_path)?;
    println!("[MAIN] Honest baseline computed ({} params).\n", honest_baseline.len());

    // Build 5 known, ordered, labeled vectors: 4 honest (tight jitter cluster),
    // 1 adversarial (deliberately scaled outlier).
    let mut labeled_nodes: Vec<(String, Vec<f32>)> = Vec::with_capacity(5);

    for i in 0..4 {
        let jitter_scale = 1.0 + (i as f32) * 0.01;
        let honest_vector: Vec<f32> = honest_baseline.iter().map(|g| g * jitter_scale).collect();
        labeled_nodes.push((format!("honest-node-{}", i), honest_vector));
    }
    let adversarial_vector: Vec<f32> = honest_baseline.iter().map(|g| g * -50.0 + 100.0).collect();
    labeled_nodes.push(("adversarial-node".to_string(), adversarial_vector));

    // Print the known submission order explicitly.
    println!("[MAIN] Known input order (index -> label):");
    for (i, (label, _)) in labeled_nodes.iter().enumerate() {
        println!("  [{}] {}", i, label);
    }

    // Print pairwise distances manually so you can see the adversarial node's
    // distances are enormous compared to the honest cluster's.
    println!("\n[MAIN] Pairwise squared Euclidean distances:");
    for i in 0..labeled_nodes.len() {
        for j in (i + 1)..labeled_nodes.len() {
            let dist = squared_euclidean_distance(&labeled_nodes[i].1, &labeled_nodes[j].1);
            println!(
                "  dist({}, {}) = {:.4}",
                labeled_nodes[i].0, labeled_nodes[j].0, dist
            );
        }
    }

    // Extract just the vectors (unshuffled, known order) for the real call.
    let gradient_vectors: Vec<Vec<f32>> = labeled_nodes.iter().map(|(_, v)| v.clone()).collect();
    let byzantine_bounds = 1; // tolerate 1 adversary among 5 nodes

    println!("\n[MAIN] Calling parallel_krum_filter directly on unshuffled, known-order input...\n");
    let selected = parallel_krum_filter(&gradient_vectors, byzantine_bounds);

    match selected {
        Some(selected_vector) => {
            // Match the returned vector back to its known label by exact equality.
            let matched_label = labeled_nodes
                .iter()
                .find(|(_, v)| v == &selected_vector)
                .map(|(label, _)| label.clone());

            match matched_label {
                Some(label) => {
                    println!("=== RESULT ===");
                    println!("Krum selected vector matching known node: '{}'", label);
                    if label == "adversarial-node" {
                        println!("*** WARNING: Krum selected the ADVERSARIAL node. Aggregation FAILED to filter it out. ***");
                    } else {
                        println!("*** CONFIRMED: Krum correctly selected an HONEST node and excluded the adversarial vector. ***");
                    }
                }
                None => {
                    println!("=== RESULT ===");
                    println!("Could not match returned vector to any known input (unexpected).");
                }
            }
        }
        None => {
            println!("=== RESULT ===");
            println!("Krum returned None (insufficient node count for given byzantine_bounds).");
        }
    }

    println!("\n=== DETERMINISTIC KRUM VALIDATION COMPLETE ===");
    Ok(())
}
