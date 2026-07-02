mod protocol;
mod engine;
mod network;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== BIFROST INITIALIZATION ===");

    let test_port = 50051;

    // 1. Spawn the Server into a background task thread
    tokio::spawn(async move {
        if let Err(e) = network::run_server(test_port).await {
            eprintln!("Server crashed: {:?}", e);
        }
    });

    // 2. Pause brief moment to allow socket allocation on your Arch machine
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // 3. Trigger your mock client stream
    network::run_mock_client(test_port).await?;

    Ok(())
}
