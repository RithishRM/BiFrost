use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::{Request,Response,Status,Streaming};
use tokio_stream::{Stream, StreamExt};

pub mod proto {
    tonic::include_proto!("bifrost");
}

use proto::operational_hub_server::{OperationalHub, OperationalHubServer};
use proto::operational_hub_client::OperationalHubClient;
use proto::{InboundGradient, StreamResponse};

#[derive(Debug,Default)]
pub struct BifrostServer{
    pub gradient_store: Arc<Mutex<Vec<InboundGradient>>>,
}

#[tonic::async_trait]
impl OperationalHub for BifrostServer{
    async fn stream_gradients(&self , request:Request<Streaming<InboundGradient>>,) -> Result<Response<StreamResponse>,Status>{
        let mut stream = request.into_inner();  //Unwaprs req from grpc
        let mut packet_count = 0 ;
        let mut node_identifier = String::from("Unknown");

        println!("[SERVER] Incoming HTTP/2 Streaming pipeline opened");

        while let Some(result) = stream.next().await{
            match result{
                Ok(packet) => {
                    node_identifier = packet.node_id.clone();
                    packet_count += 1;
                    
                    let mut store = self.gradient_store.lock().await; //To lock shared mem
                    store.push(packet);
                }
                Err(err) => {
                    eprintln!("[SERVER] Network Stream packet error !!!");
                    return Err(Status::internal("Stream broken during tranission"));
                }
            }
        }
        println!("[SERVER] Stream closed cleanly by client. Received {} packets from node: '{}'.", packet_count, node_identifier);

        let response = StreamResponse{
            success:true,
            message:format!("Successfully buffered {} gradient frames.",packet_count),
        };
        Ok(Response::new(response))
    }
}

//Server Socket Listener
pub async fn run_server(port:u16) -> Result<(),Box<dyn std::error::Error>>{
    let addr = format!("127.0.0.1:{}",port).parse()?;
    let server_instance = BifrostServer::default();

    println!("[SERVER] Booting BiFrost PeaceTime Engine hub on {}...",addr);

    tonic::transport::Server::builder()
        .add_service(OperationalHubServer::new(server_instance))
        .serve(addr)
        .await?;

    
    Ok(())
}

pub async fn run_mock_client(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let server_addr = format!("http://127.0.0.1:{}", port);
    println!("[CLIENT] Connecting to BiFrost server at {}...", server_addr);

    // Establish the persistent HTTP/2 channel handshake
    let mut client = OperationalHubClient::connect(server_addr).await?;

    // Create a bounded multi-producer, single-consumer channel to push messages into
    let (tx, rx) = tokio::sync::mpsc::channel(32);

    // Convert the receiver end into a standard stream layout that tonic expects
    let request_stream = tokio_stream::wrappers::ReceiverStream::new(rx);

    // Spawn a background runtime thread to populate data without blocking networking
    tokio::spawn(async move {
        println!("[CLIENT] Simulating local AI gradient generation loop...");
        for i in 1..=5 {
            let mock_gradient = InboundGradient {
                node_id: "arch-edge-node-1".to_string(),
                round_id: 1,
                indices: vec![0, 1, 2, 3, 4],
                values: vec![0.1 * i as f32; 5], // Mocking varying float data
            };

            if tx.send(mock_gradient).await.is_err() {
                eprintln!("[CLIENT] Internal pipe broken; stream receiver dropped.");
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
        }
        println!("[CLIENT] Finished generating mock updates. Closing pipeline.");
    });

    // Fire the stream across the network socket
    let response = client.stream_gradients(request_stream).await?;

    println!(
        "[CLIENT] Server Response Acknowledgment: {:?}",
        response.into_inner()
    );

    Ok(())
}



