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

pub async fn run_real_client(port: u16,trained_gradients: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
    
    let server_addr = format!("http://127.0.0.1:{}",port);
    println!("[CLIENT] Connecting to Bifrost server at {}",server_addr);

    let mut client = OperationalHubClient::connect(server_addr).await?;

    let (tx,rx) = tokio::sync::mpsc::channel(32);

    let request_stream = tokio_stream::wrappers::ReceiverStream::new(rx);

    tokio::spawn(async move {
        println!("[CLIENT] Preparing local AI trained gradients for transport...");
        let gradient_packet = InboundGradient {node_id: "arch-edge-node-1".to_string(),round_id: 1,indices: (0..trained_gradients.len() as u32).collect(),values: trained_gradients, };
    if tx.send(gradient_packet).await.is_err(){
            eprintln!("[CLIENT] Transmission channel error; stream receiver dropped.");}
            println!("[CLIENT] Real gradient payload pushed to gRPC network pipe.");
    });

    let response = client.stream_gradients(request_stream).await?;

    println!(
        "[CLIENT] Server Response Acknowledgment: {:?}",
        response.into_inner()
    );

    Ok(())
}



