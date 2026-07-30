use tonic::{transport::Server, Request, Response, Status, Streaming};
use tokio_stream::StreamExt;

use crate::shuffler::MixNetShuffler;
use crate::aggregation::parallel_krum_filter;

pub mod proto {
    tonic::include_proto!("bifrost");
}

use proto::operational_hub_server::{OperationalHub, OperationalHubServer};
use proto::operational_hub_client::OperationalHubClient;
use proto::{InboundGradient, StreamResponse};#[derive(Debug,Default)]
pub struct BifrostServer{
    pub shuffler:MixNetShuffler,
    pub byzantine_bounds:usize,
}

impl BifrostServer{
    pub fn new(threshold:usize,byzantine_bounds:usize)-> Self{
        Self{
            shuffler:MixNetShuffler::new(threshold),
            byzantine_bounds,
        }
    }
}

#[tonic::async_trait]
impl OperationalHub for BifrostServer{
    async fn stream_gradients(&self , request:Request<Streaming<InboundGradient>>,) -> Result<Response<StreamResponse>,Status>{
        let mut stream = request.into_inner();  //Unwaprs req from grpc
        let mut packet_count = 0 ;
        let mut buffered_vector: Vec<f32> = Vec::with_capacity(113);

        println!("[SERVER] Incoming HTTP/2 Streaming pipeline opened");

        while let Some(result) = stream.next().await{
            match result{
                Ok(packet) => {
                    packet_count += 1;
                    buffered_vector.extend(packet.values);
                }
                Err(_err) => {
                    eprintln!("[SERVER] Network Stream packet error !!!");
                    return Err(Status::internal("Stream broken during tranission"));
                }
            }
        }
        println!("[SERVER] Stream closed cleanly by client. Received {} packets ", packet_count);

        let master_weights = if let Some(shuffled_batch) = self.shuffler.submit_and_shuffle(buffered_vector.clone()){
            println!("[SERVER] Batch Threshold met! Executing Parallel Krum Aggregation...");
            parallel_krum_filter(&shuffled_batch,self.byzantine_bounds).unwrap_or(buffered_vector)
        }else{
            buffered_vector
        };

        let response = StreamResponse{
            success:true,
            message:format!("Successfully processed {} packets",packet_count),
            master_weights
        };
        Ok(Response::new(response))
    }
}

//Server Socket Listener
pub async fn run_server(port:u16,threshold:usize,byzantine_bounds:usize) -> Result<(),Box<dyn std::error::Error>>{
    let addr = format!("127.0.0.1:{}",port).parse()?;
    let service = BifrostServer::new(threshold,byzantine_bounds);

    println!("[SERVER] Booting BiFrost PeaceTime Engine hub on {}...",addr);

    Server::builder()
        .add_service(OperationalHubServer::new(service))
        .serve(addr)
        .await?;
    
    Ok(())
}

pub async fn run_real_client(port: u16,gradients_payload: Vec<f32>,node_id: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    
    let server_addr = format!("http://127.0.0.1:{}",port);
    println!("[CLIENT] Connecting to Bifrost server at {}",server_addr);

    let mut client = OperationalHubClient::connect(server_addr).await?;
    
    let packet = InboundGradient{
        node_id:node_id.to_string(),
        round_id:1,
        indices:(0..gradients_payload.len() as u32).collect(),
        values: gradients_payload,
    };
    
    let outbound_stream = tokio_stream::iter(vec![packet]);
    let response = client.stream_gradients(outbound_stream).await?;
    let res = response.into_inner();

    Ok(res.master_weights)
}



