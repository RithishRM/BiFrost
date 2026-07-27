use rand::seq::SliceRandom;
use std::sync::{Arc,Mutex};

#[derive(Debug,Clone,Default)]
pub struct MixNetShuffler{
    buffer:Arc<Mutex<Vec<Vec<f32>>>>,threshold:usize,
}

impl MixNetShuffler{
    pub fn new(threshold:usize) -> Self{
        Self{
            buffer:Arc::new(Mutex::new(Vec::new())),threshold
        }
    }

    pub fn submit_and_shuffle(&self,vector:Vec<f32>) -> Option<Vec<Vec<f32>>>{
        let mut guard = self.buffer.lock().unwrap();
        guard.push(vector);
        
        if guard.len() >= self.threshold{
            let mut batch = std::mem::take(&mut *guard);

            let mut rng = rand::rng();
            batch.shuffle(&mut rng);
            println!("[MIXNET] Anonymized and shuffled batch of {} node vectors.", batch.len());
            Some(batch)
        }else{
            None
        }
    }
}
