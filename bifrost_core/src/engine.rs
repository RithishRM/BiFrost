use serde::{Deserialize, Serialize};
use std::fs::File;
use std::collections::VecDeque;
use std::path::Path;
use std::error::Error;
use ndarray::{Array1, Array2, ArrayView2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_distr::{Normal, Distribution};
use std::cmp::Ordering;

// Fixed: Imported the gRPC communication struct so the compiler recognizes it
use crate::protocol::GradientUpdate;

#[derive( Serialize, Deserialize, Debug)]
pub struct RawFlowRecord {
    #[serde(rename = "Destination_Port")]
    pub Destination_Port : u32,
    #[serde(rename = "Flow_Duration")]
    pub Flow_Duration : f32,
    #[serde(rename = "Total_Fwd_Packets")]
    pub Total_Fwd_Packets : u32,
    #[serde(rename = "Total_Backward_Packets")]
    pub Total_Backward_Packets : u32,
    #[serde(rename = "Label")]
    pub Label : String,
}

pub struct Bifrostmodel {
    // i -> input, h -> hidden, o ->output, r -> recurring
    ihweight : Array2<f32>,
    hhweight : Array2<f32>,
    howeight : Array2<f32>,
    hbias : Array1<f32>,
    obias : Array1<f32>,
}

impl Bifrostmodel {
    pub fn new(input_dim: usize, hidden_dim : usize, output_dim : usize) ->Self {
        // Xavier/Glorot uniform boundary thresholds calculation
        let ih_bound = (6.0/(input_dim + hidden_dim) as f32).sqrt();
        let hh_bound = (6.0/(input_dim + hidden_dim) as f32).sqrt();
        let ho_bound = (6.0/(input_dim + hidden_dim) as f32).sqrt();

        let iweights = Array2::random((input_dim, hidden_dim), Uniform::new(-ih_bound, ih_bound).unwrap());
        let rweights = Array2::random((hidden_dim, hidden_dim), Uniform::new(-hh_bound, hh_bound).unwrap());
        let oweights = Array2::random((hidden_dim, output_dim), Uniform::new(-ho_bound, ho_bound).unwrap());
        let hidden_bias = Array1::zeros(hidden_dim);
        let output_bias = Array1::zeros(output_dim);

        Self {
            ihweight : iweights,
            hhweight: rweights,
            howeight: oweights,
            hbias: hidden_bias,
            obias: output_bias,
        }
    }

    pub fn forward(&self, sequence : &[f32]) -> (f32, Vec<Array1<f32>>) {
        //zero copy reshaping
        let seq_view = ndarray::ArrayView2::from_shape((3, 4), sequence)
            .expect("Sequence slice must contain exactly 12 elements (3x4)");

        let mut state_cache = Vec::with_capacity(3);

        let hidden_dim = self.hbias.len();
        let mut cur_hidden_base = Array1::zeros(hidden_dim);

        for row_stream in seq_view.outer_iter(){
            let input_trans = row_stream.dot(&self.ihweight);

            let recurrent_trans = cur_hidden_base.dot(&self.hhweight);

            let next_hidden = (input_trans + recurrent_trans + &self.hbias)
                .mapv(|x| x.tanh());

            state_cache.push(next_hidden.clone());

            cur_hidden_base = next_hidden;
        }

        let final_hidden = &state_cache[2];
        let output_tensor = final_hidden.dot(&self.howeight);
        let raw_output = output_tensor[0];
        let x = raw_output + self.obias[0];

        let y_pred = 1.0/(1.0 + (-x).exp());
        
        (y_pred, state_cache)
    }

    pub fn backward(&self, sequence: &[f32], prediction: f32, target: f32, state_cache: &[Array1<f32>]) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array1<f32>, Array1<f32>) {
        //computing the Outer Layer Error and Gradients
        let delta_o = prediction - target;
        let d_obias = Array1::from_elem(1, delta_o);
        let d_howeight = state_cache[2].clone().to_shape((8,1)).unwrap().to_owned() * delta_o;

        let mut d_ihweight = Array2::zeros((4, 8));
        let mut d_hhweight = Array2::zeros((8, 8));
        let mut d_hbias = Array1::zeros(8);

        //To track downstream delta errors through time
        let mut delta_h_next = Array1::zeros(8);

        let seq_view = ArrayView2::from_shape((3,4), sequence).unwrap();

        for t in (0..=2).rev() {

            let h_current = &state_cache[t];  //current step activation
            let h_prev = if t == 0 {
                Array1::zeros(8)
            }else{
                state_cache[t-1].clone()  //previous step activation
            };

            let error_from_output = if t == 2 {
                self.howeight.dot(&Array1::from_elem(1, delta_o))
            } else{
                Array1::zeros(8)
            };

            let error_from_recurrent = self.hhweight.dot(&delta_h_next);

            let total_error_vector = error_from_output + error_from_recurrent;

            //Applying Tanh derviative element-wise
            let tanh_deri = h_current.mapv(|h| 1.0 - h * h);
            let delta_h_current = total_error_vector * tanh_deri;

            d_hbias += &delta_h_current;

            let h_prev_matrix = h_prev.to_shape((8, 1)).unwrap();
            
            let delta_h_owned = delta_h_current.clone();
            let delta_h_matrix = delta_h_owned.to_shape((1, 8)).unwrap();
            d_hhweight += &h_prev_matrix.dot(&delta_h_matrix);

            let x_t = seq_view.row(t).to_owned();
            let x_t_matrix = x_t.to_shape((4, 1)).unwrap();
            d_ihweight += &x_t_matrix.dot(&delta_h_matrix);

            delta_h_next = delta_h_current;
        }

        (d_ihweight, d_hhweight, d_howeight, d_hbias, d_obias)
    }

    pub fn flatten_gradients(
        &self,
        d_ihweight: &Array2<f32>,
        d_hhweight: &Array2<f32>,
        d_howeight: &Array2<f32>,
        d_hbias: &Array1<f32>,
        d_obias: &Array1<f32>,
        ) -> Vec<f32> {
        let mut flat_grad = Vec::with_capacity(113);

        flat_grad.extend(d_ihweight.iter().cloned());
        flat_grad.extend(d_hhweight.iter().cloned());
        flat_grad.extend(d_howeight.iter().cloned());
        flat_grad.extend(d_hbias.iter().cloned());
        flat_grad.extend(d_obias.iter().cloned());

        flat_grad
    }

}

pub fn L2NormCalc(gradients: &[f32]) -> f32 {
    gradients
        .iter()
        .map(|&g| g * g)
        .sum::<f32>()
        .sqrt()
}

pub fn gradientClipping(gradVec : &mut Vec<f32>, max_norm : f32) {
    //finding magnitude of the gradient vector
    let mag : f32 = L2NormCalc(gradVec);

    if mag > max_norm {
        let scale = max_norm / mag;
        for element in gradVec.iter_mut() {
            *element *= scale;
        }
    }
}

pub fn generate_gaussian_noise(
    max_norm: f32,
    epsilon: f32,
    delta: f32,
    rng: &mut impl rand::Rng,
) -> f32 {
    let log_term = (1.25/delta).ln();
    let numerator = max_norm * (2.0 * log_term).sqrt();
    let sigma = numerator/epsilon;

    let normal_dist = Normal::new(0.0, sigma)
        .expect("Failed to initialize Normal distribution. Ensure sigma is valid and positive.");

    normal_dist.sample(rng)
}

// Fixed: Renamed to perfectly match main.rs expectations
pub fn calculate_absolute_magnitudes(gradients : &[f32]) -> Vec<f32> {
    gradients
        .iter()
        .map(|&g| g.abs())
        .collect()
}

// Fixed: Renamed to perfectly match main.rs expectations, and corrected internal variable names
pub fn calculate_top_k_threshold(magnitudes : &mut [f32], compression_ratio : f32) -> f32{
    let n = magnitudes.len();
    if n == 0{
        return 0.0;
    }
    let top_elements = ((n as f32) * compression_ratio).round() as usize;
    // Fixed: Removed the trailing semicolon
    let top_elements = top_elements.clamp(1, n);

    let target_index = n - top_elements;

    magnitudes.select_nth_unstable_by(target_index, |a,b| {
        a.partial_cmp(b).unwrap_or(Ordering::Equal)
    });

    magnitudes[target_index]
}

pub fn build_index_mask(gradients : &[f32], cut_off_threshold : f32) -> Vec<u32> {
    gradients
        .iter()
        .enumerate()
        .filter(|&(_, &g)| g.abs() >= cut_off_threshold)
        .map(|(index, _)| index as u32)
        .collect()
}

pub fn extract_top_k_values(gradients : &[f32], index_mask : &[u32]) -> Vec<f32>{
    index_mask
        .iter()
        .map(|&idx| gradients[idx as usize])
        .collect()
}

pub fn compress_and_package(
    node_id: String,
    round_id: u32,
    gradients: &[f32],
    compression_ratio: f32,
) -> GradientUpdate {
    let mut magnitudes = calculate_absolute_magnitudes(gradients);

    let threshold = calculate_top_k_threshold(&mut magnitudes, compression_ratio);

    let indices = build_index_mask(gradients, threshold);

    let values = extract_top_k_values(gradients, &indices);

    GradientUpdate {
        node_id,
        round_id,
        indices,
        values,
    }
}

pub struct ErrorAccumulator {
    pub residue: Vec<f32>,
}

impl ErrorAccumulator {
    pub fn new(capacity : usize) -> Self {
        Self {
            // Fixed: Changed period (.) to colon (:)
            residue: vec![0.0; capacity],
        }
    }
}

pub fn calculate_residue(
    accumulated_gradients: &[f32],
    transmitted_indices: &[u32]
) -> Vec<f32> {
    let mut residue = accumulated_gradients.to_vec();

    for &idx in transmitted_indices {
        residue[idx as usize] = 0.0;
    }

    residue
}

pub struct FlowSequenceWindow {
    window_size : usize,
    buffer : VecDeque<Vec<f32>>,
}

impl FlowSequenceWindow {
    pub fn new(window_size : usize) ->Self {
        let buffer = VecDeque::with_capacity(window_size + 1);
        
        Self {
            window_size,
            buffer,
        }
    }

    pub fn eviction_engine(&mut self, new_vector: Vec<f32>) -> Option<Vec<Vec<f32>>> {
        if self.buffer.len() >= self.window_size{
            self.buffer.pop_front();
        }

        self.buffer.push_back(new_vector);

        if self.buffer.len() == self.window_size {
            let current_window : Vec<Vec<f32>> = self.buffer.iter().cloned().collect();
            Some(current_window)
        }else{
            None
        }
    }
}

pub fn max_bounds<P: AsRef<Path>>(path: P) -> Result<RawFlowRecord, Box<dyn Error>> {
    let mut max_duration : f32 = 0.0;
    let mut max_fwd_packets : u32 = 0;
    let mut max_bwd_packets : u32 = 0;

    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);
    
    for record_result in rdr.deserialize::<RawFlowRecord>() {
        let record = record_result?;
        
        if record.Flow_Duration > max_duration {
            max_duration = record.Flow_Duration;
        }
        
        if record.Total_Fwd_Packets > max_fwd_packets {
            max_fwd_packets = record.Total_Fwd_Packets;
        }

        if record.Total_Backward_Packets > max_bwd_packets {
            max_bwd_packets = record.Total_Backward_Packets;
        }
    }
    Ok(RawFlowRecord{
        Destination_Port: 0,
        Flow_Duration: max_duration,
        Total_Fwd_Packets: max_fwd_packets,
        Total_Backward_Packets: max_bwd_packets,
        Label: String::new(),
    })
}

pub fn normalize<P: AsRef<Path>>(
    path: P,
    bounds : &RawFlowRecord,
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>), Box<dyn Error>> {
    //To store processed tensors
    let mut all_features = Vec::new();
    let mut all_labels = Vec::new();

    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);

    //Applying Transformation mapping
    for record_result in rdr.deserialize::<RawFlowRecord>(){
        let record = record_result?;

        let duration_scaling = if record.Flow_Duration > 0.0{
            record.Flow_Duration/ bounds.Flow_Duration
        }else{
            0.0
        };

        let fwd_pac_scaling = if record.Total_Fwd_Packets > 0 {
            record.Total_Fwd_Packets as f32/ bounds.Total_Fwd_Packets as f32
        }else{
            0.0
        };

        let bwd_pac_scaling = if record.Total_Backward_Packets > 0 {
            record.Total_Backward_Packets as f32/ bounds.Total_Backward_Packets as f32
        }else{
            0.0
        };
        //Port Spatial Constraint Mapping
        let scaled_port = record.Destination_Port as f32 / 65535.0;

        //Target Label Normalization Mapping (0.0 = Benign, 1.0 = Threat Anomaly)
        let label_val = match record.Label.to_uppercase().trim() {
            "BENIGN" => 0.0,
            _ => 1.0,
        };

        let feature_vector = vec![scaled_port, duration_scaling, fwd_pac_scaling, bwd_pac_scaling];
        all_features.push(feature_vector);
        all_labels.push(label_val);
    }
    Ok((all_features, all_labels))
}

pub fn train_local_model<P : AsRef<Path>>(
    path: P,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let window_size = 3;

    let (x_train, y_train) = load_and_preprocess_dataset(path, window_size)?;

    println!("Successfully loaded {} sequence samples!", x_train.len());

    let model = Bifrostmodel::new(4,8,1);

    // Initialized batch memory state arrays before the sequence driver loop starts
    let mut batch_gradients = vec![0.0f32; 113];
    let sample_count = x_train.len() as f32;

    let max_norm_c = 1.0f32;  //The maximum L2-norm clipping threshold parameter C
    let epsilon = 1.0f32;     //Privacy budget loss parameter
    let delta = 1e-5f32;      //Privacy failure tolerance threshold

    let mut rng = rand::rng();

    for (sequence_slice, target_label) in x_train.iter().zip(y_train.iter()) {
        let (prediction, state_cache) = model.forward(sequence_slice);

        let (d_ih, d_hh, d_ho, d_hb, d_ob) = model.backward(sequence_slice, prediction, *target_label, &state_cache);
        
        // Layer Flattening Step
        let mut sample_flat_gradients = model.flatten_gradients(&d_ih, &d_hh, &d_ho, &d_hb, &d_ob);
        
        gradientClipping(&mut sample_flat_gradients, max_norm_c);

        // Thread-Safe Batch Accumulation Loop
        for (batch_g, sample_g) in batch_gradients.iter_mut().zip(sample_flat_gradients.iter()) {
            *batch_g += sample_g;
        }
    }
    if sample_count > 0.0 {
        for g in batch_gradients.iter_mut(){
            let noise_offset = generate_gaussian_noise(max_norm_c, epsilon, delta, &mut rng);
            *g += noise_offset;
        }

        //Normalize the Anonymized parameters
        for g in batch_gradients.iter_mut() {
            *g /= sample_count;
        }
        println!("DP-SGD Batch normalization complete! Anonymized gradient parameters generated.");
    }

    Ok(batch_gradients)
}

pub fn load_and_preprocess_dataset<P : AsRef<Path>>(
    path: P,
    window_size: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>), Box<dyn Error>> {
    let bounds = max_bounds(&path)?;

    let (features, labels) = normalize(&path, &bounds)?;

    let mut sequenced_features = Vec::new();
    let mut sequenced_labels = Vec::new();

    let mut window = FlowSequenceWindow::new(window_size);

    for (feature_vector, label) in features.into_iter().zip(labels) {
        if let Some(complete_sequence) = window.eviction_engine(feature_vector) {
            
            let flat_sequence: Vec<f32> = complete_sequence
                .into_iter()
                .flatten()
                .collect();

            sequenced_features.push(flat_sequence);

            sequenced_labels.push(label);
        }
    }

    Ok((sequenced_features, sequenced_labels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_bifrost_engine_pipeline() {
        // 1. Create a dummy CSV payload matching your RawFlowRecord keys
        let csv_data = "\
Destination_Port,Flow_Duration,Total_Fwd_Packets,Total_Backward_Packets,Label
80,1000.0,5,10,BENIGN
443,2000.0,10,20,ATTACK
8080,500.0,2,2,BENIGN
22,5000.0,15,30,ATTACK
";

        // 2. Write it out to a temporary local test file
        let test_path = Path::new("test_flows.csv");
        let mut file = File::create(test_path).unwrap();
        file.write_all(csv_data.as_bytes()).unwrap();

        // 3. Test data ingestion & sliding window flattening
        // With 4 rows and window_size = 3, we should get exactly 2 sequence windows
        let window_size = 3;
        let preprocess_res = load_and_preprocess_dataset(test_path, window_size);
        assert!(preprocess_res.is_ok(), "Preprocessing pipeline crashed!");
        
        // Fixed: Added an underscore prefix to suppress the unused variable warning
        let (x_train, _y_train) = preprocess_res.unwrap();
        assert_eq!(x_train.len(), 2, "Should have extracted exactly 2 temporal window frames");
        assert_eq!(x_train[0].len(), 12, "Each flat sequence must contain exactly 12 metrics (3 steps x 4 features)");

        // 4. Test execution model math (Forward + Backward + Batch Flattening)
        let train_res = train_local_model(test_path);
        assert!(train_res.is_ok(), "Local training execution loop failed!");

        // 5. Clean up the disk footprint
        std::fs::remove_file(test_path).unwrap();
    }
}
