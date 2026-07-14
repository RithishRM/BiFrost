use rayon::prelude::*;

pub fn squared_euclidean_distance(v1: &[f32],v2: &[f32]) -> f32{
    let mut sum = 0.0;
    for i in 0..v1.len(){
        let diff = v1[i] - v2[i];
        sum += diff*diff;
    }
    sum
}

#[cfg(test)]

mod tests{
    use super::*;
    
    #[test]
    fn test_high_dimentional_distance(){
        let node_a = vec![1.0f32;113];
        let node_b = vec![2.0f32;113];
        let distance = squared_euclidean_distance(&node_a,&node_b);
        assert_eq!(distance,113.0,"Math engine distance calculation error !");
        println!("[TEST] Geometry Engine Passed Verified Calucation");
    }
}


pub fn parallel_krum_filter(gradients: &[Vec<f32>],byzantine_bounds:usize) -> Option<Vec<f32>>{
    let n =  gradients.len();

    if n<=2*byzantine_bounds + 2{
        eprintln!("[KRUM] Insufficient node count ({}) to tolerate {} adversaries",n,byzantine_bounds);
        return None;
    }

    let neighbors_to_sum = n - byzantine_bounds -2;

    println!("[KRUM] Evaluating geometric scores across {} parallel workers...",rayon::current_num_threads());

    let scores:Vec<(usize,f32)> = (0..n)
        .into_par_iter()
        .map(|i|{
            let mut distances = Vec::with_capacity(n-1);
            for j in 0..n{
                if i!=j{
                    distances.push(squared_euclidean_distance(&gradients[i],&gradients[j]));
                }
                }
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let score:f32 = distances.iter().take(neighbors_to_sum).sum();
            (i,score)

        })
        .collect();
    let best_match = scores
                            .iter()
                            .min_by(|(_, score_a), (_, score_b)| {
                                score_a.partial_cmp(score_b).unwrap_or(std::cmp::Ordering::Equal)
                            });

    if let Some(&(best_index, lowest_score)) = best_match {
            println!("[KRUM] Filter complete! Selected Node Index [{}] (Score: {})", best_index, lowest_score);
            Some(gradients[best_index].clone())
    }else {
        None
    }
}
