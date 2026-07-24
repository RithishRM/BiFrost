use serde::{Deserialize, Serialize};

#[derive( Serialize, Deserialize, Debug, PartialEq)]
pub struct GradientUpdate{
    pub node_id : String,
    pub round_id: u32,
    pub indices : Vec<u32>,
    pub values : Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_serialization(){
        let g = GradientUpdate{
            node_id : "one".to_string(),
            round_id : 2,
            indices : vec![0,1,2],
            values : vec![3.0, 4.0, 5.0]
        };
        
        let serialized = serde_json::to_string(&g).unwrap();
        
        fs::write("target/test_payload.json", serialized).unwrap();
        
        let mesg : String = fs::read_to_string("target/test_payload.json").unwrap();
        
        let ng : GradientUpdate = serde_json::from_str(&mesg).unwrap();
        
        assert_eq!(g, ng, "testing {:?} and {:?}", g, ng);
    }
}
