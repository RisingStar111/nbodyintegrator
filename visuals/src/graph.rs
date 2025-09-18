pub struct PointGraph {
    pub values: Vec<(f64, f64)>,
    pub max_x: Option<f64>,
    pub max_y: Option<f64>,
    pub min_x: Option<f64>,
    pub min_y: Option<f64>,
}

impl PointGraph {
    pub fn series_fill_screen(&self) -> Vec<(f64, f64)> {
        if self.values.len() == 0 {return vec![]}
        let fraction = 1./self.values.len() as f64;
        let min_y = if self.min_y.is_some() {self.min_y.unwrap()} else {self.values.iter().min_by(|a, b| a.0.total_cmp(&b.0)).unwrap().0};
        let max_y = if self.max_y.is_some() {self.max_y.unwrap()} else {self.values.iter().max_by(|a, b| a.0.total_cmp(&b.0)).unwrap().0};
        if min_y == max_y {return vec![]}
        self.values.iter().enumerate().map(|(i, v)| (i as f64 * fraction, (v.0 - min_y) / (max_y - min_y))).collect()
    }
}