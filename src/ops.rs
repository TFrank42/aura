#[repr(C, packed)]
pub struct BlockQ4_0 {
    pub d: f32,       
    pub qs: [u8; 16], 
}

pub fn dequantize_row_q4_0(data: &[u8], x: &mut [f32]) {
    let nb = x.len() / 32;
    let blocks: &[BlockQ4_0] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const BlockQ4_0, nb)
    };
    for i in 0..nb {
        let d = blocks[i].d;
        for j in 0..16 {
            let b = blocks[i].qs[j];
            let v0 = (b & 0x0F) as i8 - 8;
            let v1 = (b >> 4) as i8 - 8;
            x[i * 32 + j] = (v0 as f32) * d;
            x[i * 32 + j + 16] = (v1 as f32) * d;
        }
    }
}

pub fn matvec_mt(matrix: &[f32], vector: &[f32], out: &mut [f32], out_dim: usize, in_dim: usize, _threads: usize) {
    for r in 0..out_dim {
        let mut sum = 0.0;
        let row_start = r * in_dim;
        for c in 0..in_dim {
            sum += matrix[row_start + c] * vector[c];
        }
        out[r] = sum;
    }
}

pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv_ss = 1.0 / (ss + eps).sqrt();
    for i in 0..x.len() {
        out[i] = weight[i] * (x[i] * inv_ss);
    }
}

pub fn softmax(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0;
    for val in x.iter_mut() {
        *val = (*val - max).exp();
        sum += *val;
    }
    for val in x.iter_mut() {
        *val /= sum;
    }
}

pub fn silu(x: &mut [f32]) {
    for val in x.iter_mut() {
        *val = (*val) * (1.0 / (1.0 + (-*val).exp()));
    }
}
