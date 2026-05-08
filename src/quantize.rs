use crate::TensorType;

#[derive(Debug, Clone)]
pub struct LoadedTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

pub fn dequantize(tensor_type: TensorType, raw: &[u8], shape: &[usize]) -> Result<LoadedTensor, String> {
    let data = match tensor_type {
        TensorType::F32 => dequantize_f32(raw)?,
        TensorType::F16 => dequantize_f16(raw)?,
        TensorType::Q4_0 => dequantize_q4_0(raw)?,
        TensorType::Q8_0 => dequantize_q8_0(raw)?,
        _ => return Err(format!("Type {:?} is unsupported in Rust! Please requantize to Q4_0.", tensor_type)),
    };

    Ok(LoadedTensor {
        data,
        shape: shape.to_vec(),
    })
}

fn dequantize_f32(raw: &[u8]) -> Result<Vec<f32>, String> {
    if raw.len() % 4 != 0 { return Err("F32 bad length".into()); }
    let mut out = Vec::with_capacity(raw.len() / 4);
    for chunk in raw.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn dequantize_f16(raw: &[u8]) -> Result<Vec<f32>, String> {
    if raw.len() % 2 != 0 { return Err("F16 bad length".into()); }
    let mut out = Vec::with_capacity(raw.len() / 2);
    for chunk in raw.chunks_exact(2) {
        out.push(f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
    }
    Ok(out)
}

fn dequantize_q4_0(raw: &[u8]) -> Result<Vec<f32>, String> {
    const BLOCK_BYTES: usize = 18;
    const BLOCK_ELEMS: usize = 32;
    if raw.len() % BLOCK_BYTES != 0 { return Err("Q4_0 bad length".into()); }
    let mut out = Vec::with_capacity((raw.len() / BLOCK_BYTES) * BLOCK_ELEMS);
    
    for block in raw.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..18];
        
        // UN-SWIZZLE: GGUF stores 16 low nibbles, THEN 16 high nibbles!
        for i in 0..16 { 
            out.push(((qs[i] & 0x0F) as i8 - 8) as f32 * d); 
        }
        for i in 0..16 { 
            out.push((((qs[i] >> 4) & 0x0F) as i8 - 8) as f32 * d); 
        }
    }
    Ok(out)
}

fn dequantize_q8_0(raw: &[u8]) -> Result<Vec<f32>, String> {
    const BLOCK_BYTES: usize = 34;
    const BLOCK_ELEMS: usize = 32;
    if raw.len() % BLOCK_BYTES != 0 { return Err("Q8_0 bad length".into()); }
    let mut out = Vec::with_capacity((raw.len() / BLOCK_BYTES) * BLOCK_ELEMS);
    for block in raw.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for &q in &block[2..34] { out.push((q as i8 as f32) * d); }
    }
    Ok(out)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits & 0x7C00) >> 10;
    let frac = (bits & 0x03FF) as u32;

    let f32_bits = if exp == 0 {
        if frac == 0 { sign } else {
            let mut f = frac; let mut s = 0;
            while (f & 0x0400) == 0 { f <<= 1; s += 1; }
            sign | ((127 - 15 - s + 1) << 23) | ((f & 0x03FF) << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (frac << 13)
    } else {
        sign | ((exp as u32 + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn dummy() {}
}
