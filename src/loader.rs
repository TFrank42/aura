use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use crate::{dequantize, GgufFile, LoadedTensor, TensorInfo};

pub fn load_tensor(gguf: &GgufFile, tensor: &TensorInfo, path: &str) -> Result<LoadedTensor, String> {
    let mut file = File::open(path).map_err(|e| format!("failed to open GGUF file: {}", e))?;

    let absolute_offset = gguf.data_offset + tensor.offset;
    file.seek(SeekFrom::Start(absolute_offset))
        .map_err(|e| format!("failed to seek tensor '{}': {}", tensor.name, e))?;

    let byte_len = tensor.byte_size();
    if byte_len == 0 {
        return Err(format!("tensor '{}' has zero byte size", tensor.name));
    }

    let mut raw = vec![0u8; byte_len];
    file.read_exact(&mut raw)
        .map_err(|e| format!("failed to read tensor '{}': {}", tensor.name, e))?;

    dequantize(tensor.tensor_type, &raw, &tensor.dimensions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorType;

    #[test]
    fn test_tensor_byte_size_nonzero() {
        let t = TensorInfo {
            name: "x".to_string(),
            dimensions: vec![32],
            tensor_type: TensorType::F32,
            offset: 0,
        };
        assert_eq!(t.byte_size(), 128);
    }

    #[test]
    fn test_load_tensor_rejects_missing_file() {
        let gguf = GgufFile {
            path: "dummy.gguf".to_string(),
            header: crate::GgufHeader {
                version: 3,
                tensor_count: 0,
                metadata_count: 0,
            },
            metadata: std::collections::HashMap::new(),
            tensors: vec![],
            data_offset: 0,
            alignment: 32,
        };

        let tensor = TensorInfo {
            name: "bad".to_string(),
            dimensions: vec![32],
            tensor_type: TensorType::F32,
            offset: 0,
        };

        let err = load_tensor(&gguf, &tensor, "does-not-exist.gguf").unwrap_err();
        assert!(err.contains("failed to open GGUF file"));
    }
}
