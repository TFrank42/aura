pub struct GGUFHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub _metadata_count: u64,
}

pub struct GGUFParser;

impl GGUFParser {
    pub fn peek_header(data: &[u8]) -> Option<GGUFHeader> {
        if data.len() < 24 || &data[0..4] != b"GGUF" {
            return None;
        }
        let version = u32::from_le_bytes(data[4..8].try_into().ok()?);
        let tensor_count = u64::from_le_bytes(data[8..16].try_into().ok()?);
        let metadata_count = u64::from_le_bytes(data[16..24].try_into().ok()?);

        Some(GGUFHeader { version, tensor_count,  _metadata_count: metadata_count })
    }
}
