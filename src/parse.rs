use crate::storage::Storage;

#[derive(Debug)]
pub enum ParseError {
    InvalidHeader,
    UnknownInstruction,
    IncompleteData,
}

pub fn parse(mut data: impl Iterator<Item = u32>) -> Result<super::Module, ParseError> {
    let header = {
        if data.next().ok_or(ParseError::IncompleteData)? != spirv::MAGIC_NUMBER {
            return Err(ParseError::InvalidHeader);
        }
        let version_raw = data.next().ok_or(ParseError::IncompleteData)?.to_le_bytes();
        super::Header {
            version: (version_raw[2], version_raw[1], version_raw[0]),
            generator: 0,
        }
    };
    Ok(super::Module {
        header,
        struct_declarations: Storage::new(),
        functions: Storage::new(),
        entry_points: Vec::new(),
    })
}

pub fn parse_u8_slice(data: &[u8]) -> Result<super::Module, ParseError> {
    use std::convert::TryInto;

    if data.len() % 4 != 0 {
        return Err(ParseError::IncompleteData);
    }

    parse(data
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
    )
}

#[cfg(test)]
mod test {
    #[test]
    fn parse() {
        let bin = vec![
            // Magic number.           Version number: 1.0.
            0x03, 0x02, 0x23, 0x07,    0x00, 0x00, 0x01, 0x00,
            // Generator number: 0.    Bound: 0.
            0x00, 0x00, 0x00, 0x00,    0x00, 0x00, 0x00, 0x00,
            // Reserved word: 0.
            0x00, 0x00, 0x00, 0x00,
            // OpMemoryModel.          Logical.
            0x0e, 0x00, 0x03, 0x00,    0x00, 0x00, 0x00, 0x00,
            // GLSL450.
            0x01, 0x00, 0x00, 0x00,
        ];
        let _ = super::parse_u8_slice(&bin).unwrap();
    }
}
