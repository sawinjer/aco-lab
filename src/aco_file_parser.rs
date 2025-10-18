use std::{
    fs::File,
    io::{self, BufRead, BufReader},
    num::ParseIntError,
};

pub struct AcoFileParser {
    path: String,
}

impl AcoFileParser {
    pub fn new(path: String) -> Self {
        AcoFileParser { path }
    }

    pub fn read_file(&self) -> Result<AcoFile, AcoFileParseError> {
        let file = File::open(self.path.clone()).map_err(|err| AcoFileParseError::IoError(err))?;
        let reader = BufReader::new(file);
        let mut matrix_size: Option<usize> = None;
        let mut result_elements: Vec<f64> = Vec::new();

        for line in reader.lines() {
            let line = line.map_err(|err| AcoFileParseError::IoError(err))?;

            if line.starts_with('c') {
                continue;
            }

            if line.starts_with('p') {
                let parts: Vec<&str> = line.split(' ').collect();
                let size = parts.get(1).map(|size| {
                    size.parse::<usize>()
                        .map_err(|err| AcoFileParseError::InvalidSizeValue(err))
                });
                let size = match size {
                    Some(size) => Ok(size),
                    None => Err(AcoFileParseError::NoSizeSet),
                }??;

                matrix_size = Some(size);
            }

            if line.starts_with('i') {
                if let None = matrix_size {
                    return Err(AcoFileParseError::IvalidFileStructure);
                }

                let mut elements: Vec<f64> = line
                    .split(' ')
                    .filter_map(|el| {
                        if el == "i" {
                            return None;
                        }

                        el.parse::<f64>().ok()
                    })
                    .collect();

                if elements.len() != matrix_size.unwrap() {
                    return Err(AcoFileParseError::IvalidFileStructure);
                }

                result_elements.append(&mut elements);
            }
        }

        match matrix_size {
            Some(size) => Ok(AcoFile {
                size,
                rows: result_elements,
            }),
            None => Err(AcoFileParseError::EmptyFile),
        }
    }
}

pub struct AcoFile {
    pub size: usize,
    pub rows: Vec<f64>,
}

#[derive(Debug)]
pub enum AcoFileParseError {
    IoError(io::Error),
    NoSizeSet,
    InvalidSizeValue(ParseIntError),
    IvalidFileStructure,
    EmptyFile,
}
