use rayon::prelude::*;

use crate::error::{Result, MoonclipError};
use crate::manifest::CompressionAlgo;

/// Chunk size for parallel compression. Each chunk becomes an independent
/// zstd frame; the frames are concatenated. A concatenation of zstd frames
/// is itself a valid zstd stream, so old readers (streaming decode) and the
/// fallback path below decode it transparently.
const PARALLEL_CHUNK: usize = 4 << 20; // 4 MB

/// Compress raw bytes according to the chosen algorithm.
///
/// Buffers larger than `PARALLEL_CHUNK` are split and compressed in
/// parallel with rayon (one zstd frame per chunk). Single-shot frames
/// carry the content size in the header, enabling parallel decompression.
pub fn compress(data: &[u8], algo: &CompressionAlgo) -> Result<Vec<u8>> {
    match algo {
        CompressionAlgo::None => Ok(data.to_vec()),
        CompressionAlgo::Zstd { level } => {
            if data.len() <= PARALLEL_CHUNK {
                let mut out = zstd::bulk::compress(data, *level)
                    .map_err(|e| MoonclipError::Compression(e.to_string()))?;
                // Allocated at zstd's worst case, the input's size and more;
                // a delta compresses to a fraction of it, and a save holds a
                // batch of these until they are written (GPU-443).
                out.shrink_to_fit();
                return Ok(out);
            }
            // Every frame is compressed straight into its own slot of one
            // buffer, each slot as large as zstd's worst case for its chunk,
            // and the frames are then moved down to sit end to end. Gathering
            // them as separate frames and concatenating them held the tensor's
            // compressed bytes twice at the peak, per thread at once - for
            // state that barely compresses, two copies of every tensor being
            // compressed (GPU-443). The bytes produced are the same frames.
            let bound = zstd::zstd_safe::compress_bound(PARALLEL_CHUNK);
            let chunks = data.len().div_ceil(PARALLEL_CHUNK);
            let mut out = vec![0u8; bound * chunks];
            let sizes: std::io::Result<Vec<usize>> = data
                .par_chunks(PARALLEL_CHUNK)
                .zip(out.par_chunks_mut(bound))
                .map(|(chunk, slot)| zstd::bulk::compress_to_buffer(chunk, slot, *level))
                .collect();
            let sizes = sizes.map_err(|e| MoonclipError::Compression(e.to_string()))?;
            let mut end = 0;
            for (i, size) in sizes.into_iter().enumerate() {
                out.copy_within(i * bound..i * bound + size, end);
                end += size;
            }
            out.truncate(end);
            out.shrink_to_fit();
            Ok(out)
        }
    }
}

/// Frame layout of (possibly multi-frame) zstd data:
/// (offset, compressed_len, content_len) per frame.
/// Returns None if any frame is malformed or lacks a content size in the
/// header (e.g. legacy streaming-compressed data) — caller must fall back
/// to sequential streaming decode.
fn frame_layout(data: &[u8]) -> Option<Vec<(usize, usize, usize)>> {
    let mut frames = Vec::new();
    let mut off = 0;
    while off < data.len() {
        let rest = &data[off..];
        let csize = zstd::zstd_safe::find_frame_compressed_size(rest).ok()?;
        if csize == 0 || csize > rest.len() {
            return None;
        }
        let dsize = zstd::zstd_safe::get_frame_content_size(&rest[..csize])
            .ok()
            .flatten()?;
        frames.push((off, csize, dsize as usize));
        off += csize;
    }
    Some(frames)
}

/// Decompress bytes produced by `compress`.
///
/// Multi-frame data with known content sizes is decompressed in parallel;
/// anything else falls back to sequential streaming decode (handles legacy
/// single-frame data without a content size header).
pub fn decompress(data: &[u8], algo: &CompressionAlgo) -> Result<Vec<u8>> {
    match algo {
        CompressionAlgo::None => Ok(data.to_vec()),
        CompressionAlgo::Zstd { .. } => {
            if data.is_empty() {
                return Ok(Vec::new());
            }
            if let Some(frames) = frame_layout(data) {
                let total: usize = frames.iter().map(|f| f.2).sum();
                let mut out = vec![0u8; total];

                // Split the output buffer into one mutable slice per frame.
                let mut slices: Vec<&mut [u8]> = Vec::with_capacity(frames.len());
                let mut rest: &mut [u8] = &mut out;
                for f in &frames {
                    let (head, tail) = rest.split_at_mut(f.2);
                    slices.push(head);
                    rest = tail;
                }

                frames
                    .par_iter()
                    .zip(slices)
                    .try_for_each(|(f, dst)| -> std::result::Result<(), String> {
                        if dst.is_empty() {
                            return Ok(());
                        }
                        let n = zstd::bulk::decompress_to_buffer(&data[f.0..f.0 + f.1], dst)
                            .map_err(|e| e.to_string())?;
                        if n != dst.len() {
                            return Err(format!(
                                "frame decompressed to {} bytes, expected {}",
                                n,
                                dst.len()
                            ));
                        }
                        Ok(())
                    })
                    .map_err(MoonclipError::Compression)?;
                return Ok(out);
            }

            zstd::decode_all(std::io::Cursor::new(data))
                .map_err(|e| MoonclipError::Compression(e.to_string()))
        }
    }
}

/// Decompress at most `max_bytes` of raw data from the start of the stream.
///
/// `data` may be a truncated window of the full compressed stream: zstd
/// decodes block-by-block (blocks are ≤128 KB), so a prefix of the input
/// yields a prefix of the output. Used for cheap change-density sampling
/// against a base tensor without decompressing it entirely. May return
/// fewer bytes than requested; never errors once at least one byte was
/// produced.
pub fn decompress_prefix(data: &[u8], algo: &CompressionAlgo, max_bytes: usize) -> Result<Vec<u8>> {
    match algo {
        CompressionAlgo::None => Ok(data[..data.len().min(max_bytes)].to_vec()),
        CompressionAlgo::Zstd { .. } => {
            use std::io::Read;
            let mut decoder = zstd::stream::read::Decoder::new(std::io::Cursor::new(data))
                .map_err(|e| MoonclipError::Compression(e.to_string()))?;
            let mut out = vec![0u8; max_bytes];
            let mut filled = 0;
            while filled < max_bytes {
                match decoder.read(&mut out[filled..]) {
                    Ok(0) => break,
                    Ok(n) => filled += n,
                    // Truncated input: keep whatever was already decoded.
                    Err(_) if filled > 0 => break,
                    Err(e) => return Err(MoonclipError::Compression(e.to_string())),
                }
            }
            out.truncate(filled);
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_zstd() {
        let data = vec![42u8; 100_000];
        let algo = CompressionAlgo::Zstd { level: 3 };
        let compressed = compress(&data, &algo).unwrap();
        assert!(compressed.len() < data.len());
        let decompressed = decompress(&compressed, &algo).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn roundtrip_none() {
        let data = b"hello world".to_vec();
        let algo = CompressionAlgo::None;
        let compressed = compress(&data, &algo).unwrap();
        assert_eq!(data, compressed);
        let decompressed = decompress(&compressed, &algo).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn empty_data() {
        let algo = CompressionAlgo::Zstd { level: 3 };
        let compressed = compress(&[], &algo).unwrap();
        let decompressed = decompress(&compressed, &algo).unwrap();
        assert_eq!(decompressed, Vec::<u8>::new());
    }

    #[test]
    fn high_compression_level() {
        let data = vec![0u8; 50_000];
        let algo = CompressionAlgo::Zstd { level: 9 };
        let compressed = compress(&data, &algo).unwrap();
        let decompressed = decompress(&compressed, &algo).unwrap();
        assert_eq!(data, decompressed);
        assert!(compressed.len() < 100, "Repetitive data should compress to almost nothing");
    }

    #[test]
    fn random_data_still_roundtrips() {
        // Pseudo-random data doesn't compress well but should still roundtrip
        let data: Vec<u8> = (0..10_000).map(|i| ((i * 7 + 13) % 256) as u8).collect();
        let algo = CompressionAlgo::Zstd { level: 1 };
        let compressed = compress(&data, &algo).unwrap();
        let decompressed = decompress(&compressed, &algo).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn decompress_garbage_errors() {
        let algo = CompressionAlgo::Zstd { level: 3 };
        let result = decompress(b"not valid zstd data", &algo);
        assert!(result.is_err());
    }

    #[test]
    fn multi_frame_roundtrip() {
        // Larger than PARALLEL_CHUNK → multiple concatenated frames
        let data: Vec<u8> = (0..(PARALLEL_CHUNK * 3 + 12345))
            .map(|i| ((i * 31 + 7) % 251) as u8)
            .collect();
        let algo = CompressionAlgo::Zstd { level: 3 };
        let compressed = compress(&data, &algo).unwrap();

        // Parallel path
        let decompressed = decompress(&compressed, &algo).unwrap();
        assert_eq!(data, decompressed);

        // Streaming fallback path must also handle concatenated frames
        let streamed = zstd::decode_all(std::io::Cursor::new(&compressed[..])).unwrap();
        assert_eq!(data, streamed);
    }

    #[test]
    fn legacy_streaming_frame_roundtrips() {
        // Data compressed the old way (streaming, no content size in header)
        let data = vec![7u8; 5_000_000];
        let compressed = zstd::encode_all(std::io::Cursor::new(&data[..]), 3).unwrap();
        let algo = CompressionAlgo::Zstd { level: 3 };
        let decompressed = decompress(&compressed, &algo).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn prefix_decompression() {
        let data: Vec<u8> = (0..2_000_000).map(|i| (i % 256) as u8).collect();
        let algo = CompressionAlgo::Zstd { level: 3 };
        let compressed = compress(&data, &algo).unwrap();

        let prefix = decompress_prefix(&compressed, &algo, 65_536).unwrap();
        assert_eq!(prefix.len(), 65_536);
        assert_eq!(&prefix[..], &data[..65_536]);

        // Truncated compressed window still yields a usable prefix
        let window = &compressed[..compressed.len().min(256 * 1024)];
        let prefix2 = decompress_prefix(window, &algo, 65_536).unwrap();
        assert!(!prefix2.is_empty());
        assert_eq!(&prefix2[..], &data[..prefix2.len()]);
    }
}
