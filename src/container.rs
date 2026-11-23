//! Self-describing container format for `rcn` (`RCN1`).
//!
//! v1 layout: `[MAGIC(4)][Header(7)][optional global dict][BlockEntry * num_blocks (13 each)][payloads…]`.
//! v2 layout: same, plus a member table after the optional global dict:
//! `[u32 member_count][Member × N][BlockEntry × num_blocks][payloads…]`.
//!
//! Each block payload is described by its `BlockEntry` (compressed length, original length,
//! method, CRC32 of the *original* block).
//!
//! Global XWRT dictionary: when present (flags bit 0), the dictionary is stored
//! right after the header and used for all Text blocks instead of per-block dictionaries.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crc32fast::Hasher;
use std::io::{Cursor, Read};

pub const MAGIC: &[u8; 4] = b"RCN1";
/// Single-stream container (no member table).
pub const VERSION: u8 = 1;
/// Named-member archive (member table after the optional global dict).
pub const VERSION_V2: u8 = 2;

/// Flag: global XWRT dictionary present in container.
pub const FLAG_GLOBAL_DICT: u8 = 0x01;

/// Maximum UTF-8 byte length of a stored member path.
pub const MAX_MEMBER_NAME_LEN: usize = 4096;

/// Container header (7 bytes after the 4-byte magic).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub version: u8,
    pub flags: u8,
    pub block_size_log: u8,
    pub num_blocks: u32,
}

/// Per-block record (13 bytes): tells the decoder how to find and validate the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockEntry {
    pub comp_len: u32,
    pub orig_len: u32,
    pub method: u8,
    pub crc32: u32,
}

/// One named file in a v2 archive. `first_block` / `num_blocks` index the
/// container's `BlockEntry` table (contiguous range; `num_blocks == 0` is empty).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Member {
    pub name: String,
    pub first_block: u32,
    pub num_blocks: u32,
}

impl Header {
    /// Serialize the header (including magic) onto `w`.
    ///
    /// # Panics
    ///
    /// Never panics in practice: writing to an in-memory `Vec` cannot fail. The `.unwrap()`
    /// is only to satisfy the `byteorder` `io::Result` contract.
    pub fn write(&self, w: &mut Vec<u8>) {
        w.extend_from_slice(MAGIC);
        w.push(self.version);
        w.push(self.flags);
        w.push(self.block_size_log);
        w.write_u32::<LittleEndian>(self.num_blocks)
            .expect("writing to an in-memory Vec cannot fail");
    }

    /// Parse a header (and consume the magic) from `r`.
    ///
    /// # Errors
    ///
    /// Returns an I/O error on truncation or a bad magic.
    pub fn read(r: &mut Cursor<&[u8]>) -> std::io::Result<Self> {
        let mut m = [0u8; 4];
        r.read_exact(&mut m)?;
        if &m != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad rcn magic",
            ));
        }
        Ok(Self {
            version: r.read_u8()?,
            flags: r.read_u8()?,
            block_size_log: r.read_u8()?,
            num_blocks: r.read_u32::<LittleEndian>()?,
        })
    }
}

impl BlockEntry {
    /// Serialize this entry (13 bytes).
    ///
    /// # Panics
    ///
    /// Never panics in practice: writing to an in-memory `Vec` cannot fail.
    pub fn write(&self, w: &mut Vec<u8>) {
        w.write_u32::<LittleEndian>(self.comp_len)
            .expect("writing to an in-memory Vec cannot fail");
        w.write_u32::<LittleEndian>(self.orig_len)
            .expect("writing to an in-memory Vec cannot fail");
        w.push(self.method);
        w.write_u32::<LittleEndian>(self.crc32)
            .expect("writing to an in-memory Vec cannot fail");
    }

    /// Parse an entry from `r`.
    ///
    /// # Errors
    ///
    /// Returns an I/O error on truncation.
    pub fn read(r: &mut Cursor<&[u8]>) -> std::io::Result<Self> {
        Ok(Self {
            comp_len: r.read_u32::<LittleEndian>()?,
            orig_len: r.read_u32::<LittleEndian>()?,
            method: r.read_u8()?,
            crc32: r.read_u32::<LittleEndian>()?,
        })
    }
}

/// Read a global XWRT dictionary from `data` starting at `offset`.
///
/// Returns `(dict_bytes, new_offset)` if present (flags bit 0 set),
/// or `(Vec::new(), offset)` if not present.
///
/// # Errors
///
/// Returns [`RcnError::InvalidContainer`] when the flag is set but the length
/// prefix is truncated or the dictionary body overruns `data`.
pub fn read_global_dict(
    data: &[u8],
    offset: usize,
    flags: u8,
) -> Result<(Vec<u8>, usize), crate::error::RcnError> {
    if (flags & FLAG_GLOBAL_DICT) == 0 {
        return Ok((Vec::new(), offset));
    }
    if offset + 4 > data.len() {
        return Err(crate::error::RcnError::InvalidContainer(
            "truncated global dictionary length prefix".into(),
        ));
    }
    let dict_len = u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as usize;
    let new_offset = offset + 4;
    if new_offset + dict_len > data.len() {
        return Err(crate::error::RcnError::InvalidContainer(format!(
            "global dictionary length {dict_len} overruns container"
        )));
    }
    Ok((
        data[new_offset..new_offset + dict_len].to_vec(),
        new_offset + dict_len,
    ))
}

/// Write a global XWRT dictionary to `out`.
/// Returns the number of bytes written (4 for length prefix + dict_bytes).
pub fn write_global_dict(out: &mut Vec<u8>, dict_bytes: &[u8]) -> usize {
    out.extend_from_slice(&(dict_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(dict_bytes);
    4 + dict_bytes.len()
}

/// Validate a stored member path: relative, `/`-separated, no `..` / `.` /
/// empty components, no NUL, no `\`, not absolute.
///
/// # Errors
///
/// Returns [`RcnError::InvalidContainer`] when `name` is not a safe archive path.
pub fn validate_member_name(name: &str) -> Result<(), crate::error::RcnError> {
    if name.is_empty() {
        return Err(crate::error::RcnError::InvalidContainer(
            "empty member name".into(),
        ));
    }
    if name.len() > MAX_MEMBER_NAME_LEN {
        return Err(crate::error::RcnError::InvalidContainer(format!(
            "member name exceeds {MAX_MEMBER_NAME_LEN} bytes"
        )));
    }
    if name.starts_with('/') {
        return Err(crate::error::RcnError::InvalidContainer(
            "absolute member path".into(),
        ));
    }
    if name.contains('\0') {
        return Err(crate::error::RcnError::InvalidContainer(
            "member name contains NUL".into(),
        ));
    }
    if name.contains('\\') {
        return Err(crate::error::RcnError::InvalidContainer(
            "member name contains backslash".into(),
        ));
    }
    for part in name.split('/') {
        if part.is_empty() || part == "." || part == ".." {
            return Err(crate::error::RcnError::InvalidContainer(format!(
                "unsafe member path component in '{name}'"
            )));
        }
    }
    Ok(())
}

/// Serialize a v2 member table onto `out`.
///
/// # Errors
///
/// Returns [`RcnError::InvalidContainer`] when a name is invalid or longer than
/// [`u16::MAX`] / [`MAX_MEMBER_NAME_LEN`].
pub fn write_members(
    out: &mut Vec<u8>,
    members: &[Member],
) -> Result<usize, crate::error::RcnError> {
    let start = out.len();
    out.extend_from_slice(&(members.len() as u32).to_le_bytes());
    for m in members {
        validate_member_name(&m.name)?;
        let name_bytes = m.name.as_bytes();
        if name_bytes.len() > u16::MAX as usize {
            return Err(crate::error::RcnError::InvalidContainer(
                "member name exceeds u16 length".into(),
            ));
        }
        out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(&m.first_block.to_le_bytes());
        out.extend_from_slice(&m.num_blocks.to_le_bytes());
    }
    Ok(out.len() - start)
}

/// Read a v2 member table from `data` starting at `offset`.
///
/// Returns `(members, new_offset)`.
///
/// # Errors
///
/// Returns [`RcnError::InvalidContainer`] on truncation, a bad name, or a
/// name longer than [`MAX_MEMBER_NAME_LEN`].
pub fn read_members(
    data: &[u8],
    offset: usize,
) -> Result<(Vec<Member>, usize), crate::error::RcnError> {
    if offset + 4 > data.len() {
        return Err(crate::error::RcnError::InvalidContainer(
            "truncated member count".into(),
        ));
    }
    let count = u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as usize;
    let mut off = offset + 4;
    let mut members = Vec::with_capacity(count);
    for _ in 0..count {
        if off + 2 > data.len() {
            return Err(crate::error::RcnError::InvalidContainer(
                "truncated member name length".into(),
            ));
        }
        let name_len = u16::from_le_bytes([data[off], data[off + 1]]) as usize;
        off += 2;
        if name_len == 0 || name_len > MAX_MEMBER_NAME_LEN {
            return Err(crate::error::RcnError::InvalidContainer(format!(
                "invalid member name length {name_len}"
            )));
        }
        if off + name_len + 8 > data.len() {
            return Err(crate::error::RcnError::InvalidContainer(
                "truncated member record".into(),
            ));
        }
        let name_bytes = &data[off..off + name_len];
        off += name_len;
        let name = std::str::from_utf8(name_bytes).map_err(|_| {
            crate::error::RcnError::InvalidContainer("member name is not UTF-8".into())
        })?;
        validate_member_name(name)?;
        let first_block = u32::from_le_bytes([
            data[off],
            data[off + 1],
            data[off + 2],
            data[off + 3],
        ]);
        off += 4;
        let num_blocks = u32::from_le_bytes([
            data[off],
            data[off + 1],
            data[off + 2],
            data[off + 3],
        ]);
        off += 4;
        members.push(Member {
            name: name.to_string(),
            first_block,
            num_blocks,
        });
    }
    Ok((members, off))
}

/// CRC32 of `buf` (used to validate decompressed blocks against corruption).
#[must_use]
pub fn crc32(buf: &[u8]) -> u32 {
    let mut h = Hasher::new();
    h.update(buf);
    h.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_roundtrip() {
        let header = Header {
            version: VERSION,
            flags: 0,
            block_size_log: 16,
            num_blocks: 2,
        };
        let entries = [
            BlockEntry {
                comp_len: 10,
                orig_len: 100,
                method: 1,
                crc32: 0x1234_5678,
            },
            BlockEntry {
                comp_len: 20,
                orig_len: 200,
                method: 0,
                crc32: 0x9abc_def0,
            },
        ];
        let mut buf = Vec::new();
        header.write(&mut buf);
        for e in &entries {
            e.write(&mut buf);
        }
        let mut cur = Cursor::new(buf.as_slice());
        let got_h = Header::read(&mut cur).expect("header read");
        assert_eq!(got_h, header);
        let got_e0 = BlockEntry::read(&mut cur).expect("entry 0");
        let got_e1 = BlockEntry::read(&mut cur).expect("entry 1");
        assert_eq!(got_e0, entries[0]);
        assert_eq!(got_e1, entries[1]);
    }

    #[test]
    fn crc32_is_deterministic() {
        assert_eq!(crc32(b"rcn"), crc32(b"rcn"));
        assert_ne!(crc32(b"rcn"), crc32(b"xxx"));
    }

    #[test]
    fn bad_magic_errors() {
        let buf = b"XXXX\x01\x00\x10\x00\x00\x00\x00";
        let mut cur = Cursor::new(buf.as_slice());
        assert!(Header::read(&mut cur).is_err());
    }

    #[test]
    fn read_global_dict_truncated_length_prefix_errors() {
        let err = read_global_dict(&[0, 1, 2], 0, FLAG_GLOBAL_DICT).unwrap_err();
        assert!(matches!(err, crate::error::RcnError::InvalidContainer(_)));
    }

    #[test]
    fn read_global_dict_overlong_dict_len_errors() {
        let mut data = vec![10, 0, 0, 0]; // claims 10 bytes
        data.extend_from_slice(b"short"); // only 5
        let err = read_global_dict(&data, 0, FLAG_GLOBAL_DICT).unwrap_err();
        assert!(matches!(err, crate::error::RcnError::InvalidContainer(_)));
    }

    #[test]
    fn read_global_dict_absent_flag_is_empty() {
        let (bytes, off) = read_global_dict(b"xxxx", 2, 0).expect("ok");
        assert!(bytes.is_empty());
        assert_eq!(off, 2);
    }

    #[test]
    fn global_dict_flag_roundtrip_preserves_bytes() {
        let dict = b"hello-dict";
        let mut buf = Vec::new();
        write_global_dict(&mut buf, dict);
        let (got, end) = read_global_dict(&buf, 0, FLAG_GLOBAL_DICT).expect("read");
        assert_eq!(got, dict);
        assert_eq!(end, buf.len());
    }

    #[test]
    fn member_table_roundtrip() {
        let members = vec![
            Member {
                name: "input.bin".into(),
                first_block: 0,
                num_blocks: 2,
            },
            Member {
                name: "mydir/a.txt".into(),
                first_block: 2,
                num_blocks: 1,
            },
            Member {
                name: "mydir/empty.dat".into(),
                first_block: 0,
                num_blocks: 0,
            },
        ];
        let mut buf = Vec::new();
        write_members(&mut buf, &members).expect("write");
        let (got, end) = read_members(&buf, 0).expect("read");
        assert_eq!(got, members);
        assert_eq!(end, buf.len());
    }

    #[test]
    fn validate_member_name_rejects_unsafe_paths() {
        assert!(validate_member_name("input.bin").is_ok());
        assert!(validate_member_name("mydir/a.txt").is_ok());
        assert!(validate_member_name("").is_err());
        assert!(validate_member_name("/abs").is_err());
        assert!(validate_member_name("..").is_err());
        assert!(validate_member_name("../x").is_err());
        assert!(validate_member_name("a/../b").is_err());
        assert!(validate_member_name("a/./b").is_err());
        assert!(validate_member_name("a//b").is_err());
        assert!(validate_member_name("a\\b").is_err());
        assert!(validate_member_name("a\0b").is_err());
    }

    #[test]
    fn read_members_rejects_dotdot_name() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u32.to_le_bytes());
        let name = b"../etc/passwd";
        buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        assert!(read_members(&buf, 0).is_err());
    }

    #[test]
    fn read_members_truncated_count_errors() {
        let err = read_members(&[0, 1], 0).unwrap_err();
        assert!(matches!(err, crate::error::RcnError::InvalidContainer(_)));
    }
}
